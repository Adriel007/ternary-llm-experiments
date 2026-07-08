
from __future__ import annotations

import json
import math
import os
import sys
import time
import traceback

ROOT = "/content/PhD-propose"
sys.path[:0] = [ROOT, os.path.join(ROOT, "packages/bitnet_core/src")]

import numpy as np  
import torch  
import torch.nn.functional as TF  

from experiments.interp.eap_ig_2b import MID, OUT, DRIVE  
from experiments.interp.h2_2b_correction import (  
    SEQ_LEN, BATCH, N_TRAIN_SEQS, N_VAL_SEQS, RANK, ALPHA, LR, WD,
    MAX_STEPS, EVAL_EVERY, PATIENCE, BUDGETS,
    install_unified_forward, attach_adapters, set_layers_active, set_all_prec,
    _token_pool, eval_loss, ppl,
)

N_TEST_SEQS = 512          
SKIP_SEQS = 96             
SEEDS = [0, 1, 2]          
N_RANDOM = 10              
SEED = 0                   

HARDEN_DIR = os.path.join(DRIVE, "h2_2b_harden_checkpoints")
HARDEN_CKPT = os.path.join(HARDEN_DIR, "h2_2b_harden_state.json")
HARDEN_LOCAL = os.path.join(OUT, "h2_2b_hardening.json")
RUN2_LOCAL = os.path.join(ROOT, "reports/data/h2_2b_correction.json")
RUN2_DRIVE = os.path.join(DRIVE, "h2_2b_corr_checkpoints", "h2_2b_corr_state.json")

CFG = {
    "model": MID, "seq_len": SEQ_LEN, "batch": BATCH,
    "n_train_seqs": N_TRAIN_SEQS, "n_val_seqs": N_VAL_SEQS, "n_test_seqs": N_TEST_SEQS,
    "skip_seqs": SKIP_SEQS, "rank": RANK, "alpha": ALPHA, "lr": LR, "wd": WD,
    "max_steps": MAX_STEPS, "eval_every": EVAL_EVERY, "patience": PATIENCE,
    "budgets": BUDGETS, "seeds": SEEDS, "n_random": N_RANDOM, "master_seed": SEED,
    "reuses_rankings_from": "h2_2b_correction.py run 2 (s_l/c_l/oracle ranking; test-independent)",
    "mechanism": "joint per-allocation LoRA correction, B=0 no-op init, backbone frozen, early-stop on val",
    "calib_corpus": "Salesforce/wikitext wikitext-103-raw-v1 (streaming, train split)",
}

def _atomic_dump(obj, path):
    tmp = path + ".tmp"
    json.dump(obj, open(tmp, "w"), indent=2)
    os.replace(tmp, path)

def _load_state():
    for p in (HARDEN_CKPT, HARDEN_LOCAL):
        if os.path.exists(p):
            try:
                return json.load(open(p))
            except Exception:
                pass
    return {}

def _save_state(state):
    os.makedirs(HARDEN_DIR, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)
    _atomic_dump(state, HARDEN_CKPT)
    _atomic_dump(state, HARDEN_LOCAL)

def _load_run2():
    for p in (RUN2_LOCAL, RUN2_DRIVE):
        if os.path.exists(p):
            return json.load(open(p))
    raise FileNotFoundError("run-2 state not found at %s or %s" % (RUN2_LOCAL, RUN2_DRIVE))

def build_corpora(tok, device):
    total = N_TRAIN_SEQS + N_VAL_SEQS + SKIP_SEQS + N_TEST_SEQS
    pool = _token_pool(tok, total, SEQ_LEN)
    train = pool[:N_TRAIN_SEQS]
    val = pool[N_TRAIN_SEQS:N_TRAIN_SEQS + N_VAL_SEQS]
    test = pool[N_TRAIN_SEQS + N_VAL_SEQS + SKIP_SEQS:]
    seen = {}
    for nm, sl in (("train", train), ("val", val), ("test", test)):
        for row in sl.tolist():
            key = tuple(row)
            assert key not in seen, f"leakage: a sequence appears in both {seen.get(key)} and {nm}"
            seen[key] = nm
    mk = lambda ids: [ids[i:i + BATCH].to(device) for i in range(0, ids.shape[0], BATCH)]
    return mk(train), mk(val), mk(test)

@torch.no_grad()
def per_seq_losses(model, batches):
    out = []
    for b in batches:
        logits = model(input_ids=b).logits          
        sl = logits[:, :-1, :].float()
        lab = b[:, 1:]
        ce = TF.cross_entropy(sl.reshape(-1, sl.size(-1)), lab.reshape(-1), reduction="none")
        out.append(ce.view(b.size(0), -1).mean(dim=1).cpu().numpy())
    return np.concatenate(out)

def _point(losses):
    m = float(losses.mean())
    sem = float(losses.std(ddof=1) / math.sqrt(len(losses)))
    p = math.exp(m)
    return {"loss": m, "loss_sem": sem, "ppl": p,
            "ppl_lo": math.exp(m - 1.96 * sem), "ppl_hi": math.exp(m + 1.96 * sem)}

def joint_train_eval(model, per_layer_params, layers, train_b, val_b, test_b, seed):
    for li in layers:
        for (A, B) in per_layer_params[li]:
            B.zero_()
    flat = [p for li in layers for AB in per_layer_params[li] for p in AB]
    for p in flat:
        p.requires_grad_(True)
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(flat, lr=LR, weight_decay=WD)
    set_all_prec(model, "ternary"); set_layers_active(model, layers)
    nb, best, bad, best_state = len(train_b), float("inf"), 0, None
    for step in range(1, MAX_STEPS + 1):
        b = train_b[(step - 1) % nb]
        loss = model(input_ids=b, labels=b).loss
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if step % EVAL_EVERY == 0 or step == MAX_STEPS:
            vl = eval_loss(model, val_b)
            if vl < best - 1e-4:
                best, bad, best_state = vl, 0, [p.detach().clone() for p in flat]
            else:
                bad += 1
                if bad >= PATIENCE:
                    break
    if best_state is not None:
        with torch.no_grad():
            for p, pb in zip(flat, best_state):
                p.copy_(pb)
    losses = per_seq_losses(model, test_b)
    for p in flat:
        p.requires_grad_(False)
    set_layers_active(model, [])
    return losses

def sweep_deterministic(model, per_layer_params, train_b, val_b, test_b, rankings, state):
    det = {k: rankings[k] for k in ("s_hawq", "c_causal", "oracle")}
    state.setdefault("deterministic", {})
    cache = {}                                            
    
    for name, byseed in state["deterministic"].items():
        for seed_s, byB in byseed.items():
            for B, d in byB.items():
                cache[(int(seed_s), frozenset(d["layers"]))] = d
    for name, order in det.items():
        state["deterministic"].setdefault(name, {})
        for seed in SEEDS:
            state["deterministic"][name].setdefault(str(seed), {})
            for B in BUDGETS:
                if str(B) in state["deterministic"][name][str(seed)]:
                    continue
                t = time.time()
                layers = sorted(order[:B]); key = (seed, frozenset(layers))
                if key in cache:
                    pt = dict(cache[key])
                else:
                    losses = joint_train_eval(model, per_layer_params, layers,
                                              train_b, val_b, test_b, SEED * 9973 + seed * 101 + B)
                    pt = _point(losses); pt["layers"] = layers
                    cache[key] = pt
                state["deterministic"][name][str(seed)][str(B)] = pt
                _save_state(state)
                print("  [det] %-9s seed=%d B=%2d %s -> ppl=%.3f [%.3f,%.3f] (%.1fs)" %
                      (name, seed, B, layers, pt["ppl"], pt["ppl_lo"], pt["ppl_hi"], time.time() - t))
    set_all_prec(model, "ternary"); set_layers_active(model, [])

def sweep_random(model, per_layer_params, train_b, val_b, test_b, n_layers, state):
    rng = np.random.default_rng(SEED)
    state.setdefault("random", {})
    for B in BUDGETS:
        state["random"].setdefault(str(B), [])
        
        subsets = [sorted(int(x) for x in rng.choice(n_layers, size=B, replace=False))
                   for _ in range(N_RANDOM)]
        done = len(state["random"][str(B)])
        for k in range(done, N_RANDOM):
            t = time.time()
            layers = subsets[k]
            losses = joint_train_eval(model, per_layer_params, layers,
                                      train_b, val_b, test_b, SEED * 7919 + B * 31 + k)
            pt = _point(losses); pt["layers"] = layers
            state["random"][str(B)].append(pt)
            _save_state(state)
            print("  [rand] B=%2d k=%d/%d %s -> ppl=%.3f [%.3f,%.3f] (%.1fs)" %
                  (B, k + 1, N_RANDOM, layers, pt["ppl"], pt["ppl_lo"], pt["ppl_hi"], time.time() - t))
    set_all_prec(model, "ternary"); set_layers_active(model, [])

def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    status = os.path.join(OUT, "H2_2B_HARDEN_STATUS.txt")
    open(status, "w").write("LOADING\n")
    t0 = time.time()
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        tok = AutoTokenizer.from_pretrained(MID)
        model = AutoModelForCausalLM.from_pretrained(MID, dtype=torch.bfloat16).to(dev).eval()
        model.requires_grad_(False)
        n_layers = len(model.model.layers)
        install_unified_forward()
        per_layer_params, extra_frac = attach_adapters(model, RANK, ALPHA, dev, SEED)
        print("loaded", MID, "| layers", n_layers, "| extra-mem/layer = %.4f" % extra_frac)

        
        
        run2 = _load_run2()
        for k in ("n_train_seqs", "n_val_seqs", "rank", "alpha", "lr", "wd", "max_steps", "seq_len", "batch"):
            if run2["config"][k] != CFG[k]:
                raise RuntimeError("run-2 cfg %s=%r != this run %r -> rankings/optimisation not comparable"
                                   % (k, run2["config"][k], CFG[k]))
        rankings = {k: [int(x) for x in run2["rankings"][k]] for k in ("s_hawq", "c_causal", "oracle")}
        print("reused rankings from run 2 | s_hawq[:5]=%s c_causal[:5]=%s oracle[:5]=%s"
              % (rankings["s_hawq"][:5], rankings["c_causal"][:5], rankings["oracle"][:5]))

        state = _load_state()
        if "config" in state:
            saved = {k: state["config"].get(k) for k in CFG}
            if saved != CFG:
                raise RuntimeError("checkpoint config mismatch.\n loaded %s\n cur %s" % (saved, CFG))
        state["config"] = dict(CFG)
        state["extra_mem_frac_per_layer"] = extra_frac
        state["rankings_reused"] = rankings
        _save_state(state)

        train_b, val_b, test_b = build_corpora(tok, dev)
        n_test = len(test_b) * BATCH
        print("corpora (disjoint verified): train %d | val %d | NEW test %d seqs"
              % (len(train_b) * BATCH, len(val_b) * BATCH, n_test))

        open(status, "w").write("BASE\n")
        if "base" not in state:
            set_all_prec(model, "ternary"); set_layers_active(model, [])
            base = per_seq_losses(model, test_b)
            state["base"] = _point(base)
            state["base"]["n_test_seqs"] = n_test
            _save_state(state)
            
            d = abs(state["base"]["ppl"] - run2["base"]["ppl_ternary_held"])
            print("base | ternary held-out ppl=%.3f [%.3f,%.3f] (run-2 64-seq was %.3f; |Δ|=%.3f)"
                  % (state["base"]["ppl"], state["base"]["ppl_lo"], state["base"]["ppl_hi"],
                     run2["base"]["ppl_ternary_held"], d))
        b = state["base"]
        print("base ternary test ppl=%.3f ± CI[%.3f,%.3f] on %d seqs" %
              (b["ppl"], b["ppl_lo"], b["ppl_hi"], b["n_test_seqs"]))

        open(status, "w").write("DET\n")
        print("=== (a) deterministic fronts: s_hawq/c_causal/oracle × %d seeds (joint, 512-test) ===" % len(SEEDS))
        sweep_deterministic(model, per_layer_params, train_b, val_b, test_b, rankings, state)

        open(status, "w").write("RANDOM\n")
        print("=== (a) random baseline: %d independent subsets per budget (layer-choice law) ===" % N_RANDOM)
        sweep_random(model, per_layer_params, train_b, val_b, test_b, n_layers, state)

        summary = {"budgets": BUDGETS, "base_ppl": b["ppl"]}
        for name in ("s_hawq", "c_causal", "oracle"):
            byseed = state["deterministic"][name]
            rows = {}
            for B in BUDGETS:
                ppls = [byseed[str(s)][str(B)]["ppl"] for s in SEEDS]
                rows[str(B)] = {"ppl_mean": float(np.mean(ppls)), "ppl_sd": float(np.std(ppls, ddof=1)),
                                "ppl_seeds": [float(x) for x in ppls]}
            summary[name] = rows
        rnd = {}
        for B in BUDGETS:
            ppls = [p["ppl"] for p in state["random"][str(B)]]
            rnd[str(B)] = {"ppl_median": float(np.median(ppls)), "ppl_p05": float(np.percentile(ppls, 5)),
                           "ppl_p95": float(np.percentile(ppls, 95)), "ppl_mean": float(np.mean(ppls)),
                           "ppl_all": [float(x) for x in ppls]}
        summary["random"] = rnd
        
        c_beats_rand = sum(summary["c_causal"][str(B)]["ppl_mean"] < rnd[str(B)]["ppl_median"] for B in BUDGETS)
        c_beats_s = sum(summary["c_causal"][str(B)]["ppl_mean"] < summary["s_hawq"][str(B)]["ppl_mean"] for B in BUDGETS)
        summary["c_beats_random_median_count"] = [c_beats_rand, len(BUDGETS)]
        summary["c_beats_shawq_count"] = [c_beats_s, len(BUDGETS)]
        state["summary"] = summary
        state["wall_s"] = time.time() - t0
        _save_state(state)

        print("\n=== SUMMARY (test ppl; base %.3f) ===" % b["ppl"])
        print("  B     s_hawq(mean±sd)   c_causal(mean±sd)   oracle(mean±sd)   random(med[p05,p95])")
        for B in BUDGETS:
            s, c, o = (summary[n][str(B)] for n in ("s_hawq", "c_causal", "oracle"))
            r = rnd[str(B)]
            print("  %2d   %5.2f±%.2f       %5.2f±%.2f         %5.2f±%.2f       %5.2f[%.2f,%.2f]" %
                  (B, s["ppl_mean"], s["ppl_sd"], c["ppl_mean"], c["ppl_sd"],
                   o["ppl_mean"], o["ppl_sd"], r["ppl_median"], r["ppl_p05"], r["ppl_p95"]))
        print("c_causal beats random-median at %d/%d budgets; beats s_hawq at %d/%d budgets"
              % (c_beats_rand, len(BUDGETS), c_beats_s, len(BUDGETS)))
        open(status, "w").write("DONE %.0fs\n" % (time.time() - t0))
        print("DONE %.1fs" % (time.time() - t0))
    except Exception:
        tb = traceback.format_exc()
        print(tb)
        open(status, "w").write("ERROR\n" + tb)
        raise

if __name__ == "__main__":
    main()
