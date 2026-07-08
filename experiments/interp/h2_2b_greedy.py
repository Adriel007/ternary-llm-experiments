
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
    MAX_STEPS, EVAL_EVERY, PATIENCE,
    install_unified_forward, attach_adapters, set_layers_active, set_all_prec, _token_pool, ppl,
)

N_TEST_SEQS = 512          
SKIP_SEQS = 96             
N_SCORE_SEQS = 128         
MAX_BUDGET = 8             
CAND_STEPS = 200           
EVAL_BATCH = 32            
SEEDS_FINAL = [0, 1, 2]    
REPORT_BUDGETS = [1, 2, 3, 5, 8]   
SEED = 0

HARDEN_RUN = os.path.join(ROOT, "reports/data/h2_2b_hardening.json")  
GDIR = os.path.join(DRIVE, "h2_2b_greedy_checkpoints")
GCKPT = os.path.join(GDIR, "h2_2b_greedy_state.json")
GLOCAL = os.path.join(OUT, "h2_2b_greedy.json")

CFG = {
    "model": MID, "seq_len": SEQ_LEN, "batch": BATCH, "n_train_seqs": N_TRAIN_SEQS,
    "n_val_seqs": N_VAL_SEQS, "n_test_seqs": N_TEST_SEQS, "skip_seqs": SKIP_SEQS,
    "n_score_seqs": N_SCORE_SEQS, "max_budget": MAX_BUDGET, "cand_steps": CAND_STEPS,
    "max_steps": MAX_STEPS, "rank": RANK, "alpha": ALPHA, "lr": LR, "wd": WD,
    "seeds_final": SEEDS_FINAL, "report_budgets": REPORT_BUDGETS, "seed": SEED,
    "mechanism": "forward greedy joint per-allocation LoRA correction; candidates ranked on SCORE, path measured on TEST",
}

def _atomic_dump(obj, path):
    tmp = path + ".tmp"
    json.dump(obj, open(tmp, "w"), indent=2)
    os.replace(tmp, path)

def _load_state():
    for p in (GCKPT, GLOCAL):
        if os.path.exists(p):
            try:
                return json.load(open(p))
            except Exception:
                pass
    return {}

def _save_state(state):
    os.makedirs(GDIR, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)
    _atomic_dump(state, GCKPT)
    _atomic_dump(state, GLOCAL)

def build_corpora(tok, device):
    sizes = [N_TRAIN_SEQS, N_VAL_SEQS, SKIP_SEQS, N_TEST_SEQS, N_SCORE_SEQS]
    pool = _token_pool(tok, sum(sizes), SEQ_LEN)
    off = np.cumsum([0] + sizes)
    train, val, _, test, score = (pool[off[i]:off[i + 1]] for i in range(5))
    seen = {}
    for nm, sl in (("train", train), ("val", val), ("test", test), ("score", score)):
        for row in sl.tolist():
            key = tuple(row)
            assert key not in seen, f"leakage: sequence in both {seen.get(key)} and {nm}"
            seen[key] = nm
    train_b = [train[i:i + BATCH].to(device) for i in range(0, train.shape[0], BATCH)]
    return train_b, val.to(device), test.to(device), score.to(device)

@torch.no_grad()
def eval_mean_loss(model, ids, bs=EVAL_BATCH):
    losses = [float(model(input_ids=ids[i:i + bs], labels=ids[i:i + bs]).loss.item())
              for i in range(0, ids.shape[0], bs)]

    sizes = [min(bs, ids.shape[0] - i) for i in range(0, ids.shape[0], bs)]
    return float(np.average(losses, weights=sizes))

@torch.no_grad()
def per_seq_losses(model, ids, bs=EVAL_BATCH):
    out = []
    for i in range(0, ids.shape[0], bs):
        b = ids[i:i + bs]
        logits = model(input_ids=b).logits
        sl = logits[:, :-1, :].float()
        lab = b[:, 1:]
        ce = TF.cross_entropy(sl.reshape(-1, sl.size(-1)), lab.reshape(-1), reduction="none")
        out.append(ce.view(b.size(0), -1).mean(dim=1).cpu().numpy())
    return np.concatenate(out)

def _point(losses):
    m = float(losses.mean()); sem = float(losses.std(ddof=1) / math.sqrt(len(losses)))
    return {"loss": m, "loss_sem": sem, "ppl": math.exp(m),
            "ppl_lo": math.exp(m - 1.96 * sem), "ppl_hi": math.exp(m + 1.96 * sem)}

def joint_train(model, plp, layers, train_b, val_ids, steps, seed):
    for li in layers:
        for (A, B) in plp[li]:
            B.zero_()
    flat = [p for li in layers for AB in plp[li] for p in AB]
    for p in flat:
        p.requires_grad_(True)
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(flat, lr=LR, weight_decay=WD)
    set_all_prec(model, "ternary"); set_layers_active(model, layers)
    nb, best, bad, best_state = len(train_b), float("inf"), 0, None
    for step in range(1, steps + 1):
        b = train_b[(step - 1) % nb]
        loss = model(input_ids=b, labels=b).loss
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if step % EVAL_EVERY == 0 or step == steps:
            vl = eval_mean_loss(model, val_ids)
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
    for p in flat:
        p.requires_grad_(False)

def score_candidate(model, plp, layers, train_b, val_ids, score_ids, seed):
    joint_train(model, plp, layers, train_b, val_ids, CAND_STEPS, seed)
    sc = eval_mean_loss(model, score_ids)
    set_layers_active(model, [])
    return sc

def measure_path_point(model, plp, layers, train_b, val_ids, test_ids, seeds):
    per = []
    for s in seeds:
        joint_train(model, plp, layers, train_b, val_ids, MAX_STEPS, SEED * 31 + s)
        per.append(_point(per_seq_losses(model, test_ids)))
        set_layers_active(model, [])
    ppls = [p["ppl"] for p in per]
    return {"layers": sorted(layers), "ppl_mean": float(np.mean(ppls)),
            "ppl_sd": float(np.std(ppls, ddof=1)) if len(ppls) > 1 else 0.0,
            "ppl_seeds": [float(x) for x in ppls], "seq_ci_seed0": [per[0]["ppl_lo"], per[0]["ppl_hi"]]}

def run_greedy(model, plp, n_layers, train_b, val_ids, score_ids, state):
    state.setdefault("cand_cache", {})            
    state.setdefault("greedy_path", [])           
    cache = state["cand_cache"]
    S = list(state["greedy_path"][-1]["layers"]) if state["greedy_path"] else []
    while len(S) < MAX_BUDGET:
        k = len(S) + 1
        t = time.time()
        cands = [j for j in range(n_layers) if j not in S]
        scores = {}
        for j in cands:
            key = ",".join(map(str, sorted(S + [j])))
            if key in cache:
                scores[j] = cache[key]
            else:
                scores[j] = score_candidate(model, plp, S + [j], train_b, val_ids, score_ids, SEED)
                cache[key] = scores[j]
                _save_state(state)
        jstar = min(scores, key=scores.get)
        S = sorted(S + [jstar])
        state["greedy_path"].append({"k": k, "added": int(jstar), "layers": list(S),
                                     "score": float(scores[jstar]),
                                     "score_ppl": ppl(scores[jstar])})
        _save_state(state)
        print("  [greedy] k=%2d add L%-2d -> S=%s | score-ppl=%.3f (%d cand, %.0fs)" %
              (k, jstar, S, ppl(scores[jstar]), len(cands), time.time() - t))
    return [d["layers"] for d in state["greedy_path"]]

def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    status = os.path.join(OUT, "H2_2B_GREEDY_STATUS.txt")
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
        plp, extra = attach_adapters(model, RANK, ALPHA, dev, SEED)
        print("loaded", MID, "| layers", n_layers, "| extra/layer=%.4f" % extra)

        state = _load_state()
        if "config" in state and {k: state["config"].get(k) for k in CFG} != CFG:
            raise RuntimeError("checkpoint config mismatch")
        state["config"] = dict(CFG); state["extra_mem_frac_per_layer"] = extra
        _save_state(state)

        train_b, val_ids, test_ids, score_ids = build_corpora(tok, dev)
        print("corpora (disjoint): train %d | val %d | test %d | score %d" %
              (len(train_b) * BATCH, val_ids.shape[0], test_ids.shape[0], score_ids.shape[0]))

        open(status, "w").write("BASE\n")
        if "base" not in state:
            set_all_prec(model, "ternary"); set_layers_active(model, [])
            state["base"] = _point(per_seq_losses(model, test_ids)); _save_state(state)
        b = state["base"]
        print("base ternary test ppl=%.3f [%.3f,%.3f]" % (b["ppl"], b["ppl_lo"], b["ppl_hi"]))

        open(status, "w").write("GREEDY\n")
        print("=== forward greedy selection (rank candidates on SCORE, %d steps) -> path to B=%d ==="
              % (CAND_STEPS, MAX_BUDGET))
        path = run_greedy(model, plp, n_layers, train_b, val_ids, score_ids, state)

        open(status, "w").write("MEASURE\n")
        print("=== measure greedy path on TEST (full train, %d seeds, per-seq CI) ===" % len(SEEDS_FINAL))
        state.setdefault("greedy_pareto", {})
        for Bd in REPORT_BUDGETS:
            if Bd > MAX_BUDGET or str(Bd) in state["greedy_pareto"]:
                continue
            t = time.time()
            layers = path[Bd - 1]                       
            pt = measure_path_point(model, plp, layers, train_b, val_ids, test_ids, SEEDS_FINAL)
            state["greedy_pareto"][str(Bd)] = pt; _save_state(state)
            print("  [greedy-pareto] B=%2d %s -> ppl=%.3f±%.3f (%.0fs)" %
                  (Bd, layers, pt["ppl_mean"], pt["ppl_sd"], time.time() - t))

        cmp = {"base_ppl": b["ppl"]}
        if os.path.exists(HARDEN_RUN):
            h = json.load(open(HARDEN_RUN))["summary"]
            for Bd in REPORT_BUDGETS:
                g = state["greedy_pareto"][str(Bd)]["ppl_mean"]
                row = {"greedy": g}
                for nm in ("c_causal", "s_hawq", "oracle", "random"):
                    if nm == "random":
                        row["random_median"] = h["random"][str(Bd)]["ppl_median"]
                        row["random_p05"] = h["random"][str(Bd)]["ppl_p05"]
                    else:
                        row[nm] = h[nm][str(Bd)]["ppl_mean"]
                cmp[str(Bd)] = row
            
            cmp["greedy_beats_random_median"] = [
                sum(state["greedy_pareto"][str(Bd)]["ppl_mean"] < h["random"][str(Bd)]["ppl_median"]
                    for Bd in REPORT_BUDGETS), len(REPORT_BUDGETS)]
            cmp["greedy_beats_random_p05"] = [
                sum(state["greedy_pareto"][str(Bd)]["ppl_mean"] < h["random"][str(Bd)]["ppl_p05"]
                    for Bd in REPORT_BUDGETS), len(REPORT_BUDGETS)]
        state["comparison"] = cmp
        state["wall_s"] = time.time() - t0
        _save_state(state)

        print("\n=== GREEDY CEILING vs hardening rankings (test ppl; base %.3f) ===" % b["ppl"])
        print("  B    greedy   c_causal  s_hawq   oracle   random med[p05]")
        for Bd in REPORT_BUDGETS:
            if str(Bd) in cmp:
                r = cmp[str(Bd)]
                print("  %2d   %5.2f    %5.2f     %5.2f    %5.2f    %5.2f[%.2f]" %
                      (Bd, r["greedy"], r["c_causal"], r["s_hawq"], r["oracle"],
                       r["random_median"], r["random_p05"]))
        if "greedy_beats_random_p05" in cmp:
            print("greedy beats random-median at %d/%d ; beats random-p05 (best random draws) at %d/%d"
                  % (*cmp["greedy_beats_random_median"], *cmp["greedy_beats_random_p05"]))
        open(status, "w").write("DONE %.0fs\n" % (time.time() - t0))
        print("DONE %.1fs" % (time.time() - t0))
    except Exception:
        tb = traceback.format_exc()
        print(tb); open(status, "w").write("ERROR\n" + tb)
        raise

if __name__ == "__main__":
    main()
