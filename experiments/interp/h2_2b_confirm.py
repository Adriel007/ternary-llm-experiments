
from __future__ import annotations

import json
import os
import sys
import time
import traceback

ROOT = "/content/PhD-propose"
sys.path[:0] = [ROOT, os.path.join(ROOT, "packages/bitnet_core/src")]

import numpy as np  
import torch  

from experiments.interp.eap_ig_2b import MID, OUT, DRIVE  
from experiments.interp.h2_2b_correction import (  
    SEQ_LEN, RANK, ALPHA, MAX_STEPS as _MAXSTEPS,
    install_unified_forward, attach_adapters, set_layers_active, set_all_prec, ppl,
)
from experiments.interp.h2_2b_greedy import (  
    SEED, build_corpora, joint_train, eval_mean_loss, per_seq_losses, _point,
)
from experiments.env_stamp import env_stamp  
from experiments.fidelity import assert_ppl_faithful  

N_FRESH_SEQS = 512                
FRESH_SKIP_SEQS = 2048            
REPORT_BUDGETS = [2, 5, 8]
PRIMARY_B = 5                     
SEEDS_FINAL = [0, 1, 2]
N_RANDOM_DRAWS = 3
BASE_PPL_TOL = 0.20               
FRESH_PPL_LO, FRESH_PPL_HI = 15.0, 80.0   

RUN2 = os.path.join(ROOT, "reports/data/h2_2b_correction.json")
GREEDY = os.path.join(ROOT, "reports/data/h2_2b_greedy.json")
CDIR = os.path.join(DRIVE, "h2_2b_confirm_checkpoints")
CCKPT = os.path.join(CDIR, "h2_2b_confirm_state.json")
CLOCAL = os.path.join(OUT, "h2_2b_confirm.json")

CFG = {
    "model": MID, "seq_len": SEQ_LEN, "rank": RANK, "alpha": ALPHA,
    "n_fresh_seqs": N_FRESH_SEQS, "fresh_skip_seqs": FRESH_SKIP_SEQS,
    "report_budgets": REPORT_BUDGETS, "primary_b": PRIMARY_B,
    "seeds_final": SEEDS_FINAL, "n_random_draws": N_RANDOM_DRAWS, "seed": SEED,
    "purpose": "pre-registered fresh-slice confirmation of the §3.18 headline (c_ell vs HAWQ vs random) "
               "on a disjoint wikitext 512-seq slice; tests slice-robustness of the ranking",
}

def _atomic_dump(obj, path):
    tmp = path + ".tmp"
    json.dump(obj, open(tmp, "w"), indent=2)
    os.replace(tmp, path)

def _load_state():
    for p in (CCKPT, CLOCAL):
        if os.path.exists(p):
            try:
                return json.load(open(p))
            except Exception:
                pass
    return {}

def _save_state(state):
    os.makedirs(CDIR, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)
    _atomic_dump(state, CCKPT)
    _atomic_dump(state, CLOCAL)

def _fresh_wiki_test(tok, device, old_test):
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train", streaming=True)
    eos = tok.eos_token_id
    need = (FRESH_SKIP_SEQS + N_FRESH_SEQS) * SEQ_LEN
    buf = []
    for ex in ds:
        t = ex["text"].strip()
        if not t:
            continue
        buf.extend(tok(t, add_special_tokens=False).input_ids)
        buf.append(eos)
        if len(buf) >= need:
            break
    pool = torch.tensor(buf[:need], dtype=torch.long).view(FRESH_SKIP_SEQS + N_FRESH_SEQS, SEQ_LEN)
    fresh = pool[FRESH_SKIP_SEQS:]
    old_rows = {tuple(r) for r in old_test.tolist()}
    for r in fresh.tolist():
        assert tuple(r) not in old_rows, "fresh slice leaked into the old reused test slice"
    return fresh.to(device)

def measure(model, plp, layers, train_b, val_ids, test_fresh, seeds):
    pts = []
    for s in seeds:
        joint_train(model, plp, layers, train_b, val_ids, _MAXSTEPS, SEED * 31 + s)
        pts.append(_point(per_seq_losses(model, test_fresh)))
        set_layers_active(model, [])
    ppls = [p["ppl"] for p in pts]
    return {"layers": sorted(layers),
            "ppl_mean": float(np.mean(ppls)),
            "ppl_sd": float(np.std(ppls, ddof=1)) if len(ppls) > 1 else 0.0,
            "ppl_seeds": [float(x) for x in ppls]}

def _selftest(model, test_old, test_fresh, ref_ppl):
    print("[selftest] running...")
    set_all_prec(model, "ternary"); set_layers_active(model, [])
    base_old = ppl(eval_mean_loss(model, test_old))
    base_fresh = ppl(eval_mean_loss(model, test_fresh))
    
    d = assert_ppl_faithful(base_old, ref_ppl, tol=BASE_PPL_TOL,
                            label="old-slice base-ppl ANCHOR vs greedy.json")
    assert FRESH_PPL_LO < base_fresh < FRESH_PPL_HI and np.isfinite(base_fresh), (
        "fresh-slice base ppl %.4f outside sane window (%.0f,%.0f)" % (base_fresh, FRESH_PPL_LO, FRESH_PPL_HI))
    print("[selftest] old-slice base ppl=%.4f (ref %.4f, Δ%.3f) | fresh-slice base ppl=%.4f | PASS"
          % (base_old, ref_ppl, d, base_fresh))
    return base_fresh

def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    status = os.path.join(OUT, "H2_2B_CONFIRM_STATUS.txt")
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

        train_b, val_ids, test_old, _score = build_corpora(tok, dev)   
        open(status, "w").write("FRESH\n")
        test_fresh = _fresh_wiki_test(tok, dev, test_old)
        print("corpora: train %d-b | val %d | old-test %d | fresh-test %d" %
              (len(train_b), val_ids.shape[0], test_old.shape[0], test_fresh.shape[0]))

        run2 = json.load(open(RUN2)); greedy = json.load(open(GREEDY))
        c_val = np.array([float(run2["c_l"][str(i)]) for i in range(n_layers)])
        s_trace = np.array([float(run2["s_l"][str(i)]["trace"]) for i in range(n_layers)])
        greedy_order = [d["added"] for d in greedy["greedy_path"]]
        c_order = [int(x) for x in np.argsort(-c_val)]
        s_order = [int(x) for x in np.argsort(-s_trace)]
        rankings = {"greedy": greedy_order, "c_causal": c_order, "s_hawq": s_order}

        open(status, "w").write("SELFTEST\n")
        base_fresh = _selftest(model, test_old, test_fresh, float(greedy["base"]["ppl"]))

        state = _load_state()
        if "config" in state and {k: state["config"].get(k) for k in CFG} != CFG:
            raise RuntimeError("checkpoint config mismatch")
        state["config"] = dict(CFG); state["extra_mem_frac_per_layer"] = extra
        state["env"] = env_stamp()
        state.setdefault("base", {"fresh_ppl": base_fresh})
        state.setdefault("fronts", {})
        state.setdefault("random", {})
        _save_state(state)

        open(status, "w").write("FRONTS\n")
        for name, order in rankings.items():
            for B in REPORT_BUDGETS:
                key = "%s@%d" % (name, B)
                if key in state["fronts"]:
                    continue
                t = time.time()
                layers = sorted(order[:B])
                state["fronts"][key] = measure(model, plp, layers, train_b, val_ids, test_fresh, SEEDS_FINAL)
                _save_state(state)
                r = state["fronts"][key]
                print("  [%s] %s -> fresh %.3f±%.3f (%.0fs)" %
                      (key, layers, r["ppl_mean"], r["ppl_sd"], time.time() - t))

        open(status, "w").write("RANDOM\n")
        rng = np.random.default_rng(1234)
        for d in range(N_RANDOM_DRAWS):
            key = "draw%d@%d" % (d, PRIMARY_B)
            if key in state["random"]:
                continue
            layers = sorted(int(x) for x in rng.choice(n_layers, PRIMARY_B, replace=False))
            t = time.time()
            state["random"][key] = measure(model, plp, layers, train_b, val_ids, test_fresh, [SEEDS_FINAL[0]])
            _save_state(state)
            r = state["random"][key]
            print("  [%s] %s -> fresh %.3f (%.0fs)" % (key, layers, r["ppl_mean"], time.time() - t))

        open(status, "w").write("VERDICT\n")
        def at(name, B):
            return state["fronts"]["%s@%d" % (name, B)]["ppl_mean"]
        rand_med = float(np.median([state["random"]["draw%d@%d" % (d, PRIMARY_B)]["ppl_mean"]
                                    for d in range(N_RANDOM_DRAWS)]))
        c5, h5, g5 = at("c_causal", PRIMARY_B), at("s_hawq", PRIMARY_B), at("greedy", PRIMARY_B)
        verdict = {
            "primary_b": PRIMARY_B,
            "fresh_base_ppl": base_fresh,
            "random_median@primary": rand_med,
            "c_causal@primary": c5, "s_hawq@primary": h5, "greedy@primary": g5,
            "C1_order_c_lt_hawq": c5 < h5,                 
            "C2_order_c_le_randmed": c5 <= rand_med + 1e-9,  
            "order_greedy_lt_c": g5 < c5,
        }
        state["verdict"] = verdict; state["wall_s"] = time.time() - t0
        _save_state(state)

        print("\n=== FRESH-SLICE CONFIRM (fresh wikitext test ppl; primary B=%d) ===" % PRIMARY_B)
        print("  ranking@B        fresh-ppl")
        for name in ("greedy", "c_causal", "s_hawq"):
            for B in REPORT_BUDGETS:
                r = state["fronts"]["%s@%d" % (name, B)]
                print("  %-12s   %6.3f±%.3f" % ("%s@%d" % (name, B), r["ppl_mean"], r["ppl_sd"]))
        print("  %-12s   %6.3f   (no correction)" % ("base", base_fresh))
        print("  %-12s   %6.3f   (median of %d draws)" % ("random@%d" % PRIMARY_B, rand_med, N_RANDOM_DRAWS))
        print("PRE-REGISTERED @B=%d: C1 c<HAWQ = %s (c %.3f vs HAWQ %.3f) | C2 c<=rand-med = %s (rand-med %.3f) "
              "| greedy<c = %s" % (PRIMARY_B, verdict["C1_order_c_lt_hawq"], c5, h5,
                                   verdict["C2_order_c_le_randmed"], rand_med, verdict["order_greedy_lt_c"]))
        open(status, "w").write("DONE %.0fs\n" % (time.time() - t0))
        print("DONE %.1fs" % (time.time() - t0))
    except Exception:
        tb = traceback.format_exc()
        print(tb); open(status, "w").write("ERROR\n" + tb)
        raise

if __name__ == "__main__":
    main()
