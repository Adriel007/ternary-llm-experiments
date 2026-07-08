
from __future__ import annotations

import json
import os
import sys
import time
import traceback

ROOT = "/content/PhD-propose"
sys.path[:0] = [ROOT, os.path.join(ROOT, "packages/bitnet_core/src")]

import numpy as np  

from experiments.dynamics.data import TokenLoader, build_tinystories_tokens  
from experiments.dynamics.experiment import PoCConfig, train_one  

GUIDED = [0, 2, 3, 7, 1, 4, 5, 6]   
REVERSE = GUIDED[::-1]              

CFG = PoCConfig(
    seeds=(0,), steps=1500, batch_size=16, seq_len=128, warmup=100,
    eval_every=300, log_every=100, eval_batches=20,
    n_train_tokens=6_000_000, n_val_tokens=300_000,
    out_dir=os.path.join(ROOT, "artifacts/poc"),
)

def main() -> None:
    out = CFG.out_dir
    os.makedirs(out, exist_ok=True)
    status = os.path.join(out, "MP_STATUS.txt")
    open(status, "w").write("RUNNING\n")
    t0 = time.time()
    try:
        tr, va = build_tinystories_tokens(CFG.data_dir, CFG.n_train_tokens, CFG.n_val_tokens)
        loaders = (TokenLoader(tr, CFG.seq_len, CFG.device), TokenLoader(va, CFG.seq_len, CFG.device))

        def fit(quantize, seed, fp_layers=None):
            r = train_one(quantize, seed, CFG, loaders, fp_layers=fp_layers)
            del r["model"]
            print(f"  seed{seed} q={quantize} fp_layers={sorted(fp_layers) if fp_layers else []} "
                  f"-> val={r['final_val_loss']:.4f} ({r['wall_s']:.0f}s)", flush=True)
            return r["final_val_loss"]

        rng = np.random.default_rng(0)
        res = {"config": {"steps": CFG.steps, "guided_order": GUIDED}, "seed0": {}, "headline": {}}

        res["seed0"]["full_ternary_k0"] = fit(True, 0, None)
        res["seed0"]["full_fp_k8"] = fit(False, 0, None)

        pareto = {}
        for k in (1, 2, 3):
            g = set(GUIDED[:k]); rv = set(REVERSE[:k])
            rnd = set(int(x) for x in rng.choice(8, size=k, replace=False))
            entry = {
                "guided_layers": sorted(g), "guided": fit(True, 0, g),
                "reverse_layers": sorted(rv), "reverse": fit(True, 0, rv),
            }
            if k == 1:
                entry["random_layers"] = sorted(rnd); entry["random"] = fit(True, 0, rnd)
            pareto[str(k)] = entry
        res["seed0"]["pareto"] = pareto

        for seed in (0, 1, 2):
            if seed == 0:  
                g1 = pareto["1"]["guided"]; r1 = pareto["1"]["reverse"]
            else:
                g1 = fit(True, seed, {0}); r1 = fit(True, seed, {6})
            res["headline"][str(seed)] = {"guided_L0": g1, "reverse_L6": r1, "gap": r1 - g1}

        gaps = [res["headline"][str(s)]["gap"] for s in (0, 1, 2)]
        res["headline_summary"] = {"gap_mean": float(np.mean(gaps)), "gap_std": float(np.std(gaps)), "n": 3}

        json.dump(res, open(os.path.join(out, "mp.json"), "w"), indent=2)
        open(status, "w").write(f"DONE {time.time()-t0:.0f}s\n")
        print("MP DONE", round(time.time() - t0, 1), "s")
        print("headline gap (reverse-L6 minus guided-L0):", round(res["headline_summary"]["gap_mean"], 4),
              "+/-", round(res["headline_summary"]["gap_std"], 4))
    except Exception:
        tb = traceback.format_exc(); print(tb)
        open(status, "w").write("ERROR\n" + tb)
        raise

if __name__ == "__main__":
    main()
