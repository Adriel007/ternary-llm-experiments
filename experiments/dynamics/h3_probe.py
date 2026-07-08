
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

from experiments.dynamics import metrics as M  
from experiments.dynamics.data import TokenLoader, build_tinystories_tokens  
from experiments.dynamics.experiment import (  
    PoCConfig,
    pareto_quantize,
    quantize_sensitivity,
    relax_sensitivity,
    spearman,
    train_one,
)

CFG = PoCConfig(
    seeds=(0,), steps=2000, batch_size=16, seq_len=128, warmup=100,
    eval_every=200, log_every=50, eval_batches=20,
    n_train_tokens=6_000_000, n_val_tokens=300_000,
    hessian_iters=20, hessian_bsz=8, hessian_batches=2,
    out_dir=os.path.join(ROOT, "artifacts/poc"),
)

def main() -> None:
    out = CFG.out_dir
    os.makedirs(out, exist_ok=True)
    status = os.path.join(out, "H3_STATUS.txt")
    open(status, "w").write("RUNNING\n")
    t0 = time.time()
    try:
        tr, va = build_tinystories_tokens(CFG.data_dir, CFG.n_train_tokens, CFG.n_val_tokens)
        tl, vl = TokenLoader(tr, CFG.seq_len, CFG.device), TokenLoader(va, CFG.seq_len, CFG.device)
        loaders = (tl, vl)

        fp = train_one(False, 0, CFG, loaders); fpm = fp.pop("model")
        tern = train_one(True, 0, CFG, loaders); tm = tern.pop("model")
        torch.save(fpm.state_dict(), os.path.join(out, "fp_seed0.pt"))
        torch.save(tm.state_dict(), os.path.join(out, "ternary_seed0.pt"))
        print(f"trained: fp_val={fp['final_val_loss']:.4f} tern_val={tern['final_val_loss']:.4f}")

        n = len(fpm.model.layers)
        layers = list(range(n))

        sens = quantize_sensitivity(fpm, vl, CFG)
        ptq = [sens["delta"][l] for l in layers]

        dqfp = M.dequantized_float_copy(fpm)
        groups = M.layer_weight_groups(dqfp)
        xb = next(iter(vl.iter_eval(CFG.hessian_bsz, 1)))
        lam = [M.top_hessian_eigenvalue(dqfp, xb, params=groups[l], n_iter=CFG.hessian_iters) for l in layers]
        del dqfp

        rho = spearman(lam, ptq)

        sens_order = sorted(layers, key=lambda l: sens["delta"][l])     
        curv_order = sorted(layers, key=lambda l: lam[l])               
        pareto_sens = pareto_quantize(fpm, vl, CFG, sens_order)
        pareto_curv = pareto_quantize(fpm, vl, CFG, curv_order)
        rng = np.random.default_rng(0)
        rand = [pareto_quantize(fpm, vl, CFG, list(rng.permutation(layers))) for _ in range(10)]
        rand_mean = np.mean(rand, axis=0).tolist()
        rand_std = np.std(rand, axis=0).tolist()

        relax = relax_sensitivity(tm, vl, CFG)
        relax_delta = [relax["delta"][l] for l in layers]

        result = {
            "config": {"steps": CFG.steps, "seed": 0},
            "fp_val_loss": fp["final_val_loss"],
            "ternary_val_loss": tern["final_val_loss"],
            "fp_base_val_loss": sens["fp_base_val_loss"],
            "layers": layers,
            "ptq_sensitivity": ptq,
            "lambda_per_layer": lam,
            "spearman_lambda_ptq": rho,
            "sens_order": sens_order,
            "curv_order": curv_order,
            "pareto_sens": pareto_sens,
            "pareto_curv": pareto_curv,
            "pareto_random_mean": rand_mean,
            "pareto_random_std": rand_std,
            "relax_diag_delta": relax_delta,
            "relax_diag_base": relax["base_val_loss"],
        }
        json.dump(result, open(os.path.join(out, "h3.json"), "w"), indent=2)
        open(status, "w").write(f"DONE {time.time()-t0:.0f}s\nspearman={rho:.3f}\n")
        print("H3 DONE", round(time.time() - t0, 1), "s | spearman(lambda,ptq)=", round(rho, 3))
        print("ptq_sensitivity:", [round(x, 4) for x in ptq])
        print("lambda_per_layer:", [round(x, 1) for x in lam])
        print("pareto_sens:", [round(x, 4) for x in pareto_sens])
        print("pareto_random:", [round(x, 4) for x in rand_mean])
        print("relax_diag (should be <=0):", [round(x, 4) for x in relax_delta])
    except Exception:
        tb = traceback.format_exc(); print(tb)
        open(status, "w").write("ERROR\n" + tb)
        raise

if __name__ == "__main__":
    main()
