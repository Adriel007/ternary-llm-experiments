
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
    quantize_sensitivity,
    spearman,
    train_one,
)

CFG = PoCConfig(
    seeds=(0, 1, 2), steps=2000, batch_size=16, seq_len=128, warmup=100,
    eval_every=200, log_every=50, eval_batches=20,
    n_train_tokens=6_000_000, n_val_tokens=300_000,
    hessian_iters=20, hessian_bsz=8, hessian_batches=2,
    out_dir=os.path.join(ROOT, "artifacts/poc"),
)
DRIVE = "/content/drive/MyDrive/PhD_PoC"

def _per_layer_lambda(fpm, vl, cfg: PoCConfig) -> list[float]:
    dq = M.dequantized_float_copy(fpm)
    groups = M.layer_weight_groups(dq)
    xb = next(iter(vl.iter_eval(cfg.hessian_bsz, 1)))
    lam = [M.top_hessian_eigenvalue(dq, xb, params=groups[l], n_iter=cfg.hessian_iters)
           for l in sorted(groups)]
    del dq
    return lam

def _run_seed(seed: int, cfg: PoCConfig, loaders) -> dict:
    _, vl = loaders
    fp = train_one(False, seed, cfg, loaders)
    fpm = fp.pop("model")
    layers = list(range(len(fpm.model.layers)))
    sens = quantize_sensitivity(fpm, vl, cfg)             
    ptq = [sens["delta"][l] for l in layers]
    lam = _per_layer_lambda(fpm, vl, cfg)                 
    rho = spearman(lam, ptq)                              
    del fpm
    if cfg.device == "cuda":
        torch.cuda.empty_cache()
    return {
        "seed": seed,
        "layers": layers,
        "fp_val_loss": fp["final_val_loss"],
        "fp_base_val_loss": sens["fp_base_val_loss"],
        "ptq_sensitivity": ptq,
        "lambda_per_layer": lam,
        "spearman_lambda_ptq": rho,
        "argmax_ptq_layer": int(np.argmax(ptq)),
        "argmax_lambda_layer": int(np.argmax(lam)),
    }

def _summary(rows: list[dict]) -> dict:
    rhos = [r["spearman_lambda_ptq"] for r in rows]
    ptq_stack = np.array([r["ptq_sensitivity"] for r in rows], dtype=float)
    lam_stack = np.array([r["lambda_per_layer"] for r in rows], dtype=float)
    return {
        "rho_per_seed": rhos,
        "rho_mean": float(np.mean(rhos)),
        "rho_std": float(np.std(rhos)),
        "rho_pooled": spearman(lam_stack.reshape(-1).tolist(), ptq_stack.reshape(-1).tolist()),
        "layer0_is_top_ptq_all_seeds": bool(all(r["argmax_ptq_layer"] == 0 for r in rows)),
        "ptq_mean": ptq_stack.mean(0).tolist(),
        "ptq_std": ptq_stack.std(0).tolist(),
        "lambda_mean": lam_stack.mean(0).tolist(),
        "lambda_std": lam_stack.std(0).tolist(),
    }

def _save(obj: dict, name: str) -> str:
    p = os.path.join(CFG.out_dir, name)
    json.dump(obj, open(p, "w"), indent=2)
    if os.path.isdir(DRIVE):
        try:
            json.dump(obj, open(os.path.join(DRIVE, name), "w"), indent=2)
        except Exception as e:  
            print("drive save failed:", e)
    return p

def _selftest() -> None:
    cfg = PoCConfig(
        seeds=(0,), steps=6, batch_size=4, seq_len=32, warmup=2,
        eval_every=3, log_every=3, eval_batches=2,
        n_train_tokens=60_000, n_val_tokens=20_000,
        hessian_iters=3, hessian_bsz=2, hessian_batches=1,
        out_dir=CFG.out_dir,
    )
    tr, va = build_tinystories_tokens(cfg.data_dir, cfg.n_train_tokens, cfg.n_val_tokens)
    loaders = (TokenLoader(tr, cfg.seq_len, cfg.device), TokenLoader(va, cfg.seq_len, cfg.device))
    r = _run_seed(0, cfg, loaders)
    assert np.isfinite(r["spearman_lambda_ptq"]), "rho not finite"
    assert all(np.isfinite(x) for x in r["lambda_per_layer"]), "lambda not finite"
    assert all(d >= -1e-6 for d in r["ptq_sensitivity"]), "PTQ delta must be >= 0"
    if cfg.device == "cuda":
        torch.cuda.empty_cache()
    print("SELFTEST OK | rho=%.3f | ptq=%s" % (
        r["spearman_lambda_ptq"], [round(x, 4) for x in r["ptq_sensitivity"]]))

def main() -> None:
    os.makedirs(CFG.out_dir, exist_ok=True)
    status = os.path.join(CFG.out_dir, "H3CORR_STATUS.txt")
    open(status, "w").write("SELFTEST\n")
    t0 = time.time()
    try:
        _selftest()
        open(status, "w").write("RUNNING\n")
        tr, va = build_tinystories_tokens(CFG.data_dir, CFG.n_train_tokens, CFG.n_val_tokens)
        loaders = (TokenLoader(tr, CFG.seq_len, CFG.device), TokenLoader(va, CFG.seq_len, CFG.device))
        cfg_meta = {"steps": CFG.steps, "seeds": list(CFG.seeds), "hessian_iters": CFG.hessian_iters,
                    "hessian_bsz": CFG.hessian_bsz, "eval_batches": CFG.eval_batches}
        rows: list[dict] = []
        for seed in CFG.seeds:
            r = _run_seed(seed, CFG, loaders)
            rows.append(r)
            print("seed %d | rho=%.3f | argmax(ptq)=L%d argmax(lam)=L%d | fp_val=%.4f"
                  % (seed, r["spearman_lambda_ptq"], r["argmax_ptq_layer"],
                     r["argmax_lambda_layer"], r["fp_val_loss"]))
            _save({"config": cfg_meta, "rows": rows, "summary": _summary(rows),
                   "wall_s": time.time() - t0}, "h3_corr_multiseed.json")  
        s = _summary(rows)
        _save({"config": cfg_meta, "rows": rows, "summary": s, "wall_s": time.time() - t0},
              "h3_corr_multiseed.json")
        open(status, "w").write("DONE %.0fs rho=%.3f±%.3f\n" % (time.time() - t0, s["rho_mean"], s["rho_std"]))
        print("DONE %.1fs | rho=%.3f±%.3f | per-seed %s | pooled %.3f | layer0-top-all=%s"
              % (time.time() - t0, s["rho_mean"], s["rho_std"],
                 [round(x, 2) for x in s["rho_per_seed"]], s["rho_pooled"],
                 s["layer0_is_top_ptq_all_seeds"]))
        print("ptq_mean:", [round(x, 4) for x in s["ptq_mean"]])
        print("lambda_mean:", [round(x, 1) for x in s["lambda_mean"]])
    except Exception:
        tb = traceback.format_exc()
        print(tb)
        open(status, "w").write("ERROR\n" + tb)
        raise

if __name__ == "__main__":
    main()
