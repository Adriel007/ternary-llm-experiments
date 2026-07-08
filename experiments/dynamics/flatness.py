
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
from torch.nn.attention import SDPBackend, sdpa_kernel  

from experiments.dynamics import metrics as M  
from experiments.dynamics.data import TokenLoader, build_tinystories_tokens  
from experiments.dynamics.experiment import PoCConfig, train_one  

CFG = PoCConfig(
    seeds=(0, 1, 2), steps=2000, batch_size=16, seq_len=128, warmup=100,
    eval_every=200, log_every=50, eval_batches=20,
    n_train_tokens=6_000_000, n_val_tokens=300_000,
    hessian_iters=20, hessian_bsz=8,
    out_dir=os.path.join(ROOT, "artifacts/poc"),
)
N_HUTCH = 60
DRIVE = "/content/drive/MyDrive/PhD_PoC"

def hessian_trace_total(dq_model, x, n_hutch=N_HUTCH, seed=0):
    params = [p for p in dq_model.parameters() if p.requires_grad]
    g = torch.Generator(device="cpu").manual_seed(seed)
    with sdpa_kernel(SDPBackend.MATH):
        loss = dq_model(input_ids=x, labels=x)["loss"]
        grads = torch.autograd.grad(loss, params, create_graph=True)
        ests = []
        for _ in range(n_hutch):
            v = [(torch.randint(0, 2, p.shape, generator=g).to(p) * 2 - 1) for p in params]
            dot = sum((gr * vv).sum() for gr, vv in zip(grads, v))
            Hv = torch.autograd.grad(dot, params, retain_graph=True)
            ests.append(float(sum((hv * vv).sum() for hv, vv in zip(Hv, v)).item()))
    return float(np.mean(ests)), float(np.std(ests))

def main():
    out = CFG.out_dir
    os.makedirs(out, exist_ok=True)
    status = os.path.join(out, "FLAT_STATUS.txt")
    open(status, "w").write("RUNNING\n")
    t0 = time.time()
    try:
        tr, va = build_tinystories_tokens(CFG.data_dir, CFG.n_train_tokens, CFG.n_val_tokens)
        loaders = (TokenLoader(tr, CFG.seq_len, CFG.device), TokenLoader(va, CFG.seq_len, CFG.device))
        vl = loaders[1]
        xb = next(iter(vl.iter_eval(CFG.hessian_bsz, 1)))

        rows = {"ternary": [], "fp": []}
        for seed in CFG.seeds:
            for quant, key in [(False, "fp"), (True, "ternary")]:
                r = train_one(quant, seed, CFG, loaders); model = r.pop("model")
                dq = M.dequantized_float_copy(model)
                lam = M.top_hessian_eigenvalue(dq, xb, n_iter=CFG.hessian_iters)
                trace, trace_sd = hessian_trace_total(dq, xb)
                del dq, model; torch.cuda.empty_cache()
                row = {"seed": seed, "lambda_max": lam, "trace": trace, "trace_sd": trace_sd,
                       "final_train_loss": r["train"]["train_loss"][-1],
                       "final_val_loss": r["final_val_loss"]}
                row["train_val_gap"] = row["final_val_loss"] - row["final_train_loss"]
                rows[key].append(row)
                print(f"[seed {seed}] {key:8s} lambda_max={lam:.1f} trace={trace:.1f} "
                      f"gap={row['train_val_gap']:+.4f}", flush=True)

        def msd(key, field):
            a = np.array([r[field] for r in rows[key]], float)
            return {"mean": float(a.mean()), "std": float(a.std()), "n": len(a)}

        summary = {f: {k: msd(k, f) for k in ("ternary", "fp")}
                   for f in ("lambda_max", "trace", "train_val_gap", "final_val_loss")}
        summary["trace_ratio_tern_over_fp"] = summary["trace"]["ternary"]["mean"] / summary["trace"]["fp"]["mean"]
        summary["lambda_ratio_tern_over_fp"] = summary["lambda_max"]["ternary"]["mean"] / summary["lambda_max"]["fp"]["mean"]

        result = {"config": {"steps": CFG.steps, "n_hutch": N_HUTCH, "seeds": list(CFG.seeds)},
                  "rows": rows, "summary": summary}
        json.dump(result, open(os.path.join(out, "flatness.json"), "w"), indent=2)
        if os.path.isdir(DRIVE):
            json.dump(result, open(os.path.join(DRIVE, "flatness.json"), "w"), indent=2)
        open(status, "w").write(f"DONE {time.time()-t0:.0f}s\n")
        print("FLATNESS DONE", round(time.time() - t0, 1), "s", flush=True)
        print("trace  ternary=%.1f fp=%.1f  ratio=%.2f" % (
            summary["trace"]["ternary"]["mean"], summary["trace"]["fp"]["mean"],
            summary["trace_ratio_tern_over_fp"]), flush=True)
        print("lambda ternary=%.1f fp=%.1f  ratio=%.2f" % (
            summary["lambda_max"]["ternary"]["mean"], summary["lambda_max"]["fp"]["mean"],
            summary["lambda_ratio_tern_over_fp"]), flush=True)
    except Exception:
        tb = traceback.format_exc(); print(tb)
        open(status, "w").write("ERROR\n" + tb)
        raise

if __name__ == "__main__":
    main()
