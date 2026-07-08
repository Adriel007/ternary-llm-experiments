
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
from torch import autocast  

from bitnet_core.bitlinear import BitLinear  
from bitnet_core.quant import quantize_weight_ste, soft_ternary_weight, ternarize  
from experiments.dynamics import metrics as M  
from experiments.dynamics.data import TokenLoader, build_tinystories_tokens  
from experiments.dynamics.experiment import (  
    PoCConfig, build_model, train_one, _lr_at, _param_groups,
)
from experiments.dynamics.flatness import hessian_trace_total  

CFG = PoCConfig(
    seeds=(0,), steps=2000, batch_size=16, seq_len=128, warmup=100,
    eval_every=200, log_every=50, eval_batches=20,
    n_train_tokens=6_000_000, n_val_tokens=300_000,
    hessian_iters=20, hessian_bsz=8,
    out_dir=os.path.join(ROOT, "artifacts/poc"),
)
TAU0, TAU_MIN = 1.0, 0.02          
N_HUTCH = 40
DRIVE = "/content/drive/MyDrive/PhD_PoC"

def set_soft_tau(model, tau):
    n = 0
    for m in model.modules():
        if isinstance(m, BitLinear) and m.quantize:
            m.soft_tau = tau
            n += 1
    return n

def _selftest():
    torch.manual_seed(0)
    W = torch.randn(256, 256, requires_grad=True)
    hard = quantize_weight_ste(W)
    soft = soft_ternary_weight(W, tau=1e-3)
    codes_hard, beta = ternarize(W)
    codes_soft = (soft.detach() / beta).round().clamp(-1, 1)
    agree = (codes_hard == codes_soft).float().mean().item()
    soft.sum().backward()
    grad_ok = (W.grad is not None) and bool(torch.isfinite(W.grad).all().item())
    
    u = (W.detach() / beta)
    interior = (u.sub(0.5).abs() > 0.05) & (u.add(0.5).abs() > 0.05)
    maxdiff_int = (soft.detach() - hard)[interior].abs().max().item()
    print(f"[selftest] code-agreement={agree:.4f} grad_ok={grad_ok} "
          f"interior_maxdiff={maxdiff_int:.4g} (tau=1e-3)", flush=True)
    assert agree > 0.99, "soft tau->0 must match hard ternary codes"
    assert grad_ok, "soft quant must be differentiable with finite grads"
    assert maxdiff_int < 1e-2, "soft must converge to hard away from the boundary"
    
    lin = BitLinear(64, 64).eval()
    x = torch.randn(4, 64)
    with torch.no_grad():
        y_hard = lin(x)
        lin.soft_tau = 1e-3
        y_soft = lin(x)
    rel = (y_soft - y_hard).norm().item() / (y_hard.norm().item() + 1e-9)
    print(f"[selftest] BitLinear soft-vs-hard rel-err={rel:.4g}", flush=True)
    assert rel < 0.05, "soft BitLinear at tiny tau must track the hard forward"
    print("[selftest] PASS", flush=True)

def train_soft(seed, cfg, loaders, tau0=TAU0, tau_min=TAU_MIN):
    train_loader, _ = loaders
    device = cfg.device
    model = build_model(True, seed, device)
    model.train()
    opt = torch.optim.AdamW(_param_groups(model, cfg.weight_decay), lr=cfg.lr, betas=cfg.betas)
    data_gen = torch.Generator().manual_seed(seed)        
    hist = {"step": [], "train_loss": [], "tau": []}
    run_loss = None
    for step in range(cfg.steps):
        frac = step / max(1, cfg.steps - 1)
        tau = tau0 * (tau_min / tau0) ** frac              
        set_soft_tau(model, tau)
        lr = _lr_at(step, cfg)
        for pg in opt.param_groups:
            pg["lr"] = lr
        x = train_loader.batch(cfg.batch_size, data_gen)
        with autocast(device_type="cuda", dtype=cfg.amp_dtype, enabled=cfg.amp and device == "cuda"):
            loss = model(input_ids=x, labels=x)["loss"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        lv = loss.item(); run_loss = lv if run_loss is None else 0.9 * run_loss + 0.1 * lv
        if step % cfg.log_every == 0 or step == cfg.steps - 1:
            hist["step"].append(step); hist["train_loss"].append(run_loss); hist["tau"].append(tau)
    set_soft_tau(model, None)                              
    return model.eval(), hist

def eval_hard(model, loaders, cfg):
    train_loader, val_loader = loaders
    tr = M.eval_loss(model, train_loader, cfg.batch_size, cfg.eval_batches, cfg.amp_dtype, cfg.device)
    va = M.eval_loss(model, val_loader, cfg.batch_size, cfg.eval_batches, cfg.amp_dtype, cfg.device)
    return float(tr), float(va)

def _trace(model, xb):
    dq = M.dequantized_float_copy(model)
    lam = M.top_hessian_eigenvalue(dq, xb, n_iter=CFG.hessian_iters)
    tr, sd = hessian_trace_total(dq, xb, n_hutch=N_HUTCH)
    del dq; torch.cuda.empty_cache()
    return lam, tr, sd

def main():
    out = CFG.out_dir
    os.makedirs(out, exist_ok=True)
    status = os.path.join(out, "SOFTQ_STATUS.txt")
    open(status, "w").write("RUNNING\n")
    t0 = time.time()
    try:
        _selftest()                                        
        tr_ids, va_ids = build_tinystories_tokens(CFG.data_dir, CFG.n_train_tokens, CFG.n_val_tokens)
        loaders = (TokenLoader(tr_ids, CFG.seq_len, CFG.device), TokenLoader(va_ids, CFG.seq_len, CFG.device))
        xb = next(iter(loaders[1].iter_eval(CFG.hessian_bsz, 1)))

        rows = []
        for seed in CFG.seeds:
            
            rfp = train_one(False, seed, CFG, loaders); fp_val = float(rfp["final_val_loss"]); del rfp
            rste = train_one(True, seed, CFG, loaders); m_ste = rste["model"]
            ste_tr, ste_va = eval_hard(m_ste, loaders, CFG)
            ste_lam, ste_trace, ste_sd = _trace(m_ste, xb); del m_ste, rste; torch.cuda.empty_cache()

            m_soft, shist = train_soft(seed, CFG, loaders)
            soft_tr, soft_va = eval_hard(m_soft, loaders, CFG)
            soft_lam, soft_trace, soft_sd = _trace(m_soft, xb); del m_soft; torch.cuda.empty_cache()

            row = {
                "seed": seed, "fp_val": fp_val,
                "ste":  {"val": ste_va,  "train": ste_tr,  "tax": ste_va - fp_val,
                         "gap": ste_va - ste_tr,  "lambda_max": ste_lam,  "trace": ste_trace,  "trace_sd": ste_sd},
                "soft": {"val": soft_va, "train": soft_tr, "tax": soft_va - fp_val,
                         "gap": soft_va - soft_tr, "lambda_max": soft_lam, "trace": soft_trace, "trace_sd": soft_sd},
                "soft_hist": shist,
            }
            rows.append(row)
            print(f"[seed {seed}] STE  tax={row['ste']['tax']:+.4f} trace={ste_trace:.0f} gap={row['ste']['gap']:+.4f}", flush=True)
            print(f"[seed {seed}] SOFT tax={row['soft']['tax']:+.4f} trace={soft_trace:.0f} gap={row['soft']['gap']:+.4f}", flush=True)

        def mean(key, sub):
            return float(np.mean([r[sub][key] for r in rows]))
        summary = {sub: {k: mean(k, sub) for k in ("tax", "trace", "gap", "lambda_max", "val")}
                   for sub in ("ste", "soft")}
        summary["trace_soft_over_ste"] = summary["soft"]["trace"] / summary["ste"]["trace"]
        summary["tax_soft_minus_ste"] = summary["soft"]["tax"] - summary["ste"]["tax"]
        summary["gap_soft_minus_ste"] = summary["soft"]["gap"] - summary["ste"]["gap"]

        result = {"config": {"steps": CFG.steps, "seeds": list(CFG.seeds), "tau0": TAU0,
                             "tau_min": TAU_MIN, "n_hutch": N_HUTCH}, "rows": rows, "summary": summary}
        json.dump(result, open(os.path.join(out, "soft_quant_h4.json"), "w"), indent=2)
        if os.path.isdir(DRIVE):
            json.dump(result, open(os.path.join(DRIVE, "soft_quant_h4.json"), "w"), indent=2)
        open(status, "w").write(f"DONE {time.time()-t0:.0f}s\n" + json.dumps(summary))
        print("SOFT-QUANT H4 DONE %.0fs" % (time.time() - t0), flush=True)
        print("trace soft/ste=%.2f | dtax=%+.4f | dgap=%+.4f" % (
            summary["trace_soft_over_ste"], summary["tax_soft_minus_ste"], summary["gap_soft_minus_ste"]), flush=True)
    except Exception:
        tb = traceback.format_exc(); print(tb)
        open(status, "w").write("ERROR\n" + tb)
        raise

if __name__ == "__main__":
    main()
