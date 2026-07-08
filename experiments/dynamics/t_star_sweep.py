
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
from experiments.dynamics import metrics as M  
from experiments.dynamics.data import TokenLoader, build_tinystories_tokens  
from experiments.dynamics.experiment import (  
    PoCConfig, build_model, _lr_at, _param_groups,
)

CFG = PoCConfig(
    seeds=(0, 1, 2), steps=2000, batch_size=16, seq_len=128, warmup=100,
    eval_every=200, log_every=50, eval_batches=20,
    n_train_tokens=6_000_000, n_val_tokens=300_000,
    out_dir=os.path.join(ROOT, "artifacts/poc"),
)
T_FRACS = [0.0, 0.25, 0.5, 0.75, 1.0]
DRIVE = "/content/drive/MyDrive/PhD_PoC"

def set_quantize(model, flag: bool):
    n = 0
    for m in model.modules():
        if isinstance(m, BitLinear):
            m.quantize = flag
            n += 1
    return n

def _train(seed, cfg, loaders, switch_step):
    train_loader, val_loader = loaders
    device = cfg.device
    model = build_model(True, seed, device)        
    set_quantize(model, False)                     
    model.train()
    opt = torch.optim.AdamW(_param_groups(model, cfg.weight_decay), lr=cfg.lr, betas=cfg.betas)
    data_gen = torch.Generator().manual_seed(seed)  
    for step in range(cfg.steps):
        if step == switch_step:
            set_quantize(model, True)              
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
    set_quantize(model, True)                      
    model.eval()
    return float(M.eval_loss(model, val_loader, cfg.batch_size, cfg.eval_batches, cfg.amp_dtype, device))

def _fp_baseline(seed, cfg, loaders):
    train_loader, val_loader = loaders
    device = cfg.device
    model = build_model(True, seed, device)
    set_quantize(model, False)                     
    model.train()
    opt = torch.optim.AdamW(_param_groups(model, cfg.weight_decay), lr=cfg.lr, betas=cfg.betas)
    g = torch.Generator().manual_seed(seed)
    for step in range(cfg.steps):
        lr = _lr_at(step, cfg)
        for pg in opt.param_groups:
            pg["lr"] = lr
        x = train_loader.batch(cfg.batch_size, g)
        with autocast(device_type="cuda", dtype=cfg.amp_dtype, enabled=cfg.amp and device == "cuda"):
            loss = model(input_ids=x, labels=x)["loss"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
    model.eval()
    return float(M.eval_loss(model, val_loader, cfg.batch_size, cfg.eval_batches, cfg.amp_dtype, device))

def _selftest(device):
    torch.manual_seed(0)
    m = build_model(True, 0, device)
    x = torch.randint(0, 100, (2, 16), device=device)
    set_quantize(m, False); y_fp = m(input_ids=x)["logits"].float()
    set_quantize(m, True);  y_q = m(input_ids=x)["logits"].float()
    rel = (y_q - y_fp).norm().item() / (y_fp.norm().item() + 1e-9)
    nq = sum(isinstance(mm, BitLinear) for mm in m.modules())
    print(f"[selftest] BitLinears={nq} flip rel-change={rel:.4g}", flush=True)
    assert nq > 0 and rel > 1e-3, "quantize flip must change the forward"
    del m; torch.cuda.empty_cache()
    print("[selftest] PASS", flush=True)

def main():
    out = CFG.out_dir
    os.makedirs(out, exist_ok=True)
    status = os.path.join(out, "TSTAR_STATUS.txt")
    open(status, "w").write("RUNNING\n")
    t0 = time.time()
    try:
        dev = CFG.device
        _selftest(dev)
        tr, va = build_tinystories_tokens(CFG.data_dir, CFG.n_train_tokens, CFG.n_val_tokens)
        loaders = (TokenLoader(tr, CFG.seq_len, dev), TokenLoader(va, CFG.seq_len, dev))

        rows = []                                   
        fp_val = {}
        for seed in CFG.seeds:
            fp_val[seed] = _fp_baseline(seed, CFG, loaders)
            print(f"[seed {seed}] FP val={fp_val[seed]:.4f}", flush=True)
            for frac in T_FRACS:
                sw = int(round(frac * CFG.steps))
                v = _train(seed, CFG, loaders, sw)
                rows.append({"seed": seed, "t_frac": frac, "switch_step": sw,
                             "val": v, "fp_val": fp_val[seed], "tax": v - fp_val[seed]})
                print(f"[seed {seed}] t*={frac:.2f} (step {sw}) val={v:.4f} tax={v-fp_val[seed]:+.4f}", flush=True)
                
                res = {"config": {"steps": CFG.steps, "seeds": list(CFG.seeds), "t_fracs": T_FRACS},
                       "fp_val": fp_val, "rows": rows, "wall_s": time.time() - t0}
                json.dump(res, open(os.path.join(out, "t_star.json"), "w"), indent=2)
                if os.path.isdir(DRIVE):
                    json.dump(res, open(os.path.join(DRIVE, "t_star.json"), "w"), indent=2)

        summary = {}
        for frac in T_FRACS:
            taxes = [r["tax"] for r in rows if r["t_frac"] == frac]
            summary[str(frac)] = {"tax_mean": float(np.mean(taxes)), "tax_std": float(np.std(taxes)),
                                  "n": len(taxes)}
        best = min(summary, key=lambda k: summary[k]["tax_mean"])
        res = {"config": {"steps": CFG.steps, "seeds": list(CFG.seeds), "t_fracs": T_FRACS},
               "fp_val": fp_val, "rows": rows, "summary": summary, "best_t_frac": float(best),
               "wall_s": time.time() - t0}
        json.dump(res, open(os.path.join(out, "t_star.json"), "w"), indent=2)
        if os.path.isdir(DRIVE):
            json.dump(res, open(os.path.join(DRIVE, "t_star.json"), "w"), indent=2)
        open(status, "w").write("DONE %.0fs\n" % (time.time() - t0) + json.dumps(summary))
        print("T-STAR DONE %.0fs | best t*=%s | tax@0=%.4f tax@best=%.4f" % (
            time.time() - t0, best, summary["0.0"]["tax_mean"], summary[best]["tax_mean"]), flush=True)
    except Exception:
        tb = traceback.format_exc(); print(tb)
        open(status, "w").write("ERROR\n" + tb)
        raise

if __name__ == "__main__":
    main()
