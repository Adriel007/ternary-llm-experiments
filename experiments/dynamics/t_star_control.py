
from __future__ import annotations

import json
import os
import sys
import time
import traceback

ROOT = os.environ.get("PHD_ROOT", "/content/PhD-propose")
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
from experiments.dynamics.t_star_sweep import set_quantize, _selftest  

CFG = PoCConfig(
    seeds=(0, 1, 2), steps=2000, batch_size=16, seq_len=128, warmup=100,
    eval_every=200, log_every=50, eval_batches=20,
    n_train_tokens=6_000_000, n_val_tokens=300_000,
    out_dir=os.path.join(ROOT, "artifacts/poc"),
)
T_FRACS = [0.0, 0.25, 0.5, 0.75, 1.0]
MODES = ["cosine", "constant", "rewarm"]
DRIVE = "/content/drive/MyDrive/PhD_PoC"

def _lr_at_mode(step, cfg, mode, switch_step):
    w = max(1, cfg.warmup)
    if mode == "cosine":
        return _lr_at(step, cfg)
    if mode == "constant":
        return cfg.lr * min(1.0, (step + 1) / w)
    if mode == "rewarm":
        if step >= switch_step:
            return cfg.lr * min(1.0, (step - switch_step + 1) / w)
        return cfg.lr * min(1.0, (step + 1) / w)
    raise ValueError(mode)

def _train(seed, cfg, loaders, switch_step, mode):
    train_loader, val_loader = loaders
    device = cfg.device
    model = build_model(True, seed, device)
    set_quantize(model, False)
    model.train()
    opt = torch.optim.AdamW(_param_groups(model, cfg.weight_decay), lr=cfg.lr, betas=cfg.betas)
    g = torch.Generator().manual_seed(seed)
    for step in range(cfg.steps):
        if step == switch_step:
            set_quantize(model, True)
        lr = _lr_at_mode(step, cfg, mode, switch_step)
        for pg in opt.param_groups:
            pg["lr"] = lr
        x = train_loader.batch(cfg.batch_size, g)
        with autocast(device_type="cuda", dtype=cfg.amp_dtype, enabled=cfg.amp and device == "cuda"):
            loss = model(input_ids=x, labels=x)["loss"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
    set_quantize(model, True)
    model.eval()
    return float(M.eval_loss(model, val_loader, cfg.batch_size, cfg.eval_batches, cfg.amp_dtype, device))

def _fp_baseline(seed, cfg, loaders, mode):
    train_loader, val_loader = loaders
    device = cfg.device
    model = build_model(True, seed, device)
    set_quantize(model, False)
    model.train()
    opt = torch.optim.AdamW(_param_groups(model, cfg.weight_decay), lr=cfg.lr, betas=cfg.betas)
    g = torch.Generator().manual_seed(seed)
    base_mode = "cosine" if mode == "cosine" else "constant"   
    for step in range(cfg.steps):
        lr = _lr_at_mode(step, cfg, base_mode, 0)
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

def main():
    out = CFG.out_dir
    os.makedirs(out, exist_ok=True)
    status = os.path.join(out, "TSTAR_CTRL_STATUS.txt")
    open(status, "w").write("RUNNING\n")
    t0 = time.time()
    try:
        dev = CFG.device
        _selftest(dev)
        tr, va = build_tinystories_tokens(CFG.data_dir, CFG.n_train_tokens, CFG.n_val_tokens)
        loaders = (TokenLoader(tr, CFG.seq_len, dev), TokenLoader(va, CFG.seq_len, dev))

        rows = []
        fp_val = {m: {} for m in MODES}
        for mode in MODES:
            for seed in CFG.seeds:
                fp_val[mode][seed] = _fp_baseline(seed, CFG, loaders, mode)
                print("[%s seed %d] FP val=%.4f" % (mode, seed, fp_val[mode][seed]), flush=True)
                for frac in T_FRACS:
                    sw = int(round(frac * CFG.steps))
                    v = _train(seed, CFG, loaders, sw, mode)
                    tax = v - fp_val[mode][seed]
                    rows.append({"mode": mode, "seed": seed, "t_frac": frac, "switch_step": sw,
                                 "val": v, "fp_val": fp_val[mode][seed], "tax": tax})
                    print("  [%s seed %d] t*=%.2f val=%.4f tax=%+.4f"
                          % (mode, seed, frac, v, tax), flush=True)
                    res = {"config": {"steps": CFG.steps, "seeds": list(CFG.seeds),
                                      "t_fracs": T_FRACS, "modes": MODES},
                           "fp_val": fp_val, "rows": rows, "wall_s": time.time() - t0}
                    json.dump(res, open(os.path.join(out, "t_star_control.json"), "w"), indent=2)
                    if os.path.isdir(DRIVE):
                        json.dump(res, open(os.path.join(DRIVE, "t_star_control.json"), "w"), indent=2)

        summary, u_preserved = {}, {}
        for mode in MODES:
            summary[mode] = {}
            for frac in T_FRACS:
                tx = [r["tax"] for r in rows if r["mode"] == mode and r["t_frac"] == frac]
                summary[mode][str(frac)] = {"tax_mean": float(np.mean(tx)), "tax_std": float(np.std(tx))}
            means = {f: summary[mode][str(f)]["tax_mean"] for f in T_FRACS}
            best = min(means, key=means.get)
            interior = [f for f in T_FRACS if f not in (0.0, 1.0)]
            u_preserved[mode] = {
                "best_t_frac": best,
                "u_shaped": min(means[f] for f in interior) < min(means[0.0], means[1.0]),
                "tax_t0": means[0.0], "tax_best": means[best], "tax_t1": means[1.0]}
        res = {"config": {"steps": CFG.steps, "seeds": list(CFG.seeds), "t_fracs": T_FRACS,
                          "modes": MODES},
               "fp_val": fp_val, "rows": rows, "summary": summary, "u_curve": u_preserved,
               "wall_s": time.time() - t0}
        json.dump(res, open(os.path.join(out, "t_star_control.json"), "w"), indent=2)
        if os.path.isdir(DRIVE):
            json.dump(res, open(os.path.join(DRIVE, "t_star_control.json"), "w"), indent=2)
        open(status, "w").write("DONE %.0fs\n" % (time.time() - t0))
        print("\n==== t* LR-CONTROL SUMMARY (tax by mode × t*) ====", flush=True)
        for mode in MODES:
            cells = " ".join("t%.2f=%+.4f" % (f, summary[mode][str(f)]["tax_mean"]) for f in T_FRACS)
            u = u_preserved[mode]
            print("  %-9s %s | best=%.2f U-shaped=%s" % (mode, cells, u["best_t_frac"], u["u_shaped"]),
                  flush=True)
        print("\nVERDICT: U survives constant-LR =", u_preserved["constant"]["u_shaped"],
              "(if False, the §3.12 U was a schedule artefact)", flush=True)
        print("DONE %.1fs" % (time.time() - t0), flush=True)
    except Exception:
        tb = traceback.format_exc(); print(tb)
        open(status, "w").write("ERROR\n" + tb)
        raise

if __name__ == "__main__":
    main()
