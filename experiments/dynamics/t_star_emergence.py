
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

GRID = [0.0, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 0.95]
DRIVE = "/content/drive/MyDrive/PhD_PoC"
HIST_BINS = np.linspace(-3.0, 3.0, 61)   

def set_quantize(model, flag: bool) -> int:
    n = 0
    for m in model.modules():
        if isinstance(m, BitLinear):
            m.quantize = flag
            n += 1
    return n

def _ev(model, val_loader, cfg) -> float:
    return float(M.eval_loss(model, val_loader, cfg.batch_size, cfg.eval_batches, cfg.amp_dtype, cfg.device))

@torch.no_grad()
def _wbeta_hist(model) -> list[int]:
    counts = np.zeros(len(HIST_BINS) - 1, dtype=np.int64)
    for m in model.modules():
        if isinstance(m, BitLinear):
            w = m.weight.detach().float()
            beta = w.abs().mean().clamp(min=1e-5)
            ws = (w / beta).flatten().cpu().numpy()
            counts += np.histogram(ws, bins=HIST_BINS)[0]
    return counts.tolist()

def _readiness(model, val_loader, cfg, want_hist: bool) -> dict:
    set_quantize(model, False)
    l_fp = _ev(model, val_loader, cfg)
    set_quantize(model, True)                      
    l_ptq = _ev(model, val_loader, cfg)
    set_quantize(model, False)                     
    st = M.ternary_layer_stats(model)["overall"]   
    out = {"L_fp": l_fp, "R_ptq": l_ptq - l_fp, "D_fp": st["dist_to_lattice"],
           "z_fp": st["frac_zero"]}
    if want_hist:
        out["hist"] = _wbeta_hist(model)
    return out

def _run_seed(seed, cfg, loaders, grid_steps, want_hist):
    train_loader, val_loader = loaders
    device = cfg.device
    grid_set = set(grid_steps)

    model = build_model(True, seed, device)        
    set_quantize(model, False)
    model.train()
    opt = torch.optim.AdamW(_param_groups(model, cfg.weight_decay), lr=cfg.lr, betas=cfg.betas)
    gen = torch.Generator().manual_seed(seed)

    snaps, ready, hist_by_step = {}, {}, {}
    for step in range(cfg.steps):
        if step in grid_set:                       
            ready[step] = _readiness(model, val_loader, cfg, want_hist)
            if want_hist:
                hist_by_step[step] = ready[step].pop("hist")
            model.train()
            path = os.path.join(cfg.out_dir, f"_snap_s{seed}_t{step}.pt")
            torch.save({"model": {k: v.cpu() for k, v in model.state_dict().items()},
                        "opt": opt.state_dict(), "gen": gen.get_state()}, path)
            snaps[step] = path
        lr = _lr_at(step, cfg)
        for pg in opt.param_groups:
            pg["lr"] = lr
        x = train_loader.batch(cfg.batch_size, gen)
        with autocast(device_type="cuda", dtype=cfg.amp_dtype, enabled=cfg.amp and device == "cuda"):
            loss = model(input_ids=x, labels=x)["loss"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

    set_quantize(model, False)
    fp_full_val = _ev(model, val_loader, cfg)       
    del model, opt
    torch.cuda.empty_cache()

    rows = []
    for step in grid_steps:
        v, fd, fz = _fork_finish(seed, cfg, loaders, snaps[step], step)
        r = dict(ready[step])
        r.update({"seed": seed, "switch_step": step, "t_frac": step / cfg.steps,
                  "final_val": v, "final_tax": v - fp_full_val,
                  "final_D": fd, "final_z": fz, "fp_full_val": fp_full_val})
        rows.append(r)
        os.remove(snaps[step])
        print(f"[seed {seed}] s={step/cfg.steps:.2f} R_ptq={r['R_ptq']:+.3f} D_fp={r['D_fp']:.3f} "
              f"z_fp={r['z_fp']:.3f} -> tax={r['final_tax']:+.4f}", flush=True)
    return rows, fp_full_val, hist_by_step

def _fork_finish(seed, cfg, loaders, snap_path, switch_step):
    train_loader, val_loader = loaders
    device = cfg.device
    snap = torch.load(snap_path, map_location="cpu")   
    model = build_model(True, seed, device)
    model.load_state_dict({k: v.to(device) for k, v in snap["model"].items()})
    opt = torch.optim.AdamW(_param_groups(model, cfg.weight_decay), lr=cfg.lr, betas=cfg.betas)
    opt.load_state_dict(snap["opt"])
    for st in opt.state.values():                       
        for k, v in st.items():
            if isinstance(v, torch.Tensor):
                st[k] = v.to(device)
    gen = torch.Generator()
    gen.set_state(snap["gen"])
    set_quantize(model, True)                       
    model.train()
    for step in range(switch_step, cfg.steps):
        lr = _lr_at(step, cfg)
        for pg in opt.param_groups:
            pg["lr"] = lr
        x = train_loader.batch(cfg.batch_size, gen)
        with autocast(device_type="cuda", dtype=cfg.amp_dtype, enabled=cfg.amp and device == "cuda"):
            loss = model(input_ids=x, labels=x)["loss"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
    model.eval()
    v = _ev(model, val_loader, cfg)
    st = M.ternary_layer_stats(model)["overall"]
    fd, fz = st["dist_to_lattice"], st["frac_zero"]
    del model, opt
    torch.cuda.empty_cache()
    return v, fd, fz

def _spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ra, rb = a.argsort().argsort().astype(float), b.argsort().argsort().astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    return float((ra * rb).sum() / (np.sqrt((ra**2).sum() * (rb**2).sum()) + 1e-12))

def _selftest(cfg):
    dev = cfg.device
    m = build_model(True, 0, dev)
    x = torch.randint(0, 100, (2, 16), device=dev)
    set_quantize(m, False); y_fp = m(input_ids=x)["logits"].float()
    set_quantize(m, True);  y_q = m(input_ids=x)["logits"].float()
    rel = (y_q - y_fp).norm().item() / (y_fp.norm().item() + 1e-9)
    assert rel > 1e-3, "quantize flip must change the forward"
    del m

    def _short_qat(from_snapshot):
        model = build_model(True, 0, dev)
        set_quantize(model, False); model.train()
        opt = torch.optim.AdamW(_param_groups(model, cfg.weight_decay), lr=cfg.lr, betas=cfg.betas)
        gen = torch.Generator().manual_seed(0)
        loader = _TinyLoader(dev)
        if from_snapshot:                            
            snap = {"model": {k: v.cpu() for k, v in model.state_dict().items()},
                    "opt": opt.state_dict(), "gen": gen.get_state()}
            del model, opt
            model = build_model(True, 0, dev)
            model.load_state_dict({k: v.to(dev) for k, v in snap["model"].items()})
            opt = torch.optim.AdamW(_param_groups(model, cfg.weight_decay), lr=cfg.lr, betas=cfg.betas)
            opt.load_state_dict(snap["opt"]); gen.set_state(snap["gen"])
        set_quantize(model, True); model.train()
        for step in range(5):
            for pg in opt.param_groups:
                pg["lr"] = cfg.lr
            xb = loader.batch(8, gen)
            loss = model(input_ids=xb, labels=xb)["loss"]
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        return torch.cat([p.detach().flatten() for p in model.parameters()])

    direct = _short_qat(False)
    forked = _short_qat(True)
    d = (direct - forked).abs().max().item()
    print(f"[selftest] flip rel={rel:.4g}  fork-vs-direct max|Δw|={d:.3e}", flush=True)
    assert d < 1e-6, f"fork@0 must reproduce from-scratch QAT (max|Δw|={d:.3e})"
    print("[selftest] PASS", flush=True)

class _TinyLoader:
    def __init__(self, device):
        self.device = device
        self.g = torch.Generator().manual_seed(123)
        self.data = torch.randint(0, 256, (20000,), generator=self.g)

    def batch(self, bs, gen):
        ix = torch.randint(self.data.shape[0] - 33, (bs,), generator=gen)
        return torch.stack([self.data[i:i + 32] for i in ix]).to(self.device)

def main():
    out = CFG.out_dir
    os.makedirs(out, exist_ok=True)
    status = os.path.join(out, "TSTAR_EMERGE_STATUS.txt")
    open(status, "w").write("RUNNING\n")
    t0 = time.time()
    try:
        _selftest(CFG)
        tr, va = build_tinystories_tokens(CFG.data_dir, CFG.n_train_tokens, CFG.n_val_tokens)
        loaders = (TokenLoader(tr, CFG.seq_len, CFG.device), TokenLoader(va, CFG.seq_len, CFG.device))
        grid_steps = sorted({int(round(f * CFG.steps)) for f in GRID})

        all_rows, fp_full, hist0 = [], {}, {}
        for seed in CFG.seeds:
            rows, fpv, hbs = _run_seed(seed, CFG, loaders, grid_steps, want_hist=(seed == CFG.seeds[0]))
            all_rows += rows
            fp_full[seed] = fpv
            if seed == CFG.seeds[0]:
                hist0 = hbs
            res = {"config": {"steps": CFG.steps, "seeds": list(CFG.seeds), "grid": GRID,
                              "hist_bins": HIST_BINS.tolist()},
                   "rows": all_rows, "fp_full_val": fp_full,
                   "hist_seed0_by_step": hist0, "wall_s": time.time() - t0}
            json.dump(res, open(os.path.join(out, "t_star_emergence.json"), "w"), indent=2)
            if os.path.isdir(DRIVE):
                json.dump(res, open(os.path.join(DRIVE, "t_star_emergence.json"), "w"), indent=2)

        fracs = sorted({r["t_frac"] for r in all_rows})
        per = {}
        for f in fracs:
            g = [r for r in all_rows if r["t_frac"] == f]
            per[f"{f:.3f}"] = {k + "_mean": float(np.mean([r[k] for r in g]))
                               for k in ("final_tax", "R_ptq", "L_fp", "D_fp", "z_fp")}
            per[f"{f:.3f}"]["final_tax_std"] = float(np.std([r["final_tax"] for r in g]))
        
        usable = [r for r in all_rows if r["t_frac"] < 0.9]
        coup = {
            "best_t_frac": float(min(fracs, key=lambda f: per[f"{f:.3f}"]["final_tax_mean"])),
            "spearman_tax_vs_Rptq": _spearman([r["final_tax"] for r in usable], [r["R_ptq"] for r in usable]),
            "spearman_tax_vs_Lfp": _spearman([r["final_tax"] for r in usable], [r["L_fp"] for r in usable]),
            "D_fp_range": [float(min(r["D_fp"] for r in all_rows)), float(max(r["D_fp"] for r in all_rows))],
            "z_fp_range": [float(min(r["z_fp"] for r in all_rows)), float(max(r["z_fp"] for r in all_rows))],
            "Rptq_range": [float(min(r["R_ptq"] for r in all_rows)), float(max(r["R_ptq"] for r in all_rows))],
        }
        res["summary"] = {"per_frac": per, "coupling": coup}
        json.dump(res, open(os.path.join(out, "t_star_emergence.json"), "w"), indent=2)
        if os.path.isdir(DRIVE):
            json.dump(res, open(os.path.join(DRIVE, "t_star_emergence.json"), "w"), indent=2)
        open(status, "w").write("DONE %.0fs\n" % (time.time() - t0) + json.dumps(coup))
        print("EMERGE DONE %.0fs | best t*=%.2f | spearman(tax,R_ptq)=%.2f spearman(tax,L_fp)=%.2f "
              "| D_fp range=%s" % (time.time() - t0, coup["best_t_frac"],
              coup["spearman_tax_vs_Rptq"], coup["spearman_tax_vs_Lfp"], coup["D_fp_range"]), flush=True)
    except Exception:
        tb = traceback.format_exc(); print(tb)
        open(status, "w").write("ERROR\n" + tb)
        raise

if __name__ == "__main__":
    main()
