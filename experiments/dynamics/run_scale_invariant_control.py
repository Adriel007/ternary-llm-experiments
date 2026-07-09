from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback

ROOT = os.environ.get("PHD_ROOT", os.getcwd())
sys.path[:0] = [ROOT, os.path.join(ROOT, "packages/bitnet_core/src")]

import numpy as np  
import torch  
from torch.amp import autocast  

from experiments.dynamics import metrics as M  
from experiments.dynamics import scale_invariant_metrics as SI  
from experiments.dynamics.data import TokenLoader, build_tinystories_tokens  
from experiments.dynamics.experiment import (  
    PoCConfig, build_model, train_one, _param_groups, _lr_at,
)

def _bitlinear_params(model):
    groups = M.layer_weight_groups(model)
    return [w for ws in groups.values() for w in ws]

def train_fp_with_snapshot(seed, cfg, loaders, target_loss):
    train_loader, val_loader = loaders
    device = cfg.device
    model = build_model(False, seed, device)
    model.train()
    opt = torch.optim.AdamW(_param_groups(model, cfg.weight_decay), lr=cfg.lr, betas=cfg.betas)
    data_gen = torch.Generator().manual_seed(seed)   
    best_dq, best_gap, best_loss, best_step = None, float("inf"), None, None
    for step in range(cfg.steps):
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
        if step % cfg.eval_every == 0 or step == cfg.steps - 1:
            vl = M.eval_loss(model, val_loader, cfg.batch_size, cfg.eval_batches, cfg.amp_dtype, device)
            if abs(vl - target_loss) < best_gap:
                best_gap, best_loss, best_step = abs(vl - target_loss), vl, step
                del best_dq
                best_dq = M.dequantized_float_copy(model)   
            model.train()
    final_val = M.eval_loss(model, val_loader, cfg.batch_size, cfg.eval_batches, cfg.amp_dtype, device)
    return model, final_val, best_dq, best_loss, best_gap, best_step

def lenses(dq_model, x, n_probe, params=None):
    dq_model.eval()
    plist = params if params is not None else [p for p in dq_model.parameters() if p.requires_grad]
    fsc = SI.filter_scales(plist)
    return {
        "raw": SI.hutchinson_trace(dq_model, x, None, n_probe, 0, plist),
        "filter_norm": SI.hutchinson_trace(dq_model, x, fsc, n_probe, 0, plist),
        "fisher": SI.fisher_diag_trace(dq_model, x, plist),
    }

def ratio_of_means(rows, kind, lens):
    t = np.mean([r[kind]["ternary"][lens] for r in rows])
    f = np.mean([r[kind]["fp"][lens] for r in rows])
    return float(t / f) if f else float("nan")

def mean_of_ratios(rows, kind, lens):
    rs = [r[kind]["ternary"][lens] / r[kind]["fp"][lens] for r in rows if r[kind]["fp"][lens]]
    return {"mean": float(np.mean(rs)), "std": float(np.std(rs)), "n": len(rs)}

def verdict(r):
    if r > 1.15:
        return "SIGNAL SURVIVES -> genuinely sharper; headline stands."
    if r < 0.87:
        return "SIGNAL INVERTS -> raw effect was a scale artifact; rewrite headline."
    return "SIGNAL VANISHES -> curvature comparable under invariant metric; demote headline."

def run(cfg, seeds, n_probe, out_path):
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    train_ids, val_ids = build_tinystories_tokens(cfg.data_dir, cfg.n_train_tokens, cfg.n_val_tokens)
    loaders = (TokenLoader(train_ids, cfg.seq_len, cfg.device), TokenLoader(val_ids, cfg.seq_len, cfg.device))
    xb = next(iter(loaders[1].iter_eval(cfg.hessian_bsz, 1)))   

    rows = []
    for seed in seeds:
        t0 = time.time()
        rt = train_one(True, seed, cfg, loaders)
        tern = rt.pop("model"); Lt = rt["final_val_loss"]
        dq_t = M.dequantized_float_copy(tern)
        del tern; torch.cuda.empty_cache()

        fp, Lfp, dq_fp_sl, sl_loss, sl_gap, sl_step = train_fp_with_snapshot(seed, cfg, loaders, Lt)
        dq_fp_final = M.dequantized_float_copy(fp)
        del fp; torch.cuda.empty_cache()

        bl_t = _bitlinear_params(dq_t)
        bl_ff = _bitlinear_params(dq_fp_final)
        row = {
            "seed": seed, "Lt_ternary": Lt, "Lfp_final": Lfp,
            "sameloss_fp_loss": sl_loss, "sameloss_gap": sl_gap, "sameloss_step": sl_step,
            "allparams_final": {"ternary": lenses(dq_t, xb, n_probe),
                                "fp": lenses(dq_fp_final, xb, n_probe)},
            "bitlinear_final": {"ternary": lenses(dq_t, xb, n_probe, bl_t),
                                "fp": lenses(dq_fp_final, xb, n_probe, bl_ff)},
        }
        if dq_fp_sl is not None:
            bl_fs = _bitlinear_params(dq_fp_sl)
            row["allparams_sameloss"] = {"ternary": lenses(dq_t, xb, n_probe),
                                         "fp": lenses(dq_fp_sl, xb, n_probe)}
            row["bitlinear_sameloss"] = {"ternary": lenses(dq_t, xb, n_probe, bl_t),
                                         "fp": lenses(dq_fp_sl, xb, n_probe, bl_fs)}
        row["wall_s"] = time.time() - t0
        rows.append(row)
        del dq_t, dq_fp_final, dq_fp_sl; torch.cuda.empty_cache()
        print(f"[seed {seed}] Lt={Lt:.4f} Lfp={Lfp:.4f} sameloss@{sl_step}={sl_loss:.4f} "
              f"raw={row['allparams_final']['ternary']['raw']:.1f}/"
              f"{row['allparams_final']['fp']['raw']:.1f} wall={row['wall_s']:.0f}s", flush=True)
        json.dump({"partial": True, "rows": rows}, open(out_path, "w"), indent=2)

    out = {"config": {"seeds": list(seeds), "steps": cfg.steps, "n_probe": n_probe,
                      "batch_size": cfg.batch_size, "seq_len": cfg.seq_len},
           "rows": rows, "summary": {}}
    for kind in ("allparams_final", "bitlinear_final", "allparams_sameloss", "bitlinear_sameloss"):
        if not all(kind in r for r in rows):
            continue
        s = {}
        for lens in ("raw", "filter_norm", "fisher"):
            s[lens] = {"ratio_of_means": ratio_of_means(rows, kind, lens),
                       "mean_of_ratios": mean_of_ratios(rows, kind, lens)}
        out["summary"][kind] = s
    fn_all = out["summary"]["allparams_final"]["filter_norm"]["ratio_of_means"]
    raw_all = out["summary"]["allparams_final"]["raw"]["ratio_of_means"]
    out["verdict"] = {
        "raw_ratio_allparams": raw_all,
        "filter_norm_ratio_allparams_final": fn_all,
        "filter_norm_ratio_allparams_sameloss":
            out["summary"].get("allparams_sameloss", {}).get("filter_norm", {}).get("ratio_of_means"),
        "filter_norm_ratio_bitlinear_final":
            out["summary"]["bitlinear_final"]["filter_norm"]["ratio_of_means"],
        "decision_on_allparams_final_filternorm": verdict(fn_all),
    }
    json.dump(out, open(out_path, "w"), indent=2)
    print("\n=== SUMMARY ===")
    print(f"raw trace ratio (all params, should ~reproduce 2.15x): {raw_all:.3f}")
    print(f"filter-norm ratio (all params, final FP):  {fn_all:.3f}  -> {verdict(fn_all)}")
    sl = out["verdict"]["filter_norm_ratio_allparams_sameloss"]
    if sl is not None:
        print(f"filter-norm ratio (all params, same-loss FP): {sl:.3f}  -> {verdict(sl)}")
    print(f"filter-norm ratio (BitLinear only, final FP): "
          f"{out['verdict']['filter_norm_ratio_bitlinear_final']:.3f}")
    print(f"wrote {out_path}")
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "scale_invariant.json"))
    ap.add_argument("--data-dir", default=os.environ.get("DATA_DIR", "/workspace/data"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--n-probe", type=int, default=60)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        a.seeds, a.steps, a.n_probe = [0], 80, 8
    
    cfg = PoCConfig(seeds=tuple(a.seeds), steps=a.steps, batch_size=16, seq_len=128, warmup=100,
                    eval_every=200, log_every=100, eval_batches=20,
                    n_train_tokens=6_000_000 if not a.smoke else 1_000_000,
                    n_val_tokens=300_000 if not a.smoke else 100_000,
                    hessian_bsz=8, data_dir=a.data_dir,
                    out_dir=os.path.join(ROOT, "artifacts/poc"))
    try:
        run(cfg, tuple(a.seeds), a.n_probe, a.out)
    except Exception:
        print(traceback.format_exc(), flush=True)
        raise

if __name__ == "__main__":
    main()
