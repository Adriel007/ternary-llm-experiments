
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast

from bitnet_core import TernaryConfig, TernaryForCausalLM

from . import metrics as M
from .data import VOCAB_SIZE, TokenLoader, build_tinystories_tokens

@dataclass
class PoCConfig:
    seeds: tuple[int, ...] = (0, 1, 2)
    steps: int = 3000
    batch_size: int = 24
    seq_len: int = 256
    lr: float = 3e-4
    min_lr_frac: float = 0.1
    warmup: int = 150
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    betas: tuple[float, float] = (0.9, 0.95)
    eval_every: int = 250
    log_every: int = 100
    eval_batches: int = 40
    n_train_tokens: int = 8_000_000
    n_val_tokens: int = 500_000
    hessian_iters: int = 20
    hessian_batches: int = 3
    hessian_bsz: int = 8
    pareto_random_orders: int = 20
    device: str = "cuda"
    data_dir: str = "/content/data"
    out_dir: str = "/content/PhD-propose/artifacts/poc"
    amp: bool = True

    @property
    def amp_dtype(self) -> torch.dtype:
        return torch.bfloat16

def _tiny_config() -> TernaryConfig:
    return TernaryConfig.tiny(vocab_size=VOCAB_SIZE)

def build_model(quantize: bool, seed: int, device: str) -> TernaryForCausalLM:
    torch.manual_seed(seed)
    model = TernaryForCausalLM(_tiny_config(), quantize=quantize)
    return model.to(device)

def _param_groups(model: nn.Module, weight_decay: float):
    decay, no_decay = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]

def _lr_at(step: int, cfg: PoCConfig) -> float:
    if step < cfg.warmup:
        return cfg.lr * (step + 1) / cfg.warmup
    prog = (step - cfg.warmup) / max(1, cfg.steps - cfg.warmup)
    prog = min(1.0, prog)
    coeff = 0.5 * (1.0 + math.cos(math.pi * prog))
    return cfg.lr * (cfg.min_lr_frac + (1 - cfg.min_lr_frac) * coeff)

def train_one(quantize: bool, seed: int, cfg: PoCConfig, loaders, fp_layers=None) -> dict:
    train_loader, val_loader = loaders
    device = cfg.device
    model = build_model(quantize, seed, device)
    if fp_layers:
        for l in fp_layers:
            _set_layer_quantize(model, l, False)
    model.train()
    opt = torch.optim.AdamW(_param_groups(model, cfg.weight_decay), lr=cfg.lr, betas=cfg.betas)
    data_gen = torch.Generator().manual_seed(seed)   

    hist = {"step": [], "train_loss": [], "lr": []}
    eval_hist = {"step": [], "val_loss": []}
    tern_hist = {"step": [], "frac_zero": [], "dist_to_lattice": []}
    t0 = time.time()
    run_loss = None
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

        lv = loss.item()
        run_loss = lv if run_loss is None else 0.9 * run_loss + 0.1 * lv
        if step % cfg.log_every == 0 or step == cfg.steps - 1:
            hist["step"].append(step); hist["train_loss"].append(run_loss); hist["lr"].append(lr)
            if quantize:
                st = M.ternary_layer_stats(model)["overall"]
                tern_hist["step"].append(step)
                tern_hist["frac_zero"].append(st["frac_zero"])
                tern_hist["dist_to_lattice"].append(st["dist_to_lattice"])
        if step % cfg.eval_every == 0 or step == cfg.steps - 1:
            vl = M.eval_loss(model, val_loader, cfg.batch_size, cfg.eval_batches, cfg.amp_dtype, device)
            eval_hist["step"].append(step); eval_hist["val_loss"].append(vl)
            model.train()

    final_val = M.eval_loss(model, val_loader, cfg.batch_size, cfg.eval_batches, cfg.amp_dtype, device)
    return {
        "quantize": quantize,
        "seed": seed,
        "train": hist,
        "eval": eval_hist,
        "tern": tern_hist,
        "final_val_loss": final_val,
        "final_tern_stats": M.ternary_layer_stats(model) if quantize else None,
        "wall_s": time.time() - t0,
        "model": model,
    }

def _mean_lambda_max(model, val_loader, cfg: PoCConfig, params=None) -> float:
    dq = M.dequantized_float_copy(model)
    groups = M.layer_weight_groups(dq) if params == "per_layer" else None
    if params == "per_layer":
        
        x = next(iter(val_loader.iter_eval(cfg.hessian_bsz, 1)))
        out = {}
        for l, ws in groups.items():
            out[l] = M.top_hessian_eigenvalue(dq, x, params=ws, n_iter=cfg.hessian_iters)
        del dq
        return out
    vals = []
    for x in val_loader.iter_eval(cfg.hessian_bsz, cfg.hessian_batches):
        vals.append(M.top_hessian_eigenvalue(dq, x, n_iter=cfg.hessian_iters))
    del dq
    return float(np.mean(vals))

def _set_layer_quantize(model, layer_idx: int, value: bool) -> None:
    for mod in model.model.layers[layer_idx].modules():
        if mod.__class__.__name__ == "BitLinear":
            mod.quantize = value

def _ev(model, val_loader, cfg: PoCConfig) -> float:
    return M.eval_loss(model, val_loader, cfg.batch_size, cfg.eval_batches, cfg.amp_dtype, cfg.device)

def quantize_sensitivity(fp_model, val_loader, cfg: PoCConfig) -> dict:
    n_layers = len(fp_model.model.layers)
    base = _ev(fp_model, val_loader, cfg)
    deltas = {}
    for l in range(n_layers):
        _set_layer_quantize(fp_model, l, True)        
        deltas[l] = _ev(fp_model, val_loader, cfg) - base
        _set_layer_quantize(fp_model, l, False)
    return {"fp_base_val_loss": base, "delta": deltas}

def pareto_quantize(fp_model, val_loader, cfg: PoCConfig, order: list[int]) -> list[float]:
    n_layers = len(fp_model.model.layers)
    losses = []
    for k in range(n_layers + 1):
        q = set(order[:k])
        for l in range(n_layers):
            _set_layer_quantize(fp_model, l, value=(l in q))   
        losses.append(_ev(fp_model, val_loader, cfg))
    for l in range(n_layers):
        _set_layer_quantize(fp_model, l, False)               
    return losses

def relax_sensitivity(tern_model, val_loader, cfg: PoCConfig) -> dict:
    n_layers = len(tern_model.model.layers)
    base = _ev(tern_model, val_loader, cfg)
    deltas = {}
    for l in range(n_layers):
        _set_layer_quantize(tern_model, l, False)
        deltas[l] = _ev(tern_model, val_loader, cfg) - base
        _set_layer_quantize(tern_model, l, True)
    return {"base_val_loss": base, "delta": deltas}

def pareto_curves(model, val_loader, cfg: PoCConfig, order: list[int]) -> list[float]:
    n_layers = len(model.model.layers)
    losses = []
    for k in range(n_layers + 1):
        relaxed = set(order[:k])   
        for l in range(n_layers):
            _set_layer_quantize(model, l, value=(l not in relaxed))
        losses.append(M.eval_loss(model, val_loader, cfg.batch_size, cfg.eval_batches, cfg.amp_dtype, cfg.device))
    for l in range(n_layers):
        _set_layer_quantize(model, l, True)   
    return losses

def spearman(a: list[float], b: list[float]) -> float:
    def rank(x):
        order = sorted(range(len(x)), key=lambda i: x[i])
        r = [0.0] * len(x)
        for pos, i in enumerate(order):
            r[i] = pos
        return r
    ra, rb = rank(a), rank(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    cov = sum((ra[i] - ma) * (rb[i] - mb) for i in range(n))
    va = math.sqrt(sum((ra[i] - ma) ** 2 for i in range(n)))
    vb = math.sqrt(sum((rb[i] - mb) ** 2 for i in range(n)))
    return cov / (va * vb + 1e-12)

def run_poc(cfg: PoCConfig) -> dict:
    os.makedirs(cfg.out_dir, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    train_ids, val_ids = build_tinystories_tokens(cfg.data_dir, cfg.n_train_tokens, cfg.n_val_tokens)
    train_loader = TokenLoader(train_ids, cfg.seq_len, cfg.device)
    val_loader = TokenLoader(val_ids, cfg.seq_len, cfg.device)
    loaders = (train_loader, val_loader)

    results = {"config": asdict(cfg), "runs": {"ternary": [], "fp": []}, "lambda_max": {"ternary": [], "fp": []}}
    seed0_models = {}
    for seed in cfg.seeds:
        for quantize, key in [(False, "fp"), (True, "ternary")]:
            r = train_one(quantize, seed, cfg, loaders)
            model = r.pop("model")
            lam = _mean_lambda_max(model, val_loader, cfg)
            results["lambda_max"][key].append(lam)
            r["lambda_max"] = lam
            results["runs"][key].append(r)
            if seed == cfg.seeds[0]:
                seed0_models[key] = model
            else:
                del model
            torch.cuda.empty_cache()
            print(f"[seed {seed}] {key:8s} final_val={r['final_val_loss']:.4f} "
                  f"lambda_max={lam:.3f} wall={r['wall_s']:.1f}s")

    

    
    def msd(xs):
        a = np.array(xs, dtype=float)
        return {"mean": float(a.mean()), "std": float(a.std()), "n": len(a)}

    tern_finals = [r["final_val_loss"] for r in results["runs"]["ternary"]]
    fp_finals = [r["final_val_loss"] for r in results["runs"]["fp"]]
    gaps = [t - f for t, f in zip(tern_finals, fp_finals)]
    results["summary"] = {
        "ternary_final_val": msd(tern_finals),
        "fp_final_val": msd(fp_finals),
        "ternarization_tax": msd(gaps),
        "lambda_max_ternary": msd(results["lambda_max"]["ternary"]),
        "lambda_max_fp": msd(results["lambda_max"]["fp"]),
    }

    with open(os.path.join(cfg.out_dir, "results.json"), "w") as f:
        json.dump(_jsonable(results), f, indent=2)
    return results

def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o
