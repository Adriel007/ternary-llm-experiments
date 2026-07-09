from __future__ import annotations

import math

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

def _params(model):
    return [p for p in model.parameters() if p.requires_grad]

def _rademacher_like(params, gen):
    return [(torch.randint(0, 2, p.shape, generator=gen).to(p) * 2 - 1) for p in params]

def filter_scales(params):
    scales = []
    for p in params:
        w = p.detach()
        if w.dim() >= 2:
            flat = w.reshape(w.shape[0], -1)
            rn = flat.norm(dim=1, keepdim=True).clamp_min(1e-8)   
            scales.append(rn.expand_as(flat).reshape(w.shape))
        else:
            scales.append(w.abs().clamp_min(1e-8))
    return scales

def hutchinson_trace(model, x, scaling=None, n_probe=60, seed=0, params=None):
    params = params if params is not None else _params(model)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    with sdpa_kernel(SDPBackend.MATH):
        loss = model(input_ids=x, labels=x)["loss"]
        grads = torch.autograd.grad(loss, params, create_graph=True)
        acc = 0.0
        for _ in range(n_probe):
            v = _rademacher_like(params, gen)
            if scaling is not None:
                v = [vi * si for vi, si in zip(v, scaling)]
            dot = sum((g * vi).sum() for g, vi in zip(grads, v))
            Hv = torch.autograd.grad(dot, params, retain_graph=True)
            if scaling is not None:
                Hv = [hi * si for hi, si in zip(Hv, scaling)]
            acc += float(sum((hi * vi).sum() for hi, vi in zip(Hv, v)).item())
    return acc / n_probe

def fisher_diag_trace(model, x, params=None):
    params = params if params is not None else _params(model)
    logits = model(input_ids=x)["logits"]
    logp = torch.log_softmax(logits, dim=-1)
    y = torch.distributions.Categorical(logits=logits).sample()
    nll = torch.nn.functional.nll_loss(logp.reshape(-1, logp.size(-1)), y.reshape(-1))
    grads = torch.autograd.grad(nll, params)
    return float(sum((g.detach() ** 2).sum().item() for g in grads))

def early_stop_to_loss(fp_checkpoints, target_loss, eval_loss):
    best, best_gap = None, math.inf
    for ckpt in fp_checkpoints:
        gap = abs(eval_loss(ckpt) - target_loss)
        if gap < best_gap:
            best, best_gap = ckpt, gap
    return best, best_gap

def compare_all(dq_tern, dq_fp_matched, x, n_probe=60, seeds=(0, 1, 2), params_fn=None):
    def agg(model):
        p = params_fn(model) if params_fn is not None else _params(model)
        fsc = filter_scales(p)
        raw = [hutchinson_trace(model, x, None, n_probe, s, p) for s in seeds]
        fnm = [hutchinson_trace(model, x, fsc, n_probe, s, p) for s in seeds]
        fis = fisher_diag_trace(model, x, p)
        half = lambda xs: (sum(xs) / len(xs), (max(xs) - min(xs)) / 2)
        return {"raw": half(raw), "filter_norm": half(fnm), "fisher": (fis, 0.0)}
    return {"ternary": agg(dq_tern), "fp_matched": agg(dq_fp_matched)}

def print_decision(report):
    t, f = report["ternary"], report["fp_matched"]
    print(f"{'lens':>14} {'ternary':>20} {'fp_matched':>20} {'ratio t/fp':>12}")
    ratios = {}
    for k in ("raw", "filter_norm", "fisher"):
        (tm, te), (fm, fe) = t[k], f[k]
        r = tm / fm if fm else float("nan")
        ratios[k] = r
        print(f"{k:>14} {tm:>12.2f}+/-{te:<6.2f} {fm:>12.2f}+/-{fe:<6.2f} {r:>12.3f}")
    r = ratios["filter_norm"]
    if r > 1.15:
        verdict = "SIGNAL SURVIVES normalization -> genuinely sharper; the headline stands."
    elif r < 0.87:
        verdict = "SIGNAL INVERTS -> the raw effect was a scale artifact; rewrite the headline."
    else:
        verdict = "SIGNAL VANISHES -> curvature comparable under invariant metrics; demote it."
    print(f"\nfilter-norm ratio = {r:.3f}  =>  {verdict}")
    return ratios
