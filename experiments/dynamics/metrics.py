
from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
from torch.amp import autocast
from torch.nn.attention import SDPBackend, sdpa_kernel

from bitnet_core.quant import quantize_weight_ste

_EPS = 1e-5

def _is_bitlinear(m: nn.Module) -> bool:
    return m.__class__.__name__ == "BitLinear"

@torch.no_grad()
def ternary_layer_stats(model: nn.Module) -> dict:
    per_layer = {}
    tot_zero = tot_neg = tot_pos = tot_n = 0
    dist_accum = 0.0
    for name, m in model.named_modules():
        if not _is_bitlinear(m):
            continue
        w = m.weight.detach().float()
        beta = w.abs().mean().clamp(min=_EPS)
        ws = w / beta                      
        codes = ws.round().clamp(-1, 1)
        n = w.numel()
        z = (codes == 0).sum().item()
        ng = (codes < 0).sum().item()
        ps = (codes > 0).sum().item()
        dist = (ws - codes).abs().mean().item()   
        per_layer[name] = {
            "beta": beta.item(),
            "frac_zero": z / n,
            "frac_neg": ng / n,
            "frac_pos": ps / n,
            "dist_to_lattice": dist,
            "n": n,
        }
        tot_zero += z; tot_neg += ng; tot_pos += ps; tot_n += n
        dist_accum += dist * n
    overall = {
        "frac_zero": tot_zero / tot_n,
        "frac_neg": tot_neg / tot_n,
        "frac_pos": tot_pos / tot_n,
        "dist_to_lattice": dist_accum / tot_n,
        "n": tot_n,
    }
    return {"overall": overall, "per_layer": per_layer}

def dequantized_float_copy(model: nn.Module) -> nn.Module:
    m = copy.deepcopy(model)
    for mod in m.modules():
        if _is_bitlinear(mod):
            if mod.quantize:
                w_real = quantize_weight_ste(mod.weight.detach())
                mod.weight = nn.Parameter(w_real.detach().clone())
            mod.quantize = False
    return m.float()

def layer_weight_groups(model: nn.Module) -> dict:
    groups = {}
    for i, layer in enumerate(model.model.layers):
        groups[i] = [mod.weight for mod in layer.modules() if _is_bitlinear(mod)]
    return groups

def _loss_on(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    return model(input_ids=x, labels=x)["loss"]

def _power(matvec, v, n_iter: int, tol: float):
    eig = eig_prev = 0.0
    for _ in range(n_iter):
        Hv = matvec(v)
        eig = sum((hv * vv).sum() for hv, vv in zip(Hv, v)).item()  
        v = [hv.detach() for hv in Hv]
        _normalize_(v)
        if abs(eig - eig_prev) <= tol * (abs(eig) + 1e-9):
            break
        eig_prev = eig
    return eig, v

def top_hessian_eigenvalue(
    model: nn.Module,
    x: torch.Tensor,
    params: list[torch.nn.Parameter] | None = None,
    n_iter: int = 20,
    tol: float = 1e-3,
    seed: int = 0,
) -> float:
    if params is None:
        params = [p for p in model.parameters() if p.requires_grad]
    device = params[0].device
    g = torch.Generator(device="cpu").manual_seed(seed)

    def _rand_unit():
        v = [torch.randn(p.shape, generator=g).to(device) for p in params]
        _normalize_(v)
        return v

    
    with sdpa_kernel(SDPBackend.MATH):
        loss = _loss_on(model, x)
        grads = torch.autograd.grad(loss, params, create_graph=True)

        def hvp(v):
            dot = sum((gr * vv).sum() for gr, vv in zip(grads, v))
            return torch.autograd.grad(dot, params, retain_graph=True)

        lm, _ = _power(hvp, _rand_unit(), n_iter, tol)        
        if lm >= 0:
            return float(lm)
        c = abs(lm) * 1.1                                      

        def shifted(v):
            return [hv + c * vv for hv, vv in zip(hvp(v), v)]

        top_c, _ = _power(shifted, _rand_unit(), n_iter, tol)  
        return float(top_c - c)

def _normalize_(v: list[torch.Tensor]) -> None:
    norm = math.sqrt(sum((vv * vv).sum().item() for vv in v)) + 1e-12
    for vv in v:
        vv.div_(norm)

@torch.no_grad()
def eval_loss(
    model: nn.Module,
    loader,
    batch_size: int,
    n_batches: int,
    amp_dtype: torch.dtype,
    device: str,
    seed: int = 1234,
) -> float:
    was_training = model.training
    model.eval()
    losses = []
    use_amp = device == "cuda" and amp_dtype is not None
    for x in loader.iter_eval(batch_size, n_batches, seed):
        with autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            losses.append(_loss_on(model, x).item())
    if was_training:
        model.train()
    return float(sum(losses) / len(losses))

def perplexity(loss: float) -> float:
    return math.exp(loss)
