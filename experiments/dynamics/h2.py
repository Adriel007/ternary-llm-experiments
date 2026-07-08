
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
from torch.amp import autocast  
from torch.nn.attention import SDPBackend, sdpa_kernel  

from experiments.dynamics import metrics as M  
from experiments.dynamics.data import TokenLoader, build_tinystories_tokens  
from experiments.dynamics.experiment import (  
    PoCConfig, _set_layer_quantize, train_one,
)

CFG = PoCConfig(
    seeds=(0,), steps=1500, batch_size=16, seq_len=128, warmup=100,
    eval_every=300, log_every=100, eval_batches=20,
    n_train_tokens=6_000_000, n_val_tokens=300_000,
    hessian_iters=20, hessian_bsz=8,
    out_dir=os.path.join(ROOT, "artifacts/poc"),
)
N_HUTCH = 12          
DRIVE = "/content/drive/MyDrive/PhD_PoC"

def hessian_trace_layers(dq_model, x, groups, n_hutch=N_HUTCH, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    out = {}
    with sdpa_kernel(SDPBackend.MATH):
        loss = dq_model(input_ids=x, labels=x)["loss"]
        for l, params in groups.items():
            grads = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)
            ests = []
            for _ in range(n_hutch):
                v = [(torch.randint(0, 2, p.shape, generator=g).to(p) * 2 - 1) for p in params]  
                dot = sum((gr * vv).sum() for gr, vv in zip(grads, v))
                Hv = torch.autograd.grad(dot, params, retain_graph=True)
                ests.append(float(sum((hv * vv).sum() for hv, vv in zip(Hv, v)).item()))
            out[l] = float(np.mean(ests))   
    return out

@torch.no_grad()
def causal_importance_layers(model, loader, cfg):
    base = M.eval_loss(model, loader, cfg.batch_size, cfg.eval_batches, cfg.amp_dtype, cfg.device)
    out = {}
    for l, layer in enumerate(model.model.layers):
        def hook(mod, inp, output):
            h_in = inp[0]
            delta = output - h_in
            return h_in + delta.mean(dim=(0, 1), keepdim=True)   
        h = layer.register_forward_hook(hook)
        abl = M.eval_loss(model, loader, cfg.batch_size, cfg.eval_batches, cfg.amp_dtype, cfg.device)
        h.remove()
        out[l] = float(abl - base)   
    return out

def main():
    out = CFG.out_dir
    os.makedirs(out, exist_ok=True)
    status = os.path.join(out, "H2_STATUS.txt")
    open(status, "w").write("RUNNING\n")
    t0 = time.time()
    try:
        tr, va = build_tinystories_tokens(CFG.data_dir, CFG.n_train_tokens, CFG.n_val_tokens)
        loaders = (TokenLoader(tr, CFG.seq_len, CFG.device), TokenLoader(va, CFG.seq_len, CFG.device))
        vl = loaders[1]

        fp = train_one(False, 0, CFG, loaders); fpm = fp.pop("model")
        tern = train_one(True, 0, CFG, loaders); tern.pop("model")
        full_fp, full_tern = fp["final_val_loss"], tern["final_val_loss"]
        print(f"refs: full_fp={full_fp:.4f} full_ternary={full_tern:.4f}", flush=True)

        dq = M.dequantized_float_copy(fpm)
        groups = M.layer_weight_groups(dq)
        xb = next(iter(vl.iter_eval(CFG.hessian_bsz, 1)))
        s_l = hessian_trace_layers(dq, xb, groups)
        del dq
        c_l = causal_importance_layers(fpm, vl, CFG)
        del fpm
        torch.cuda.empty_cache()

        layers = sorted(s_l)
        s_order = sorted(layers, key=lambda l: s_l[l], reverse=True)   
        c_order = sorted(layers, key=lambda l: c_l[l], reverse=True)   
        print("s_l (HAWQ trace):", {l: round(s_l[l], 2) for l in layers}, flush=True)
        print("c_l (causal):    ", {l: round(c_l[l], 4) for l in layers}, flush=True)
        print("s_order:", s_order, "| c_order:", c_order,
              "| identical:", s_order == c_order, flush=True)

        cache = {}
        def fit(fp_layers, seed=0):
            key = (seed, frozenset(fp_layers))
            if key in cache:
                return cache[key]
            r = train_one(True, seed, CFG, loaders, fp_layers=set(fp_layers)); r.pop("model")
            cache[key] = r["final_val_loss"]
            print(f"  seed{seed} fp_layers={sorted(fp_layers)} -> {cache[key]:.4f}", flush=True)
            return cache[key]

        budgets = [1, 2, 3]
        pareto = {"s": {0: full_tern}, "c": {0: full_tern}}
        for k in budgets:
            pareto["s"][k] = fit(s_order[:k])
            pareto["c"][k] = fit(c_order[:k])

        diff_k = next((k for k in budgets if set(s_order[:k]) != set(c_order[:k])), None)
        headline = None
        if diff_k is not None:
            seeds = (0, 1, 2)
            gv = {s: {"s": fit(s_order[:diff_k], s), "c": fit(c_order[:diff_k], s)} for s in seeds}
            gaps = [gv[s]["s"] - gv[s]["c"] for s in seeds]   
            headline = {"budget": diff_k, "per_seed": gv,
                        "c_minus_s_gap_mean": float(np.mean(gaps)), "gap_std": float(np.std(gaps))}

        result = {
            "config": {"steps": CFG.steps, "n_hutch": N_HUTCH},
            "full_fp": full_fp, "full_ternary": full_tern,
            "layers": layers, "s_l_trace": [s_l[l] for l in layers], "c_l_causal": [c_l[l] for l in layers],
            "s_order": s_order, "c_order": c_order, "orderings_identical": s_order == c_order,
            "pareto_s": [pareto["s"][k] for k in [0] + budgets],
            "pareto_c": [pareto["c"][k] for k in [0] + budgets],
            "budgets": [0] + budgets,
            "headline": headline,
        }
        json.dump(result, open(os.path.join(out, "h2.json"), "w"), indent=2)
        
        if os.path.isdir(DRIVE):
            json.dump(result, open(os.path.join(DRIVE, "h2.json"), "w"), indent=2)
        open(status, "w").write(f"DONE {time.time()-t0:.0f}s\n")
        print("H2 DONE", round(time.time() - t0, 1), "s | identical_orderings:", result["orderings_identical"])
        print("pareto_s:", [round(x, 4) for x in result["pareto_s"]])
        print("pareto_c:", [round(x, 4) for x in result["pareto_c"]])
        if headline:
            print("headline c-vs-s gap (>0 => causal better):",
                  round(headline["c_minus_s_gap_mean"], 4), "+/-", round(headline["gap_std"], 4))
    except Exception:
        tb = traceback.format_exc(); print(tb)
        open(status, "w").write("ERROR\n" + tb)
        raise

if __name__ == "__main__":
    main()
