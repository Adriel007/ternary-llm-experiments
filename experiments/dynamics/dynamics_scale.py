
from __future__ import annotations

import json
import math
import os
import sys
import time
import traceback

ROOT = os.environ.get("PHD_ROOT", os.environ.get("POC_ROOT", "/content/PhD-propose"))
sys.path[:0] = [ROOT, os.path.join(ROOT, "packages/bitnet_core/src")]

import numpy as np  
import torch  
from torch.amp import autocast  

from bitnet_core import TernaryConfig, TernaryForCausalLM  
from experiments.dynamics import metrics as M  
from experiments.dynamics.flatness import hessian_trace_total  
from experiments.dynamics.experiment import (  
    PoCConfig, _lr_at, _param_groups, _set_layer_quantize, quantize_sensitivity,
)
from experiments.dynamics.data import VOCAB_SIZE, TokenLoader, build_wikitext_tokens  

OUT = os.path.join(ROOT, "artifacts/poc")

SCALES = [
    ("S", 512, 1408, 8, 8),     
    ("M", 768, 2112, 12, 12),
    ("L", 1024, 2816, 16, 16),
]
SEEDS = (0, 1, 2)

STEPS = 6000
BATCH = 32
SEQ_LEN = 512
LR = 3e-4
WARMUP = 200
N_TRAIN_TOKENS = 110_000_000  
N_VAL_TOKENS = 200_000        
EVAL_BATCHES = 8             
EVAL_BATCHES_FINAL = 12      
TRACE_BATCHES = 4
TRACE_BSZ = 4
N_HUTCH = 8

T_STAR_FRACS = [0.0, 0.25, 0.5, 0.75, 0.9, 1.0]

def build_config(hidden, inter, layers, heads):
    return TernaryConfig(
        vocab_size=VOCAB_SIZE, hidden_size=hidden, intermediate_size=inter,
        num_hidden_layers=layers, num_attention_heads=heads, num_key_value_heads=heads,
        max_position_embeddings=1024,
    )

def n_params(model):
    tot = sum(p.numel() for p in model.parameters())
    emb = model.model.embed_tokens.weight.numel()
    return tot, tot - emb

def build_model(config, quantize, seed, device):
    torch.manual_seed(seed)
    return TernaryForCausalLM(config, quantize=quantize).to(device)

def train(config, quantize, seed, loaders, device, switch_step=None):
    train_loader, val_loader = loaders
    start_q = quantize if switch_step is None else (switch_step <= 0)
    model = build_model(config, True, seed, device)        
    _set_all_quantize(model, start_q)
    model.train()
    opt = torch.optim.AdamW(_param_groups(model, 0.1), lr=LR, betas=(0.9, 0.95))
    gen = torch.Generator().manual_seed(seed)
    hist = {"step": [], "train_loss": [], "val_loss": []}
    tern_hist = {"step": [], "frac_zero": [], "dist_to_lattice": []}
    run = None
    t0 = time.time()
    for step in range(STEPS):
        if switch_step is not None and step == switch_step:
            _set_all_quantize(model, True)
        lr = _lr_at_local(step)
        for pg in opt.param_groups:
            pg["lr"] = lr
        x = train_loader.batch(BATCH, gen)
        with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            loss = model(input_ids=x, labels=x)["loss"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        lv = loss.item()
        run = lv if run is None else 0.9 * run + 0.1 * lv
        if step % 200 == 0 or step == STEPS - 1:
            vl = M.eval_loss(model, val_loader, BATCH, EVAL_BATCHES, torch.bfloat16, device)
            hist["step"].append(step); hist["train_loss"].append(run); hist["val_loss"].append(vl)
            if _ends_ternary(model):
                st = M.ternary_layer_stats(model)["overall"]
                tern_hist["step"].append(step)
                tern_hist["frac_zero"].append(st["frac_zero"])
                tern_hist["dist_to_lattice"].append(st["dist_to_lattice"])
            model.train()
    final_val = M.eval_loss(model, val_loader, BATCH, EVAL_BATCHES_FINAL, torch.bfloat16, device)
    return {"final_val_loss": final_val, "train": hist, "tern": tern_hist,
            "wall_s": time.time() - t0, "model": model}

def _lr_at_local(step):
    if step < WARMUP:
        return LR * (step + 1) / WARMUP
    prog = min(1.0, (step - WARMUP) / max(1, STEPS - WARMUP))
    return LR * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * prog)))

def _set_all_quantize(model, flag):
    for l in range(len(model.model.layers)):
        _set_layer_quantize(model, l, flag)

def _ends_ternary(model):
    for mod in model.model.layers[0].modules():
        if mod.__class__.__name__ == "BitLinear":
            return bool(mod.quantize)
    return False

def trace_ratio(tern_model, fp_model, val_loader, device):
    def mean_trace(model):
        dq = M.dequantized_float_copy(model)
        vals = []
        for x in val_loader.iter_eval(TRACE_BSZ, TRACE_BATCHES):
            t, _ = hessian_trace_total(dq, x, n_hutch=N_HUTCH)
            vals.append(t)
        del dq
        return float(np.mean(vals))
    tt, ft = mean_trace(tern_model), mean_trace(fp_model)
    return {"trace_ternary": tt, "trace_fp": ft, "ratio_tern_over_fp": tt / ft if ft else None}

def _dump(obj, name):
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, name)
    json.dump(_jsonable(obj), open(p + ".tmp", "w"), indent=2)
    os.replace(p + ".tmp", p)

def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    return o

def _eval_cfg(config):
    return PoCConfig(batch_size=BATCH, seq_len=SEQ_LEN, eval_batches=20, device="cuda")

def run_ladder():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("loading wikitext-103 (%dM train / %dM val tokens)..."
          % (N_TRAIN_TOKENS // 1_000_000, N_VAL_TOKENS // 1_000_000), flush=True)
    tr, va = build_wikitext_tokens(os.environ.get("DATA_DIR", "/workspace/data"),
                                   N_TRAIN_TOKENS, N_VAL_TOKENS)
    train_loader = TokenLoader(tr, SEQ_LEN, dev)
    val_loader = TokenLoader(va, SEQ_LEN, dev)
    loaders = (train_loader, val_loader)
    path = os.path.join(OUT, "dynamics_scale_ladder.json")
    if os.path.exists(path):                                       
        results = json.load(open(path))
        print("RESUME from", {k: list(results["scales"][k].get("seeds", {}))
                              for k in results["scales"]}, flush=True)
    else:
        results = {"phase": "ladder", "corpus": "wikitext-103", "steps": STEPS, "batch": BATCH,
                   "seq_len": SEQ_LEN, "seeds": list(SEEDS), "scales": {}}
        _dump(results, "dynamics_scale_ladder.json")
    for name, h, i, L, nh in SCALES:
        cfg = build_config(h, i, L, nh)
        if name in results["scales"] and "summary" in results["scales"][name]:
            print("skip %s (complete)" % name, flush=True); continue
        sc = results["scales"].get(name, {"hidden": h, "intermediate": i, "layers": L,
                                          "heads": nh, "seeds": {}})
        sc.setdefault("seeds", {})
        for seed in SEEDS:
            if str(seed) in sc["seeds"]:
                print("skip %s seed%d (done)" % (name, seed), flush=True); continue
            t0 = time.time()
            rf = train(cfg, False, seed, loaders, dev)               
            fp_val, fp_model = rf["final_val_loss"], rf["model"]
            rt = train(cfg, True, seed, loaders, dev)                
            tern_val, tern_model = rt["final_val_loss"], rt["model"]
            tax = tern_val - fp_val
            tr_ratio = trace_ratio(tern_model, fp_model, val_loader, dev)
            
            sens = quantize_sensitivity(fp_model, val_loader, _eval_cfg(cfg))
            d = sens["delta"]                       
            dl = np.array([d[k] for k in sorted(d, key=int)])
            l0 = float(dl[0]); med = float(np.median(dl)); mx = int(np.argmax(dl))
            tp, np_ = n_params(tern_model)
            sc["seeds"][str(seed)] = {
                "fp_val": fp_val, "tern_val": tern_val, "tax": tax,
                "trace": tr_ratio, "layer0_delta": l0, "median_delta": med,
                "argmax_layer": mx, "l0_over_median": l0 / med if med else None,
                "params_total": tp, "params_nonembed": np_,
                "wall_s": rf["wall_s"] + rt["wall_s"]}
            print("[%s seed%d] params=%.0fM(nonemb %.0fM) tax=%.4f trace_ratio=%.2f "
                  "L0/med=%.2f argmax=L%d (%.0fs)"
                  % (name, seed, tp / 1e6, np_ / 1e6, tax, tr_ratio["ratio_tern_over_fp"],
                     l0 / med if med else float("nan"), mx, sc["seeds"][str(seed)]["wall_s"]),
                  flush=True)
            del fp_model, tern_model
            torch.cuda.empty_cache()
            results["scales"][name] = sc
            _dump(results, "dynamics_scale_ladder.json")
        sd = sc["seeds"]                                    
        taxes = [sd[s]["tax"] for s in sd]
        trs = [sd[s]["trace"]["ratio_tern_over_fp"] for s in sd]
        l0s = [sd[s]["l0_over_median"] for s in sd if sd[s]["l0_over_median"] is not None]
        sc["summary"] = {"tax_mean": float(np.mean(taxes)), "tax_sd": float(np.std(taxes)),
                         "trace_ratio_mean": float(np.mean(trs)),
                         "l0_dom_mean": float(np.nanmean(l0s)) if l0s else float("nan")}
        results["scales"][name] = sc
        _dump(results, "dynamics_scale_ladder.json")
        print("== %s: tax %.4f±%.4f | trace_ratio %.2f | L0/med %.2f =="
              % (name, sc["summary"]["tax_mean"], sc["summary"]["tax_sd"],
                 sc["summary"]["trace_ratio_mean"], sc["summary"]["l0_dom_mean"]), flush=True)
    print("\n==== LADDER SUMMARY (does the toy story persist with scale?) ====", flush=True)
    for name, *_ in SCALES:
        s = results["scales"][name]["summary"]
        print("  %s  tax=%.4f  trace_ratio=%.2f  L0/med=%.2f"
              % (name, s["tax_mean"], s["trace_ratio_mean"], s["l0_dom_mean"]))
    return results

def run_tstar():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    name, h, i, L, nh = SCALES[1]       
    cfg = build_config(h, i, L, nh)
    tr, va = build_wikitext_tokens(os.environ.get("DATA_DIR", "/workspace/data"),
                                   N_TRAIN_TOKENS, N_VAL_TOKENS)
    loaders = (TokenLoader(tr, SEQ_LEN, dev), TokenLoader(va, SEQ_LEN, dev))
    tpath = os.path.join(OUT, "dynamics_scale_tstar.json")
    if os.path.exists(tpath):                                  
        results = json.load(open(tpath))
        print("RESUME tstar: %d rows done" % len(results["rows"]), flush=True)
    else:
        results = {"phase": "tstar", "scale": name, "hidden": h, "layers": L,
                   "corpus": "wikitext-103", "steps": STEPS, "seeds": list(SEEDS), "rows": []}
        _dump(results, "dynamics_scale_tstar.json")
    done = {(r["seed"], r["t_frac"]) for r in results["rows"]}
    for seed in SEEDS:
        for frac in T_STAR_FRACS:
            if (seed, frac) in done:
                print("skip tstar seed%d t*=%.2f (done)" % (seed, frac), flush=True); continue
            sw = int(frac * STEPS)
            r = train(cfg, True, seed, loaders, dev, switch_step=sw)
            row = {"seed": seed, "t_frac": frac, "switch_step": sw,
                   "final_val_loss": r["final_val_loss"], "wall_s": r["wall_s"]}
            results["rows"].append(row)
            _dump(results, "dynamics_scale_tstar.json")
            print("[tstar seed%d t*=%.2f sw=%d] final_val=%.4f (%.0fs)"
                  % (seed, frac, sw, r["final_val_loss"], r["wall_s"]), flush=True)
            del r
            torch.cuda.empty_cache()
    
    print("\n==== t* SWEEP (top scale %s): does the U persist? ====" % name, flush=True)
    for frac in T_STAR_FRACS:
        vals = [r["final_val_loss"] for r in results["rows"] if r["t_frac"] == frac]
        print("  t*=%.2f  final_val=%.4f ± %.4f" % (frac, float(np.mean(vals)), float(np.std(vals))))
    return results

def main():
    phase = sys.argv[1] if len(sys.argv) > 1 else "ladder"
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        res = run_ladder() if phase == "ladder" else run_tstar()
        _dump(res, "dynamics_scale_%s.json" % phase)
        print("\nDONE %s" % phase, flush=True)
    except Exception:
        tb = traceback.format_exc()
        print(tb, flush=True)
        open(os.path.join(OUT, "DYNAMICS_SCALE_%s_STATUS.txt" % phase.upper()), "w").write("ERROR\n" + tb)
        raise

if __name__ == "__main__":
    main()
