
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
import torch.nn.functional as TF  

from experiments.interp.h2_2b_correction import (  
    MID, OUT, install_unified_forward, _bitlinears, build_corpora, eval_loss, ppl,
    RANK, ALPHA, LR, WD, MAX_STEPS, EVAL_EVERY, PATIENCE, SEED,
)

RANKINGS = {
    "c_causal": [0, 1, 2, 29, 19, 5, 4, 28, 27, 9, 25, 3, 20, 26, 13, 8, 24, 10, 23, 14,
                 21, 7, 15, 12, 22, 16, 18, 11, 6, 17],
    "random":   [2, 11, 26, 21, 10, 4, 28, 16, 23, 6, 18, 25, 3, 29, 8, 0, 19, 12, 20, 13,
                 7, 5, 17, 14, 22, 9, 27, 24, 1, 15],
}
BUDGETS = [1, 2, 3, 5, 8]
BLOCKS = ["attn", "mlp", "both"]

OUT_JSON = os.path.join(OUT, "h2_2b_granularity.json")
STATUS = os.path.join(OUT, "H2_2B_GRANULARITY_STATUS.txt")

def _dump(obj):
    os.makedirs(OUT, exist_ok=True)
    tmp = OUT_JSON + ".tmp"
    json.dump(obj, open(tmp, "w"), indent=2)
    os.replace(tmp, OUT_JSON)

def _status(s):
    os.makedirs(OUT, exist_ok=True)
    open(STATUS, "w").write(s + "\n")

def _block_of(layer, bl):
    for name, m in layer.named_modules():
        if m is bl:
            if name.startswith("self_attn"):
                return "attn"
            if name.startswith("mlp"):
                return "mlp"
            return "other"
    return "other"

def attach_tagged(model, rank, alpha, device, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    sc = alpha / rank
    per_layer = {}
    extra = {"attn": None, "mlp": None}
    backbone = None
    for li, layer in enumerate(model.model.layers):
        groups = {"attn": [], "mlp": []}
        n_extra = {"attn": 0, "mlp": 0}
        n_back = 0
        for bl in _bitlinears(layer):
            blk = _block_of(layer, bl)
            d_out, d_in = bl.weight.shape
            A = (torch.randn(rank, d_in, generator=g) / math.sqrt(d_in)).to(device=device, dtype=torch.float32)
            B = torch.zeros(d_out, rank, device=device, dtype=torch.float32)
            A.requires_grad_(False); B.requires_grad_(False)
            bl._adapter = (A, B, sc)
            bl._adapter_active = False
            bl._block = blk
            n_back += bl.weight.numel()
            if blk in groups:
                groups[blk].append((A, B))
                n_extra[blk] += A.numel() + B.numel()
        per_layer[li] = groups
        if backbone is None:
            backbone, extra["attn"], extra["mlp"] = n_back, n_extra["attn"], n_extra["mlp"]
    return per_layer, {k: extra[k] / backbone for k in extra}

def set_active_block(model, layer_idxs, block):
    sel = set(int(i) for i in layer_idxs)
    for li, layer in enumerate(model.model.layers):
        for bl in _bitlinears(layer):
            on = (li in sel) and (block == "both" or getattr(bl, "_block", "other") == block)
            bl._adapter_active = on

def _flat_params(per_layer, layers, block):
    out = []
    for li in layers:
        groups = per_layer[li]
        if block in ("attn", "both"):
            out += [p for AB in groups["attn"] for p in AB]
        if block in ("mlp", "both"):
            out += [p for AB in groups["mlp"] for p in AB]
    return out

def joint_train_eval_block(model, per_layer, layers, block, train_b, val_b, test_b, seed):
    flat = _flat_params(per_layer, layers, block)
    
    for li in layers:
        groups = per_layer[li]
        sel = (groups["attn"] if block in ("attn", "both") else []) +              (groups["mlp"] if block in ("mlp", "both") else [])
        for (A, B) in sel:
            B.zero_()
    for p in flat:
        p.requires_grad_(True)
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(flat, lr=LR, weight_decay=WD)
    set_active_block(model, layers, block)
    nb, best, bad, best_state = len(train_b), float("inf"), 0, None
    for step in range(1, MAX_STEPS + 1):
        b = train_b[(step - 1) % nb]
        loss = model(input_ids=b, labels=b).loss
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if step % EVAL_EVERY == 0 or step == MAX_STEPS:
            vl = eval_loss(model, val_b)
            if vl < best - 1e-4:
                best, bad, best_state = vl, 0, [p.detach().clone() for p in flat]
            else:
                bad += 1
                if bad >= PATIENCE:
                    break
    if best_state is not None:
        with torch.no_grad():
            for p, pb in zip(flat, best_state):
                p.copy_(pb)
    test_loss = eval_loss(model, test_b)
    for p in flat:
        p.requires_grad_(False)
    set_active_block(model, [], "both")
    return float(test_loss)

def main():
    _status("LOADING")
    t0 = time.time()
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        tok = AutoTokenizer.from_pretrained(MID)
        model = AutoModelForCausalLM.from_pretrained(MID, dtype=torch.bfloat16).to(dev).eval()
        model.requires_grad_(False)
        n_layers = len(model.model.layers)
        install_unified_forward()
        per_layer, extra_frac = attach_tagged(model, RANK, ALPHA, dev, SEED)

        train_b, val_b, test_b, _calib_b = build_corpora(tok, dev)
        set_active_block(model, [], "both")
        base_test = eval_loss(model, test_b)
        
        print("loaded %s | layers %d | extra-frac attn %.4f mlp %.4f | base test ppl %.4f"
              % (MID, n_layers, extra_frac["attn"], extra_frac["mlp"], ppl(base_test)), flush=True)

        results = {"model": MID, "rank": RANK, "alpha": ALPHA, "seed": SEED,
                   "extra_frac_per_layer": extra_frac, "budgets": BUDGETS,
                   "base_test_ppl": ppl(base_test), "base_test_loss": base_test,
                   "n_test_seqs": len(test_b) * (test_b[0].shape[0] if test_b else 0),
                   "per_layer_recoverability": {}, "pareto": {}}
        _dump(results)

        _status("PART A per-layer recoverability")
        for li in range(n_layers):
            rec = {}
            for blk in ("attn", "mlp"):
                t = time.time()
                loss_b = joint_train_eval_block(model, per_layer, [li], blk,
                                                train_b, val_b, test_b, SEED * 100 + li)
                rec[blk] = {"test_ppl": ppl(loss_b), "recovered_dloss": float(base_test - loss_b)}
                print("  [recov] L%2d %-4s ppl %.3f->%.3f dloss=%+.4f (%.1fs)"
                      % (li, blk, ppl(base_test), ppl(loss_b), rec[blk]["recovered_dloss"],
                         time.time() - t), flush=True)
            results["per_layer_recoverability"][str(li)] = rec
            _dump(results)

        _status("PART B block pareto")
        for ranking_name, order in RANKINGS.items():
            results["pareto"].setdefault(ranking_name, {})
            for blk in BLOCKS:
                results["pareto"][ranking_name].setdefault(blk, {})
                for B in BUDGETS:
                    t = time.time()
                    layers = sorted(order[:B])
                    loss_b = joint_train_eval_block(model, per_layer, layers, blk,
                                                    train_b, val_b, test_b, SEED * 7 + B)
                    mem = B * (extra_frac["attn"] + extra_frac["mlp"] if blk == "both"
                               else extra_frac[blk])
                    results["pareto"][ranking_name][blk][str(B)] = {
                        "layers": layers, "test_ppl": ppl(loss_b), "test_loss": loss_b,
                        "extra_mem_frac": mem}
                    _dump(results)
                    print("  [pareto] %-9s %-4s B=%2d mem=%.4f layers %s -> ppl=%.3f (%.1fs)"
                          % (ranking_name, blk, B, mem, layers, ppl(loss_b), time.time() - t),
                          flush=True)

        results["wall_s"] = time.time() - t0
        _dump(results)

        print("\n==== GRANULARITY SUMMARY ====", flush=True)
        rec = results["per_layer_recoverability"]
        att = np.array([rec[str(l)]["attn"]["recovered_dloss"] for l in range(n_layers)])
        mlp = np.array([rec[str(l)]["mlp"]["recovered_dloss"] for l in range(n_layers)])
        print("per-layer recovered dloss: attn mean %+.4f (max %+.4f @L%d) | mlp mean %+.4f (max %+.4f @L%d)"
              % (att.mean(), att.max(), int(att.argmax()), mlp.mean(), mlp.max(), int(mlp.argmax())))
        print("layers where attn>mlp: %d/%d" % (int((att > mlp).sum()), n_layers))
        print("\nPareto (c_causal) ppl by block at each budget (mem-frac in []):")
        for B in BUDGETS:
            row = results["pareto"]["c_causal"]
            print("  B=%2d  attn %.3f [%.4f]  mlp %.3f [%.4f]  both %.3f [%.4f]"
                  % (B, row["attn"][str(B)]["test_ppl"], row["attn"][str(B)]["extra_mem_frac"],
                     row["mlp"][str(B)]["test_ppl"], row["mlp"][str(B)]["extra_mem_frac"],
                     row["both"][str(B)]["test_ppl"], row["both"][str(B)]["extra_mem_frac"]))
        _status("DONE %.0fs" % (time.time() - t0))
        print("\nDONE %.1fs" % (time.time() - t0), flush=True)
    except Exception:
        tb = traceback.format_exc()
        print(tb, flush=True)
        _status("ERROR\n" + tb)
        raise

if __name__ == "__main__":
    main()
