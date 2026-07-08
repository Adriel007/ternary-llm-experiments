
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
    MID, OUT, _bitlinears, build_corpora, eval_loss, ppl,
    RANK, ALPHA, LR, WD, MAX_STEPS, EVAL_EVERY, PATIENCE, SEED,
)

RANKINGS = {
    "c_causal": [0, 1, 2, 29, 19, 5, 4, 28, 27, 9, 25, 3, 20, 26, 13, 8, 24, 10, 23, 14,
                 21, 7, 15, 12, 22, 16, 18, 11, 6, 17],
}
BUDGETS = [1, 2, 3, 5, 8]
VARIANTS = ["static", "scalar_gate", "channel_gate", "input_gate"]
GATE_RANK = 4   

OUT_JSON = os.path.join(OUT, "h2_2b_gate.json")
STATUS = os.path.join(OUT, "H2_2B_GATE_STATUS.txt")

def _dump(obj):
    os.makedirs(OUT, exist_ok=True)
    tmp = OUT_JSON + ".tmp"
    json.dump(obj, open(tmp, "w"), indent=2)
    os.replace(tmp, OUT_JSON)

def _status(s):
    os.makedirs(OUT, exist_ok=True)
    open(STATUS, "w").write(s + "\n")

def install_gated_forward():
    from transformers.integrations.bitnet import AutoBitLinear, WeightQuant, ActQuant

    def patched_forward(self, input):
        x = self.rms_norm(input) if self.rms_norm is not None else input
        weight = WeightQuant.apply(self.weight)
        out = TF.linear(ActQuant.apply(x), weight, self.bias)
        ad = getattr(self, "_adapter", None)
        if ad is not None and getattr(self, "_adapter_active", False):
            A, B, sc = ad
            corr = TF.linear(TF.linear(x.to(A.dtype), A), B) * sc      
            gm = getattr(self, "_gate_mode", None)
            if gm is not None:
                g = self._gate
                if gm == "scalar":
                    gate = torch.sigmoid(g["s"])
                elif gm == "channel":
                    gate = torch.sigmoid(g["b"])
                elif gm == "input":
                    h = TF.linear(TF.linear(x.to(A.dtype), g["W1"]), g["W2"]) + g["b_in"]
                    gate = torch.sigmoid(h)
                corr = corr * gate.to(corr.dtype)
            out = out + corr.to(out.dtype)
        return out

    AutoBitLinear.forward = patched_forward

def attach_gated(model, rank, alpha, device, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    sc = alpha / rank
    per_layer = {}
    nb_lora = nb_back = 0
    nb_gate = {"scalar": 0, "channel": 0, "input": 0}
    for li, layer in enumerate(model.model.layers):
        mods = []
        for bl in _bitlinears(layer):
            d_out, d_in = bl.weight.shape
            A = (torch.randn(rank, d_in, generator=g) / math.sqrt(d_in)).to(device, torch.float32)
            B = torch.zeros(d_out, rank, device=device, dtype=torch.float32)
            
            s = torch.full((1,), 2.0, device=device, dtype=torch.float32)
            bch = torch.full((d_out,), 2.0, device=device, dtype=torch.float32)
            W1 = (torch.randn(GATE_RANK, d_in, generator=g) / math.sqrt(d_in)).to(device, torch.float32)
            W2 = torch.zeros(d_out, GATE_RANK, device=device, dtype=torch.float32)  
            bin_ = torch.full((d_out,), 2.0, device=device, dtype=torch.float32)
            for p in (A, B, s, bch, W1, W2, bin_):
                p.requires_grad_(False)
            bl._adapter = (A, B, sc)
            bl._adapter_active = False
            bl._gate = {"s": s, "b": bch, "W1": W1, "W2": W2, "b_in": bin_}
            bl._gate_mode = None
            mods.append(bl)
            if li == 0:
                nb_back += bl.weight.numel()
                nb_lora += A.numel() + B.numel()
                nb_gate["scalar"] += 1
                nb_gate["channel"] += d_out
                nb_gate["input"] += W1.numel() + W2.numel() + d_out
        per_layer[li] = mods
    frac = {"lora": nb_lora / nb_back}
    for k in nb_gate:
        frac["gate_" + k] = nb_gate[k] / nb_back
    return per_layer, frac

def set_active(model, layer_idxs, variant):
    gm = {"static": None, "scalar_gate": "scalar", "channel_gate": "channel",
          "input_gate": "input"}[variant]
    sel = set(int(i) for i in layer_idxs)
    for li, layer in enumerate(model.model.layers):
        for bl in _bitlinears(layer):
            bl._adapter_active = li in sel
            bl._gate_mode = gm if li in sel else None

def _trainable(model, layers, variant):
    sel = set(int(i) for i in layers)
    out = []
    for li, layer in enumerate(model.model.layers):
        if li not in sel:
            continue
        for bl in _bitlinears(layer):
            A, B, _ = bl._adapter
            out += [A, B]
            g = bl._gate
            if variant == "scalar_gate":
                out += [g["s"]]
            elif variant == "channel_gate":
                out += [g["b"]]
            elif variant == "input_gate":
                out += [g["W1"], g["W2"], g["b_in"]]
    return out

def joint_train_eval_gated(model, layers, variant, train_b, val_b, test_b, seed):
    sel = set(int(i) for i in layers)
    for li, layer in enumerate(model.model.layers):
        if li in sel:
            for bl in _bitlinears(layer):
                bl._adapter[1].zero_()                      
                bl._gate["W2"].zero_()                      
    flat = _trainable(model, layers, variant)
    for p in flat:
        p.requires_grad_(True)
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(flat, lr=LR, weight_decay=WD)
    set_active(model, layers, variant)
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
    set_active(model, [], "static")
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
        install_gated_forward()
        per_layer, frac = attach_gated(model, RANK, ALPHA, dev, SEED)
        train_b, val_b, test_b, _calib_b = build_corpora(tok, dev)
        set_active(model, [], "static")
        base_test = eval_loss(model, test_b)
        print("loaded %s | layers %d | base test ppl %.4f | frac %s"
              % (MID, n_layers, ppl(base_test), {k: round(v, 5) for k, v in frac.items()}),
              flush=True)

        results = {"model": MID, "rank": RANK, "alpha": ALPHA, "gate_rank": GATE_RANK, "seed": SEED,
                   "extra_frac_per_layer": frac, "budgets": BUDGETS,
                   "base_test_ppl": ppl(base_test), "base_test_loss": base_test,
                   "ranking": "c_causal", "pareto": {}}
        _dump(results)

        order = RANKINGS["c_causal"]
        for variant in VARIANTS:
            results["pareto"].setdefault(variant, {})
            for B in BUDGETS:
                t = time.time()
                layers = sorted(order[:B])
                loss_b = joint_train_eval_gated(model, layers, variant, train_b, val_b, test_b,
                                                SEED * 7 + B)
                gk = {"static": "lora", "scalar_gate": "gate_scalar",
                      "channel_gate": "gate_channel", "input_gate": "gate_input"}[variant]
                mem = B * (frac["lora"] + (0.0 if variant == "static" else frac[gk]))
                results["pareto"][variant][str(B)] = {
                    "layers": layers, "test_ppl": ppl(loss_b), "test_loss": loss_b,
                    "extra_mem_frac": mem}
                _dump(results)
                print("  [pareto] %-13s B=%2d mem=%.4f -> ppl=%.3f (%.1fs)"
                      % (variant, B, mem, ppl(loss_b), time.time() - t), flush=True)

        results["wall_s"] = time.time() - t0
        _dump(results)

        print("\n==== GATE SUMMARY (c_causal, base ppl %.3f) ====" % ppl(base_test), flush=True)
        P = results["pareto"]
        for B in BUDGETS:
            cells = " ".join("%s=%.3f" % (v.replace("_gate", "G").replace("static", "stat"),
                                          P[v][str(B)]["test_ppl"]) for v in VARIANTS)
            best = min(VARIANTS, key=lambda v: P[v][str(B)]["test_ppl"])
            print("  B=%2d  %s   | best=%s" % (B, cells, best))
        
        wins = {v: sum(P[v][str(B)]["test_ppl"] < P["static"][str(B)]["test_ppl"] for B in BUDGETS)
                for v in VARIANTS if v != "static"}
        print("gate beats static (of %d budgets): %s" % (len(BUDGETS), wins))
        _status("DONE %.0fs" % (time.time() - t0))
        print("\nDONE %.1fs" % (time.time() - t0), flush=True)
    except Exception:
        tb = traceback.format_exc()
        print(tb, flush=True)
        _status("ERROR\n" + tb)
        raise

if __name__ == "__main__":
    main()
