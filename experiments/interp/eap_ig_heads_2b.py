
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

from experiments.interp.eap_ig_2b import (  
    MID, OUT, DRIVE, _node_modules, _metric, _cache_run, _patched_metric, _build_pairs,
)
from experiments.interp.eap_ig_circuits_2b import _build_ioi  
from experiments.interp.eap_ig_positions_2b import _patched_metric_pos  

MIN_PER_FAMILY = 5

def _head_dims(model):
    cfg = model.config
    H = cfg.num_attention_heads
    hd = getattr(cfg, "head_dim", None) or (cfg.hidden_size // H)
    return H, hd

def _oproj(model, layer):
    return model.model.layers[layer].self_attn.o_proj

@torch.no_grad()
def _cache_oproj(model, ids, ans_c, ans_k):
    n = len(model.model.layers)
    store = {}

    def mk(i):
        def pre(m, args):
            store[i] = args[0].detach()
        return pre

    handles = [_oproj(model, i).register_forward_pre_hook(mk(i)) for i in range(n)]
    out = model(input_ids=ids)
    for h in handles:
        h.remove()
    return [store[i] for i in range(n)], _metric(out.logits[0, -1], ans_c, ans_k).item()

@torch.no_grad()
def _patch_head(model, base_ids, clean_oproj, layer, head, hd, positions, ans_c, ans_k):
    pos = torch.as_tensor(positions, device=base_ids.device, dtype=torch.long)
    sl = slice(head * hd, (head + 1) * hd)
    cl = clean_oproj[layer]

    def pre(m, args):
        x = args[0].clone()
        x[:, pos, sl] = cl[:, pos, sl]
        return (x,) + tuple(args[1:])

    h = _oproj(model, layer).register_forward_pre_hook(pre)
    out = model(input_ids=base_ids)
    h.remove()
    return _metric(out.logits[0, -1], ans_c, ans_k).item()

@torch.no_grad()
def _patch_layer_allheads(model, base_ids, clean_oproj, layer, positions, ans_c, ans_k):
    pos = torch.as_tensor(positions, device=base_ids.device, dtype=torch.long)
    cl = clean_oproj[layer]

    def pre(m, args):
        x = args[0].clone()
        x[:, pos, :] = cl[:, pos, :]
        return (x,) + tuple(args[1:])

    h = _oproj(model, layer).register_forward_pre_hook(pre)
    out = model(input_ids=base_ids)
    h.remove()
    return _metric(out.logits[0, -1], ans_c, ans_k).item()

def _solved(model, nodes, p):
    _, Lc = _cache_run(model, nodes, p["clean_ids"], p["ans_clean"], p["ans_corrupt"])
    _, Lk = _cache_run(model, nodes, p["corrupt_ids"], p["ans_clean"], p["ans_corrupt"])
    p["L_clean"], p["L_corrupt"] = Lc, Lk
    return Lc > 0 and Lk < 0

def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    status = os.path.join(OUT, "EAPIG4_STATUS.txt")
    open(status, "w").write("LOADING\n")
    t0 = time.time()
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        tok = AutoTokenizer.from_pretrained(MID)
        model = AutoModelForCausalLM.from_pretrained(MID, dtype=torch.bfloat16).to(dev).eval()
        model.requires_grad_(False)
        n_layers = len(model.model.layers)
        nodes = _node_modules(model)
        H, hd = _head_dims(model)
        print("loaded", MID, "| layers", n_layers, "| query-heads", H, "| head_dim", hd, "| dev", dev)

        raw = _build_ioi(tok) + [p for p in _build_pairs(tok) if p["family"] == "antonym"]
        for p in raw:
            p["clean_ids"] = p["clean_ids"].to(dev)
            p["corrupt_ids"] = p["corrupt_ids"].to(dev)
        fam_pairs = {"ioi": [], "antonym": []}
        for p in raw:
            if _solved(model, nodes, p):
                fam_pairs[p["family"]].append(p)
        for fam, ps in fam_pairs.items():
            print("family %-8s: %d solved" % (fam, len(ps)))
            assert len(ps) >= MIN_PER_FAMILY, f"family {fam}: only {len(ps)} pairs (<{MIN_PER_FAMILY})"

        open(status, "w").write("SELFTEST\n")
        p0 = fam_pairs["ioi"][0]
        end = [p0["clean_ids"].shape[1] - 1]
        co, Lc = _cache_oproj(model, p0["clean_ids"], p0["ans_clean"], p0["ans_corrupt"])
        cc, _ = _cache_run(model, nodes, p0["clean_ids"], p0["ans_clean"], p0["ans_corrupt"])
        Lay = 22 if n_layers > 22 else n_layers - 1   
        L_allheads = _patch_layer_allheads(model, p0["corrupt_ids"], co, Lay, end, p0["ans_clean"], p0["ans_corrupt"])
        L_sublayer = _patched_metric_pos(model, nodes, p0["corrupt_ids"], cc, ["a%d" % Lay], end,
                                         p0["ans_clean"], p0["ans_corrupt"])
        d = abs(L_allheads - L_sublayer)
        assert d < 0.05, f"per-head wiring: all-heads@END ({L_allheads:.4f}) != sublayer@END ({L_sublayer:.4f})"
        print("SELFTEST OK | layer %d all-heads@END=%.4f sublayer@END=%.4f |Δ|=%.4f" % (Lay, L_allheads, L_sublayer, d))
        open(status, "w").write("RUNNING\n")

        out = {"config": {"model": MID, "n_layers": n_layers, "n_query_heads": H, "head_dim": hd,
                          "families": {f: len(ps) for f, ps in fam_pairs.items()}, "position": "end"},
               "families": {}}
        for fam, ps in fam_pairs.items():
            acc = np.zeros((n_layers, H), dtype=np.float64)
            for j, p in enumerate(ps):
                end = [p["clean_ids"].shape[1] - 1]
                co, Lc = _cache_oproj(model, p["clean_ids"], p["ans_clean"], p["ans_corrupt"])
                _, Lk = _cache_oproj(model, p["corrupt_ids"], p["ans_clean"], p["ans_corrupt"])
                denom = Lc - Lk
                for layer in range(n_layers):
                    for head in range(H):
                        Lp = _patch_head(model, p["corrupt_ids"], co, layer, head, hd, end,
                                         p["ans_clean"], p["ans_corrupt"])
                        acc[layer, head] += (Lp - Lk) / denom
                print("  [%s] pair %d/%d done" % (fam, j + 1, len(ps)))
            acc /= len(ps)
            
            flat = [((l, h), float(acc[l, h])) for l in range(n_layers) for h in range(H)]
            flat.sort(key=lambda z: z[1], reverse=True)
            out["families"][fam] = {
                "head_effect_end": acc.tolist(),                       
                "top_heads": [{"layer": l, "head": h, "effect": v} for (l, h), v in flat[:15]],
            }
            print("  >> %s top heads:" % fam, [("L%dH%d" % (l, h), round(v, 3)) for (l, h), v in flat[:8]])

        out["wall_s"] = time.time() - t0
        path = os.path.join(OUT, "eap_ig_heads_2b.json")
        json.dump(out, open(path, "w"), indent=2)
        if os.path.isdir(DRIVE):
            try:
                json.dump(out, open(os.path.join(DRIVE, "eap_ig_heads_2b.json"), "w"), indent=2)
            except Exception as e:
                print("drive save failed:", e)
        open(status, "w").write("DONE %.0fs\n" % (time.time() - t0))
        print("DONE %.1fs" % (time.time() - t0))
    except Exception:
        tb = traceback.format_exc()
        print(tb)
        open(status, "w").write("ERROR\n" + tb)
        raise

if __name__ == "__main__":
    main()
