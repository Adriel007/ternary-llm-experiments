
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
    MID, OUT, DRIVE, _node_modules, _metric, _cache_run, _build_pairs,
)
from experiments.interp.eap_ig_circuits_2b import _build_ioi  

MIN_PER_FAMILY = 5

CAUSAL_HEADS = {"ioi": [(22, 17), (22, 19), (27, 13)], "antonym": [(19, 12), (19, 6), (27, 11)]}

def _sources(pair):
    ci = pair["clean_ids"][0]
    if pair["family"] == "ioi":
        io = (ci == pair["ans_clean"]).nonzero(as_tuple=True)[0].tolist()
        s = (ci == pair["ans_corrupt"]).nonzero(as_tuple=True)[0].tolist()
    else:
        ki = pair["corrupt_ids"][0]
        io = (ci != ki).nonzero(as_tuple=True)[0].tolist()
        s = []
    return io, s

@torch.no_grad()
def _attn_from_end(model, ids):
    out = model(input_ids=ids, output_attentions=True)
    end = ids.shape[1] - 1
    return [a[0, :, end, :].float().cpu() for a in out.attentions]   

def _solved(model, nodes, p):
    _, Lc = _cache_run(model, nodes, p["clean_ids"], p["ans_clean"], p["ans_corrupt"])
    _, Lk = _cache_run(model, nodes, p["corrupt_ids"], p["ans_clean"], p["ans_corrupt"])
    p["L_clean"], p["L_corrupt"] = Lc, Lk
    return Lc > 0 and Lk < 0

def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    status = os.path.join(OUT, "EAPIG5_STATUS.txt")
    open(status, "w").write("LOADING\n")
    t0 = time.time()
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        tok = AutoTokenizer.from_pretrained(MID)
        model = AutoModelForCausalLM.from_pretrained(
            MID, dtype=torch.bfloat16, attn_implementation="eager").to(dev).eval()
        model.requires_grad_(False)
        n_layers = len(model.model.layers)
        nodes = _node_modules(model)
        H = model.config.num_attention_heads
        print("loaded", MID, "| layers", n_layers, "| heads", H, "| eager-attn | dev", dev)

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
        att = _attn_from_end(model, fam_pairs["ioi"][0]["clean_ids"])
        s = float(att[0].sum(dim=-1).mean())
        assert abs(s - 1.0) < 0.02, f"attention rows don't sum to 1 (got {s:.3f}) -- wrong attn extraction"
        print("SELFTEST OK | END-query attention sums to %.4f" % s)
        open(status, "w").write("RUNNING\n")

        out = {"config": {"model": MID, "n_layers": n_layers, "n_heads": H,
                          "families": {f: len(ps) for f, ps in fam_pairs.items()},
                          "causal_heads": {k: v for k, v in CAUSAL_HEADS.items()}},
               "families": {}}
        for fam, ps in fam_pairs.items():
            to_io = np.zeros((n_layers, H)); to_s = np.zeros((n_layers, H))
            for j, p in enumerate(ps):
                io, sp = _sources(p)
                att = _attn_from_end(model, p["clean_ids"])      
                for layer in range(n_layers):
                    a = att[layer].numpy()                        
                    to_io[layer] += a[:, io].mean(axis=1) if io else 0.0
                    to_s[layer] += a[:, sp].mean(axis=1) if sp else 0.0
                print("  [%s] pair %d/%d done" % (fam, j + 1, len(ps)))
            to_io /= len(ps); to_s /= len(ps)
            flat = [((l, h), float(to_io[l, h])) for l in range(n_layers) for h in range(H)]
            flat.sort(key=lambda z: z[1], reverse=True)
            causal = {"L%dH%d" % (l, h): {"attn_to_io": float(to_io[l, h]), "attn_to_s": float(to_s[l, h])}
                      for (l, h) in CAUSAL_HEADS[fam]}
            out["families"][fam] = {
                "attn_to_io": to_io.tolist(), "attn_to_s": to_s.tolist(),
                "top_heads_by_attn_to_io": [{"layer": l, "head": h, "attn_to_io": v,
                                             "attn_to_s": float(to_s[l, h])} for (l, h), v in flat[:12]],
                "causal_name_movers": causal,
            }
            print("  >> %s causal name-movers QK:" % fam,
                  {k: (round(v["attn_to_io"], 2), round(v["attn_to_s"], 2)) for k, v in causal.items()})

        out["wall_s"] = time.time() - t0
        path = os.path.join(OUT, "eap_ig_heads_path_2b.json")
        json.dump(out, open(path, "w"), indent=2)
        if os.path.isdir(DRIVE):
            try:
                json.dump(out, open(os.path.join(DRIVE, "eap_ig_heads_path_2b.json"), "w"), indent=2)
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
