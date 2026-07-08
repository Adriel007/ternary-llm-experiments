
from __future__ import annotations

import json
import os
import sys
import time
import traceback

ROOT = os.environ.get("PHD_ROOT", "/content/PhD-propose")
sys.path[:0] = [ROOT, os.path.join(ROOT, "packages/bitnet_core/src")]

import numpy as np  
import torch  

from experiments.interp.eap_ig_2b import (  
    MID, OUT, DRIVE, _node_modules, _cache_run,
)
from experiments.interp.eap_ig_heads_2b import (  
    _head_dims, _cache_oproj, _patch_head, _patch_layer_allheads, _solved,
)
from experiments.interp.eap_ig_positions_2b import _patched_metric_pos  

IOI_TEMPLATES = {
    "store": "When {x} and {y} went to the store, {y} gave a drink to",
    "party": "After {x} and {y} left the party, {y} handed the keys to",
    "work":  "While {x} and {y} were at work, {y} passed a note to",
}

NAME_PAIRS = [
    ("Mary", "John"), ("Tom", "Anna"), ("Paul", "Lucy"), ("Mark", "Sara"),
    ("Jack", "Emma"), ("Mike", "Kate"), ("Henry", "Alice"), ("David", "Laura"),
    ("Peter", "Julia"), ("Sam", "Nina"), ("James", "Susan"), ("Robert", "Karen"),
    ("Michael", "Linda"), ("William", "Nancy"), ("Richard", "Helen"), ("Thomas", "Carol"),
    ("Charles", "Donna"), ("Daniel", "Ruth"), ("George", "Sharon"), ("Frank", "Sandra"),
    ("Steve", "Diana"), ("Brian", "Grace"), ("Kevin", "Rose"), ("Eric", "Jane"),
    ("Adam", "Amy"), ("Scott", "Ella"), ("Edward", "Lily"), ("Joseph", "Hannah"),
]

MIN_PAIRS_TOTAL = 50
MIN_PER_TEMPLATE = 12
NAMED_HEADS = [(22, 17), (22, 19), (27, 13)]   
TOPK = 10

def _build_ioi_multi(tok):
    def ids(s):
        return tok(s, return_tensors="pt").input_ids

    def ans_id(w):
        t = tok(" " + w, add_special_tokens=False).input_ids
        return t[0] if len(t) == 1 else None

    pairs = []
    for tname, tmpl in IOI_TEMPLATES.items():
        for A, B in NAME_PAIRS:
            ca, cb = ans_id(A), ans_id(B)
            if ca is None or cb is None:
                continue
            ci = ids(tmpl.format(x=A, y=B))
            ki = ids(tmpl.format(x=B, y=A))
            if ci.shape[1] != ki.shape[1]:
                continue
            pairs.append({"template": tname,
                          "clean_text": tmpl.format(x=A, y=B),
                          "corrupt_text": tmpl.format(x=B, y=A),
                          "clean_ids": ci, "corrupt_ids": ki, "ans_clean": ca, "ans_corrupt": cb})
    return pairs

def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    status = os.path.join(OUT, "EAPIG_B6_STATUS.txt")
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

        raw = _build_ioi_multi(tok)
        for p in raw:
            p["clean_ids"] = p["clean_ids"].to(dev)
            p["corrupt_ids"] = p["corrupt_ids"].to(dev)
        by_t = {t: [] for t in IOI_TEMPLATES}
        for p in raw:
            if _solved(model, nodes, p):
                by_t[p["template"]].append(p)
        n_total = sum(len(v) for v in by_t.values())
        for t, ps in by_t.items():
            print("template %-7s: %d solved" % (t, len(ps)))
            assert len(ps) >= MIN_PER_TEMPLATE, f"template {t}: only {len(ps)} (<{MIN_PER_TEMPLATE})"
        assert n_total >= MIN_PAIRS_TOTAL, f"only {n_total} pairs total (<{MIN_PAIRS_TOTAL})"
        print("TOTAL IOI pairs: %d across %d templates" % (n_total, len(IOI_TEMPLATES)))

        open(status, "w").write("SELFTEST\n")
        p0 = by_t[next(iter(by_t))][0]
        end = [p0["clean_ids"].shape[1] - 1]
        co, _ = _cache_oproj(model, p0["clean_ids"], p0["ans_clean"], p0["ans_corrupt"])
        cc, _ = _cache_run(model, nodes, p0["clean_ids"], p0["ans_clean"], p0["ans_corrupt"])
        Lay = 22 if n_layers > 22 else n_layers - 1
        L_all = _patch_layer_allheads(model, p0["corrupt_ids"], co, Lay, end, p0["ans_clean"], p0["ans_corrupt"])
        L_sub = _patched_metric_pos(model, nodes, p0["corrupt_ids"], cc, ["a%d" % Lay], end,
                                    p0["ans_clean"], p0["ans_corrupt"])
        d = abs(L_all - L_sub)
        assert d < 0.05, f"wiring: all-heads@END ({L_all:.4f}) != sublayer@END ({L_sub:.4f})"
        print("SELFTEST OK | layer %d |Δ|=%.4f" % (Lay, d))
        open(status, "w").write("RUNNING\n")

        acc_t = {t: np.zeros((n_layers, H), dtype=np.float64) for t in IOI_TEMPLATES}
        done = 0
        for t, ps in by_t.items():
            for p in ps:
                end = [p["clean_ids"].shape[1] - 1]
                co, Lc = _cache_oproj(model, p["clean_ids"], p["ans_clean"], p["ans_corrupt"])
                _, Lk = _cache_oproj(model, p["corrupt_ids"], p["ans_clean"], p["ans_corrupt"])
                denom = Lc - Lk
                for layer in range(n_layers):
                    for head in range(H):
                        Lp = _patch_head(model, p["corrupt_ids"], co, layer, head, hd, end,
                                         p["ans_clean"], p["ans_corrupt"])
                        acc_t[t][layer, head] += (Lp - Lk) / denom
                done += 1
                if done % 5 == 0 or done == n_total:
                    print("  patched %d/%d pairs (%.0fs)" % (done, n_total, time.time() - t0), flush=True)
            acc_t[t] /= len(ps)

        acc_pool = sum(acc_t[t] * len(by_t[t]) for t in IOI_TEMPLATES) / n_total

        def topk(acc):
            flat = [((l, h), float(acc[l, h])) for l in range(n_layers) for h in range(H)]
            flat.sort(key=lambda z: z[1], reverse=True)
            return flat

        pool_flat = topk(acc_pool)
        rank_of = {lh: i for i, (lh, _) in enumerate(pool_flat)}
        rank_of_t = {t: {lh: i for i, (lh, _) in enumerate(topk(acc_t[t]))} for t in IOI_TEMPLATES}
        per_t_topset = {t: set(lh for lh, _ in topk(acc_t[t])[:TOPK]) for t in IOI_TEMPLATES}

        named = {}
        for (l, h) in NAMED_HEADS:
            key = "L%dH%d" % (l, h)
            named[key] = {
                "pooled_effect": float(acc_pool[l, h]),
                "pooled_rank": rank_of[(l, h)],
                "per_template_effect": {t: float(acc_t[t][l, h]) for t in IOI_TEMPLATES},
                "per_template_rank": {t: rank_of_t[t][(l, h)] for t in IOI_TEMPLATES},
                "in_topk_each_template": {t: (l, h) in per_t_topset[t] for t in IOI_TEMPLATES},
                "in_topk_all_templates": all((l, h) in per_t_topset[t] for t in IOI_TEMPLATES),
            }

        floor = float(np.median(np.abs(acc_pool)))

        out = {
            "config": {"model": MID, "n_layers": n_layers, "n_query_heads": H, "head_dim": hd,
                       "templates": IOI_TEMPLATES, "n_pairs_per_template": {t: len(by_t[t]) for t in by_t},
                       "n_pairs_total": n_total, "position": "end", "topk": TOPK},
            "noise_floor_median_abs": floor,
            "named_heads": named,
            "pooled_top_heads": [{"layer": l, "head": h, "effect": v} for (l, h), v in pool_flat[:15]],
            "per_template_top_heads": {
                t: [{"layer": l, "head": h, "effect": v} for (l, h), v in topk(acc_t[t])[:10]]
                for t in IOI_TEMPLATES},
            "head_effect_end_pooled": acc_pool.tolist(),
            "wall_s": time.time() - t0,
        }
        path = os.path.join(OUT, "eap_ig_heads_b6.json")
        json.dump(out, open(path, "w"), indent=2)
        if os.path.isdir(DRIVE):
            try:
                json.dump(out, open(os.path.join(DRIVE, "eap_ig_heads_b6.json"), "w"), indent=2)
            except Exception as e:
                print("drive save failed:", e)
        open(status, "w").write("DONE %.0fs\n" % (time.time() - t0))
        print("\nDONE %.1fs | %d pairs | noise floor (median|eff|)=%.4f" % (time.time() - t0, n_total, floor))
        print("pooled top heads:", [("L%dH%d" % (l, h), round(v, 3)) for (l, h), v in pool_flat[:8]])
        for k, d in named.items():
            print("  %s pooled eff=%.3f rank=%d | in-topK all templates=%s | per-tmpl eff=%s"
                  % (k, d["pooled_effect"], d["pooled_rank"], d["in_topk_all_templates"],
                     {t: round(v, 3) for t, v in d["per_template_effect"].items()}))
    except Exception:
        tb = traceback.format_exc()
        print(tb)
        open(status, "w").write("ERROR\n" + tb)
        raise

if __name__ == "__main__":
    main()
