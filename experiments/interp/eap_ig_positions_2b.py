
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
    MID, OUT, DRIVE, _node_modules, _out0, _metric, _cache_run, _build_pairs,
)
from experiments.interp.eap_ig_circuits_2b import _build_ioi  

MIN_PER_FAMILY = 5
POS_SETS = ("content", "end", "all")

def _positions(pair, dev):
    ci, ki = pair["clean_ids"][0], pair["corrupt_ids"][0]
    diff = (ci != ki).nonzero(as_tuple=True)[0].tolist()
    end = [ci.shape[0] - 1]
    return {"content": diff, "end": end, "all": list(range(ci.shape[0]))}

@torch.no_grad()
def _patched_metric_pos(model, nodes, base_ids, patch_acts, which, positions, ans_c, ans_k):
    sel = set(which)
    pos = torch.as_tensor(positions, device=base_ids.device, dtype=torch.long)

    def mk(k):
        def hook(m, i, o):
            if k not in sel:
                return None
            base = _out0(o).clone()
            base[:, pos, :] = patch_acts[k][:, pos, :]
            if isinstance(o, tuple):
                return (base,) + tuple(o[1:])
            return base
        return hook

    handles = [mod.register_forward_hook(mk(k)) for k, mod in nodes]
    out = model(input_ids=base_ids)
    for h in handles:
        h.remove()
    return _metric(out.logits[0, -1], ans_c, ans_k).item()

def _solved(model, nodes, p):
    _, Lc = _cache_run(model, nodes, p["clean_ids"], p["ans_clean"], p["ans_corrupt"])
    _, Lk = _cache_run(model, nodes, p["corrupt_ids"], p["ans_clean"], p["ans_corrupt"])
    p["L_clean"], p["L_corrupt"] = Lc, Lk
    return Lc > 0 and Lk < 0

def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    status = os.path.join(OUT, "EAPIG3_STATUS.txt")
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
        node_keys = [k for k, _ in nodes]
        print("loaded", MID, "| layers", n_layers, "| nodes", len(nodes), "| dev", dev)

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
        cc, Lc = _cache_run(model, nodes, p0["clean_ids"], p0["ans_clean"], p0["ans_corrupt"])
        allpos = list(range(p0["clean_ids"].shape[1]))
        L_full = _patched_metric_pos(model, nodes, p0["corrupt_ids"], cc, node_keys, allpos,
                                     p0["ans_clean"], p0["ans_corrupt"])
        assert abs(L_full - Lc) < 2.0, f"all/all patch != clean (|Δ|={abs(L_full - Lc):.3f})"
        print("SELFTEST OK | L_clean=%.3f L_full=%.3f |Δ|=%.4f" % (Lc, L_full, abs(L_full - Lc)))
        open(status, "w").write("RUNNING\n")

        out = {"config": {"model": MID, "n_layers": n_layers, "n_nodes": len(nodes),
                          "families": {f: len(ps) for f, ps in fam_pairs.items()},
                          "position_sets": list(POS_SETS)},
               "families": {}}
        for fam, ps in fam_pairs.items():
            
            acc = {ps_: {k: [] for k in node_keys} for ps_ in POS_SETS}
            for j, p in enumerate(ps):
                cc, Lc = _cache_run(model, nodes, p["clean_ids"], p["ans_clean"], p["ans_corrupt"])
                _, Lk = _cache_run(model, nodes, p["corrupt_ids"], p["ans_clean"], p["ans_corrupt"])
                denom = Lc - Lk
                posmap = _positions(p, dev)
                for ps_ in POS_SETS:
                    positions = posmap[ps_]
                    for k, _m in nodes:
                        Lp = _patched_metric_pos(model, nodes, p["corrupt_ids"], cc, [k], positions,
                                                 p["ans_clean"], p["ans_corrupt"])
                        acc[ps_][k].append((Lp - Lk) / denom)
                print("  [%s] pair %d/%d done" % (fam, j + 1, len(ps)))
            eff = {ps_: {k: float(np.mean(acc[ps_][k])) for k in node_keys} for ps_ in POS_SETS}
            
            comp = [k for k in node_keys if k != "emb"]
            late_order = sorted(comp, key=lambda k: eff["end"][k], reverse=True)[:8]
            out["families"][fam] = {
                "effect": eff,
                "early_detok": {n: {ps_: eff[ps_][n] for ps_ in POS_SETS} for n in ("emb", "m0")},
                "top_end_nodes": [{"node": k, "end": eff["end"][k], "content": eff["content"][k],
                                   "all": eff["all"][k]} for k in late_order],
            }
            e = out["families"][fam]["early_detok"]
            print("  >> %s m0: content=%.2f end=%.2f all=%.2f | emb: content=%.2f end=%.2f"
                  % (fam, e["m0"]["content"], e["m0"]["end"], e["m0"]["all"],
                     e["emb"]["content"], e["emb"]["end"]))
            print("     top end-effect nodes:", [(d["node"], round(d["end"], 2)) for d in out["families"][fam]["top_end_nodes"]])

        out["wall_s"] = time.time() - t0
        path = os.path.join(OUT, "eap_ig_positions_2b.json")
        json.dump(out, open(path, "w"), indent=2)
        if os.path.isdir(DRIVE):
            try:
                json.dump(out, open(os.path.join(DRIVE, "eap_ig_positions_2b.json"), "w"), indent=2)
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
