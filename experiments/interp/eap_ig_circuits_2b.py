
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
    MID, OUT, DRIVE, M_IG, _node_modules, _metric, _cache_run, _patched_metric,
    _eap_ig_scores, _build_pairs,
)

MIN_PER_FAMILY = 5

IOI_NAMES = [
    ("Mary", "John"), ("Tom", "Anna"), ("Paul", "Lucy"), ("Mark", "Sara"),
    ("Jack", "Emma"), ("Mike", "Kate"), ("Henry", "Alice"), ("David", "Laura"),
    ("Peter", "Julia"), ("Sam", "Nina"),
]
IOI_TEMPLATE = "When {x} and {y} went to the store, {y} gave a drink to"

def _build_ioi(tok):
    def ids(s):
        return tok(s, return_tensors="pt").input_ids
    def ans_id(w):
        t = tok(" " + w, add_special_tokens=False).input_ids
        return t[0] if len(t) == 1 else None

    pairs = []
    for A, B in IOI_NAMES:
        ca, cb = ans_id(A), ans_id(B)
        if ca is None or cb is None:
            continue
        ci = ids(IOI_TEMPLATE.format(x=A, y=B))
        ki = ids(IOI_TEMPLATE.format(x=B, y=A))
        if ci.shape[1] != ki.shape[1]:
            continue
        pairs.append({"family": "ioi",
                      "clean_text": IOI_TEMPLATE.format(x=A, y=B),
                      "corrupt_text": IOI_TEMPLATE.format(x=B, y=A),
                      "clean_ids": ci, "corrupt_ids": ki, "ans_clean": ca, "ans_corrupt": cb})
    return pairs

def _effects_both(model, nodes, pair):
    cc, Lc = _cache_run(model, nodes, pair["clean_ids"], pair["ans_clean"], pair["ans_corrupt"])
    kc, Lk = _cache_run(model, nodes, pair["corrupt_ids"], pair["ans_clean"], pair["ans_corrupt"])
    denom = Lc - Lk
    den, noi = {}, {}
    for k, _ in nodes:
        Ld = _patched_metric(model, nodes, pair["corrupt_ids"], cc, [k], pair["ans_clean"], pair["ans_corrupt"])
        Ln = _patched_metric(model, nodes, pair["clean_ids"], kc, [k], pair["ans_clean"], pair["ans_corrupt"])
        den[k] = (Ld - Lk) / denom            
        noi[k] = (Lc - Ln) / denom            
    return den, noi, Lc, Lk, cc, kc

def _solved(model, nodes, p):
    _, Lc = _cache_run(model, nodes, p["clean_ids"], p["ans_clean"], p["ans_corrupt"])
    _, Lk = _cache_run(model, nodes, p["corrupt_ids"], p["ans_clean"], p["ans_corrupt"])
    p["L_clean"], p["L_corrupt"] = Lc, Lk
    return Lc > 0 and Lk < 0

def _agg_family(rows_den, rows_noi, node_keys, n_layers):
    den = {k: float(np.mean([r[k] for r in rows_den])) for k in node_keys}
    noi = {k: float(np.mean([r[k] for r in rows_noi])) for k in node_keys}
    
    robust = {k: float(min(den[k], noi[k])) for k in node_keys}
    comp = [k for k in node_keys if k != "emb"]
    order = sorted(comp, key=lambda k: robust[k], reverse=True)
    layer_imp = [abs(den["a%d" % i]) + abs(den["m%d" % i]) for i in range(n_layers)]
    tot = sum(layer_imp) or 1.0
    cdf = (np.cumsum(layer_imp) / tot).tolist()
    return {
        "denoise_mean": den, "noise_mean": noi, "robust_min": robust,
        "emb_sufficiency_denoise": den["emb"], "m0_sufficiency_denoise": den["m0"],
        "emb_necessity_noise": noi["emb"], "m0_necessity_noise": noi["m0"],
        "median_importance_depth": int(np.searchsorted(cdf, 0.5)),
        "layer_importance_denoise": layer_imp, "cdf": cdf,
        "top_robust_nodes": [{"node": k, "robust": robust[k], "denoise": den[k], "noise": noi[k]}
                             for k in order[:12]],
    }

def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    status = os.path.join(OUT, "EAPIG2_STATUS.txt")
    open(status, "w").write("LOADING\n")
    t0 = time.time()
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from scipy.stats import spearmanr
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
            for p in ps:
                print("  %-52s L=%+.2f / %-52s L=%+.2f"
                      % (p["clean_text"], p["L_clean"], p["corrupt_text"], p["L_corrupt"]))
            assert len(ps) >= MIN_PER_FAMILY, f"family {fam}: only {len(ps)} pairs (<{MIN_PER_FAMILY}); aborting"

        open(status, "w").write("SELFTEST\n")
        p0 = fam_pairs["ioi"][0]
        cc, Lc = _cache_run(model, nodes, p0["clean_ids"], p0["ans_clean"], p0["ans_corrupt"])
        L_full = _patched_metric(model, nodes, p0["corrupt_ids"], cc, node_keys, p0["ans_clean"], p0["ans_corrupt"])
        assert abs(L_full - Lc) < 2.0, f"full-patch != clean (|Δ|={abs(L_full - Lc):.3f})"
        print("SELFTEST full-patch OK | L_clean=%.3f L_full=%.3f |Δ|=%.4f" % (Lc, L_full, abs(L_full - Lc)))
        open(status, "w").write("RUNNING\n")

        out = {"config": {"model": MID, "n_layers": n_layers, "n_nodes": len(nodes), "m_ig": M_IG,
                          "families": {f: len(ps) for f, ps in fam_pairs.items()}},
               "families": {}}
        comp = [k for k in node_keys if k != "emb"]
        for fam, ps in fam_pairs.items():
            rows_den, rows_noi, eap_rows = [], [], []
            for j, p in enumerate(ps):
                den, noi, Lc, Lk, cc, kc = _effects_both(model, nodes, p)
                with torch.enable_grad():
                    sc = _eap_ig_scores(model, nodes, p, cc, kc)   
                denom = Lc - Lk
                rows_den.append(den)
                rows_noi.append(noi)
                eap_rows.append({k: sc[k] / denom for k in node_keys})
                print("  [%s] pair %d/%d done" % (fam, j + 1, len(ps)))
            summ = _agg_family(rows_den, rows_noi, node_keys, n_layers)
            
            eap_mean = {k: float(np.mean([r[k] for r in eap_rows])) for k in node_keys}
            ex = np.array([summ["denoise_mean"][k] for k in comp])
            ei = np.array([eap_mean[k] for k in comp])
            summ["eapig_spearman_vs_exact"] = float(spearmanr(ex, ei).correlation)
            summ["eapig_score_mean"] = eap_mean
            out["families"][fam] = summ
            print("  >> %s: emb_suff=%.2f m0_suff=%.2f med_depth=%d EAP-IG ρ=%.2f"
                  % (fam, summ["emb_sufficiency_denoise"], summ["m0_sufficiency_denoise"],
                     summ["median_importance_depth"], summ["eapig_spearman_vs_exact"]))

        io, an = out["families"]["ioi"], out["families"]["antonym"]
        out["contrast"] = {
            "emb_sufficiency": {"antonym": an["emb_sufficiency_denoise"], "ioi": io["emb_sufficiency_denoise"]},
            "m0_sufficiency": {"antonym": an["m0_sufficiency_denoise"], "ioi": io["m0_sufficiency_denoise"]},
            "median_importance_depth": {"antonym": an["median_importance_depth"], "ioi": io["median_importance_depth"]},
            "eapig_spearman": {"antonym": an["eapig_spearman_vs_exact"], "ioi": io["eapig_spearman_vs_exact"]},
        }
        out["wall_s"] = time.time() - t0

        path = os.path.join(OUT, "eap_ig_circuits_2b.json")
        json.dump(out, open(path, "w"), indent=2)
        if os.path.isdir(DRIVE):
            try:
                json.dump(out, open(os.path.join(DRIVE, "eap_ig_circuits_2b.json"), "w"), indent=2)
            except Exception as e:
                print("drive save failed:", e)
        open(status, "w").write("DONE %.0fs\n" % (time.time() - t0))
        print("DONE %.1fs" % (time.time() - t0))
        print("CONTRAST (minimal-pair antonym vs distributed IOI):")
        print("  emb sufficiency:", out["contrast"]["emb_sufficiency"])
        print("  m0  sufficiency:", out["contrast"]["m0_sufficiency"])
        print("  median importance depth:", out["contrast"]["median_importance_depth"])
        print("  EAP-IG Spearman:", out["contrast"]["eapig_spearman"])
        print("  IOI top robust nodes:", [(d["node"], round(d["robust"], 3)) for d in io["top_robust_nodes"][:8]])
    except Exception:
        tb = traceback.format_exc()
        print(tb)
        open(status, "w").write("ERROR\n" + tb)
        raise

if __name__ == "__main__":
    main()
