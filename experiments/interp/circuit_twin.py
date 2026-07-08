
from __future__ import annotations

import json
import os
import re
import sys
import time
import traceback

ROOT = os.environ.get("POC_ROOT", "/content/PhD-propose")
sys.path[:0] = [ROOT, os.path.join(ROOT, "packages/bitnet_core/src")]

import numpy as np  
import torch  

from experiments.interp.eap_ig_2b import (  
    _node_modules, _cache_run, _patched_metric, _eap_ig_scores,
)
from experiments.interp.eap_ig_circuits_2b import _effects_both, _solved  
from experiments.interp.eap_ig_heads_2b import (  
    _head_dims, _oproj, _cache_oproj, _patch_head, _patch_layer_allheads,
)
from experiments.interp.eap_ig_positions_2b import _patched_metric_pos  
from experiments.interp.eap_ig_heads_path_2b import _sources, _attn_from_end  
from experiments.interp.eap_ig_heads_ov_2b import (  
    _final_norm, _targets, _cache_clean, _head_write, _coupling_residual, _dla,
)
from experiments.interp.eap_ig_heads_b6 import _build_ioi_multi  

MODEL = os.environ.get("CT_MODEL", "SpectraSuite/TriLM_2.4B_Unpacked")
NPAIRS = int(os.environ.get("CT_NPAIRS", "50"))
TOPK = int(os.environ.get("CT_TOPK", "30"))
HEAD_CAND = int(os.environ.get("CT_HEAD_CAND", "15"))
MIN_PAIRS = min(NPAIRS, 50)          

_slug = re.sub(r"[^0-9A-Za-z]+", "_", MODEL.split("/")[-1]).strip("_")
TAG = os.environ.get("CT_TAG", f"circuit_twin_{_slug}")
OUT_JSON = os.environ.get("CT_OUT_JSON") or os.path.join(ROOT, "artifacts/poc", f"{TAG}.json")

NAME_FLOOR_MULT = 3.0

def _rankdata(a):
    a = np.asarray(a, dtype=float)
    order = np.argsort(a, kind="mergesort")
    sa = a[order]
    ranks = np.empty(len(a), dtype=float)
    i = 0
    n = len(a)
    while i < n:
        j = i
        while j + 1 < n and sa[j + 1] == sa[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks

def _pearson(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])

def _spearman(x, y):
    return _pearson(_rankdata(x), _rankdata(y))

def _jaccard(a, b):
    sa, sb = set(a), set(b)
    u = sa | sb
    return float(len(sa & sb) / len(u)) if u else float("nan")

def _build_pairs(tok, dev):
    raw = _build_ioi_multi(tok)
    for p in raw:
        p["family"] = "ioi"
        p["clean_ids"] = p["clean_ids"].to(dev)
        p["corrupt_ids"] = p["corrupt_ids"].to(dev)
    return raw

def _selftests(model, nodes, node_keys, H, hd, late, p0):
    ac, ak = p0["ans_clean"], p0["ans_corrupt"]
    end = p0["clean_ids"].shape[1] - 1

    cc, Lc = _cache_run(model, nodes, p0["clean_ids"], ac, ak)
    L_full = _patched_metric(model, nodes, p0["corrupt_ids"], cc, node_keys, ac, ak)
    d1 = abs(L_full - Lc)
    assert d1 < 2.0, f"full-patch != clean (|Δ|={d1:.3f}) -- node patch wiring wrong"

    co, _ = _cache_oproj(model, p0["clean_ids"], ac, ak)
    L_all = _patch_layer_allheads(model, p0["corrupt_ids"], co, late, [end], ac, ak)
    L_sub = _patched_metric_pos(model, nodes, p0["corrupt_ids"], cc, [f"a{late}"], [end], ac, ak)
    d2 = abs(L_all - L_sub)
    assert d2 < 0.05, f"per-head wiring: all-heads@END ({L_all:.4f}) != sublayer@END ({L_sub:.4f})"

    

    

    
    co2, _fr, _lg = _cache_clean(model, p0["clean_ids"])
    d3 = _coupling_residual(model, late, hd, H, co2[late][0, end])
    if d3 >= 0.05:
        print("WARNING: o_proj per-head coupling residual %.4f >= 5%% — the HEAD-level map is "
              "coupling-limited on this (activation-quantized) model; node-level map is unaffected."
              % d3, flush=True)

    att = _attn_from_end(model, p0["clean_ids"])
    d4 = float(att[0].sum(dim=-1).mean())
    assert abs(d4 - 1.0) < 0.02, f"attention rows don't sum to 1 (got {d4:.3f})"

    print("SELFTESTS | full-patch |Δ|=%.4f | heads@%d |Δ|=%.4f | coupling=%.4f (head-faithful=%s) | attn-sum=%.4f"
          % (d1, late, d2, d3, d3 < 0.05, d4), flush=True)
    return d3

def run_model():
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    status = os.path.join(os.path.dirname(OUT_JSON), f"{TAG}_STATUS.txt")
    open(status, "w").write("LOADING\n")
    t0 = time.time()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    import transformers
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, attn_implementation="eager").to(dev).eval()
    model.requires_grad_(False)                 

    n_layers = len(model.model.layers)
    nodes = _node_modules(model)
    node_keys = [k for k, _ in nodes]
    comp_keys = [k for k in node_keys if k != "emb"]
    H, hd = _head_dims(model)
    gamma, eps = _final_norm(model)
    WU = model.lm_head.weight                   

    op_in = getattr(_oproj(model, 0), "in_features", H * hd)
    assert op_in == H * hd, f"o_proj in_features {op_in} != H*hd {H*hd}; per-head slice invalid"

    late = max(1, int(round(0.73 * (n_layers - 1))))   
    print("env: torch %s | tf %s | dev %s" % (torch.__version__, transformers.__version__, dev), flush=True)
    print("loaded %s | layers %d | query-heads %d | head_dim %d | late-layer %d"
          % (MODEL, n_layers, H, hd, late), flush=True)

    raw = _build_pairs(tok, dev)
    kept = []
    for p in raw:
        if _solved(model, nodes, p):            
            kept.append(p)
        if len(kept) >= NPAIRS:
            break
    print("IOI pairs: %d built, %d solved & kept (target %d)" % (len(raw), len(kept), NPAIRS), flush=True)
    assert len(kept) >= MIN_PAIRS, f"only {len(kept)} solved IOI pairs (<{MIN_PAIRS}); aborting"

    HEAD_PAIRS = int(os.environ.get("CT_HEAD_PAIRS", str(min(len(kept), 30))))
    HEAD_PAIRS = max(1, min(HEAD_PAIRS, len(kept)))

    open(status, "w").write("SELFTEST\n")
    heads_coupling = _selftests(model, nodes, node_keys, H, hd, late, kept[0])
    heads_faithful = bool(heads_coupling < 0.05)
    open(status, "w").write("RUNNING\n")

    rows_den, rows_noi, rows_eap = [], [], []
    for j, p in enumerate(kept):
        den, noi, Lc, Lk, cc, kc = _effects_both(model, nodes, p)      
        with torch.enable_grad():
            sc = _eap_ig_scores(model, nodes, p, cc, kc)               
        denom = Lc - Lk
        rows_den.append(den)
        rows_noi.append(noi)
        rows_eap.append({k: sc[k] / denom for k in node_keys})
        if (j + 1) % 5 == 0 or j + 1 == len(kept):
            print("  node map %d/%d (%.0fs)" % (j + 1, len(kept), time.time() - t0), flush=True)

    den_mean = {k: float(np.mean([r[k] for r in rows_den])) for k in node_keys}
    noi_mean = {k: float(np.mean([r[k] for r in rows_noi])) for k in node_keys}
    eap_mean = {k: float(np.mean([r[k] for r in rows_eap])) for k in node_keys}
    robust = {k: float(min(den_mean[k], noi_mean[k])) for k in node_keys}
    layer_imp = [abs(den_mean[f"a{i}"]) + abs(den_mean[f"m{i}"]) for i in range(n_layers)]

    order = sorted(comp_keys, key=lambda k: robust[k], reverse=True)   
    top_edges = [{"node": k, "rank": r, "robust": robust[k], "denoise": den_mean[k],
                  "noise": noi_mean[k], "eapig": eap_mean[k]} for r, k in enumerate(order[:TOPK])]

    print("head grid: %d layers × %d heads × %d pairs = %d patched forwards"
          % (n_layers, H, HEAD_PAIRS, n_layers * H * HEAD_PAIRS), flush=True)
    eff = np.zeros((n_layers, H), dtype=np.float64)
    for j, p in enumerate(kept[:HEAD_PAIRS]):
        end = [p["clean_ids"].shape[1] - 1]
        co, Lc = _cache_oproj(model, p["clean_ids"], p["ans_clean"], p["ans_corrupt"])
        _, Lk = _cache_oproj(model, p["corrupt_ids"], p["ans_clean"], p["ans_corrupt"])
        denom = Lc - Lk
        for L in range(n_layers):
            for h in range(H):
                Lp = _patch_head(model, p["corrupt_ids"], co, L, h, hd, end,
                                 p["ans_clean"], p["ans_corrupt"])
                eff[L, h] += (Lp - Lk) / denom
        if (j + 1) % 5 == 0 or j + 1 == HEAD_PAIRS:
            print("  head grid %d/%d (%.0fs)" % (j + 1, HEAD_PAIRS, time.time() - t0), flush=True)
    eff /= HEAD_PAIRS
    noise_floor = float(np.median(np.abs(eff)))
    name_thresh = NAME_FLOOR_MULT * noise_floor

    to_io = np.zeros((n_layers, H), dtype=np.float64)
    to_s = np.zeros((n_layers, H), dtype=np.float64)
    for p in kept:
        io, sp = _sources(p)
        att = _attn_from_end(model, p["clean_ids"])       
        for L in range(n_layers):
            a = att[L].numpy()
            to_io[L] += a[:, io].mean(axis=1) if io else 0.0
            to_s[L] += a[:, sp].mean(axis=1) if sp else 0.0
    to_io /= len(kept)
    to_s /= len(kept)

    flat = [((L, h), float(eff[L, h])) for L in range(n_layers) for h in range(H)]
    cand = sorted(flat, key=lambda z: abs(z[1]), reverse=True)[:HEAD_CAND]
    cand_lh = [lh for lh, _ in cand]
    dla_io = {lh: [] for lh in cand_lh}
    dla_s = {lh: [] for lh in cand_lh}
    for p in kept[:HEAD_PAIRS]:
        co, fr, _lg = _cache_clean(model, p["clean_ids"])
        end = p["clean_ids"].shape[1] - 1
        fr_end = fr[0, end].float()
        tids = _targets(p)                                
        for (L, h) in cand_lh:
            w = _head_write(model, L, h, hd, co[L][0, end])
            dla_io[(L, h)].append(_dla(w, fr_end, gamma, eps, WU, tids["io"]))
            dla_s[(L, h)].append(_dla(w, fr_end, gamma, eps, WU, tids["s"]))

    candidates, name_movers, s_inhibition = [], [], []
    for (L, h), e in cand:
        aio, asu = float(to_io[L, h]), float(to_s[L, h])
        dio = float(np.mean(dla_io[(L, h)]))
        dsu = float(np.mean(dla_s[(L, h)]))
        role = None
        if e > name_thresh and aio >= asu and dio > 0:
            role = "name_mover"
        elif asu > aio and dsu < 0:
            role = "s_inhibition"
        rec = {"layer": L, "head": h, "effect": float(e), "attn_to_io": aio, "attn_to_s": asu,
               "dla_io": dio, "dla_s": dsu, "role": role}
        candidates.append(rec)
        if role == "name_mover":
            name_movers.append(rec)
        elif role == "s_inhibition":
            s_inhibition.append(rec)

    out = {
        "model": MODEL,
        "config": {"model": MODEL, "n_layers": n_layers, "n_query_heads": H, "head_dim": hd,
                   "n_pairs": len(kept), "npairs_requested": NPAIRS, "head_grid_pairs": HEAD_PAIRS,
                   "head_candidates": HEAD_CAND, "topk": TOPK, "n_nodes": len(nodes),
                   "name_floor_mult": NAME_FLOOR_MULT, "noise_floor_median_abs": noise_floor,
                   "name_effect_thresh": name_thresh,
                   "heads_coupling_residual": heads_coupling, "heads_faithful": heads_faithful},
        "granularity": ("component-level: attention & MLP sublayer residual writes + token "
                        "embedding (the §3.16 node set). 'top_edges' is this component ranking, "
                        "NOT upstream->downstream edge patching."),
        "attribution_scores": {"eapig_score_mean": eap_mean, "exact_denoise_mean": den_mean,
                               "exact_noise_mean": noi_mean, "robust_min": robust},
        "top_edges": top_edges,
        "layer_importance": layer_imp,
        "heads": {"head_effect_end": eff.tolist(), "attn_to_io": to_io.tolist(),
                  "attn_to_s": to_s.tolist(), "candidates": candidates,
                  "name_movers": name_movers, "s_inhibition": s_inhibition,
                  "faithful": heads_faithful, "coupling_residual": heads_coupling,
                  "note": ("" if heads_faithful else
                           "per-head o_proj decomposition is coupling-limited (residual %.3f >= 0.05) "
                           "on this activation-quantized model; strong name-movers (effect >> coupling) "
                           "are robust, marginal heads unreliable — node-level map is unaffected."
                           % heads_coupling)},
        "wall_s": time.time() - t0,
    }
    json.dump(out, open(OUT_JSON, "w"), indent=2)
    open(status, "w").write("DONE %.0fs nm=%d si=%d\n" % (time.time() - t0, len(name_movers), len(s_inhibition)))
    print("DONE %.1fs | wrote %s" % (time.time() - t0, OUT_JSON), flush=True)
    print("top components (robust):", [(d["node"], round(d["robust"], 3)) for d in top_edges[:8]], flush=True)
    print("name-movers:", [("L%dH%d" % (d["layer"], d["head"]), round(d["effect"], 3)) for d in name_movers], flush=True)
    print("S-inhibition:", [("L%dH%d" % (d["layer"], d["head"]), round(d["effect"], 3)) for d in s_inhibition], flush=True)
    return out

def compare(path_a, path_b, out_path=None):
    A = json.load(open(path_a))
    B = json.load(open(path_b))
    ca, cb = A["config"], B["config"]
    assert ca["n_layers"] == cb["n_layers"] and ca["n_query_heads"] == cb["n_query_heads"], (
        "twins must share architecture (n_layers=%s/%s, n_query_heads=%s/%s); refusing to compare"
        % (ca["n_layers"], cb["n_layers"], ca["n_query_heads"], cb["n_query_heads"]))
    n_layers, H = ca["n_layers"], ca["n_query_heads"]

    ra, rb = A["attribution_scores"]["robust_min"], B["attribution_scores"]["robust_min"]
    ea, eb = A["attribution_scores"]["eapig_score_mean"], B["attribution_scores"]["eapig_score_mean"]
    common = [k for k in ra if k in rb and k != "emb"]
    rob_a = [ra[k] for k in common]
    rob_b = [rb[k] for k in common]
    eap_a = [ea[k] for k in common]
    eap_b = [eb[k] for k in common]
    topk = min(A["config"]["topk"], B["config"]["topk"])
    top_a = [d["node"] for d in A["top_edges"][:topk]]
    top_b = [d["node"] for d in B["top_edges"][:topk]]

    comp = {
        "n_components": len(common),
        "spearman_robust": _spearman(rob_a, rob_b),
        "pearson_robust": _pearson(rob_a, rob_b),
        "spearman_eapig": _spearman(eap_a, eap_b),
        "jaccard_top%d" % topk: _jaccard(top_a, top_b),
        "top_a": top_a, "top_b": top_b, "intersection": sorted(set(top_a) & set(top_b)),
    }

    ha = np.array(A["heads"]["head_effect_end"], dtype=float).reshape(-1)
    hb = np.array(B["heads"]["head_effect_end"], dtype=float).reshape(-1)
    khead = min(topk, n_layers * H)
    top_ha = [tuple(map(int, x)) for x in
              sorted(((L, h) for L in range(n_layers) for h in range(H)),
                     key=lambda lh: A["heads"]["head_effect_end"][lh[0]][lh[1]], reverse=True)[:khead]]
    top_hb = [tuple(map(int, x)) for x in
              sorted(((L, h) for L in range(n_layers) for h in range(H)),
                     key=lambda lh: B["heads"]["head_effect_end"][lh[0]][lh[1]], reverse=True)[:khead]]

    def _lh_set(recs):
        return {(r["layer"], r["head"]) for r in recs}

    nm_a, nm_b = _lh_set(A["heads"]["name_movers"]), _lh_set(B["heads"]["name_movers"])
    si_a, si_b = _lh_set(A["heads"]["s_inhibition"]), _lh_set(B["heads"]["s_inhibition"])
    heads = {
        "spearman_head_effect": _spearman(ha, hb),
        "pearson_head_effect": _pearson(ha, hb),
        "jaccard_top%d_heads" % khead: _jaccard(top_ha, top_hb),
        "name_movers_a": sorted(nm_a), "name_movers_b": sorted(nm_b),
        "name_movers_overlap": sorted(nm_a & nm_b),
        "name_movers_jaccard": _jaccard(nm_a, nm_b),
        "s_inhibition_a": sorted(si_a), "s_inhibition_b": sorted(si_b),
        "s_inhibition_overlap": sorted(si_a & si_b),
        "s_inhibition_jaccard": _jaccard(si_a, si_b),
        "heads_faithful_a": bool(ca.get("heads_faithful", True)),
        "heads_faithful_b": bool(cb.get("heads_faithful", True)),
        "heads_comparison_reliable": bool(ca.get("heads_faithful", True) and cb.get("heads_faithful", True)),
    }

    result = {"models": {"a": A["model"], "b": B["model"]},
              "sources": {"a": os.path.basename(path_a), "b": os.path.basename(path_b)},
              "n_layers": n_layers, "n_query_heads": H,
              "component_ranking": comp, "heads": heads,
              "interpretation": ("high Spearman/Jaccard on components + matching name-mover / "
                                 "S-inhibition heads => the ternary twin preserves the FP IOI "
                                 "circuit (same-data twins remove the model-identity confound). "
                                 "The node/component comparison is confound-free regardless; the "
                                 "HEAD-level comparison is reliable only when "
                                 "heads_comparison_reliable=True (both twins pass the per-head "
                                 "o_proj coupling test) — else trust the component-level result.")}
    out_path = out_path or os.path.join(os.path.dirname(os.path.abspath(path_a)), "circuit_twin_compare.json")
    json.dump(result, open(out_path, "w"), indent=2)
    print("=== TWIN CIRCUIT COMPARISON ===", flush=True)
    print("  A = %s\n  B = %s" % (A["model"], B["model"]), flush=True)
    print("  components: Spearman(robust)=%.3f  Spearman(EAP-IG)=%.3f  Jaccard(top%d)=%.3f"
          % (comp["spearman_robust"], comp["spearman_eapig"], topk, comp["jaccard_top%d" % topk]), flush=True)
    print("  top-component intersection:", comp["intersection"], flush=True)
    print("  heads: Spearman(END-effect)=%.3f  Jaccard(top%d)=%.3f"
          % (heads["spearman_head_effect"], khead, heads["jaccard_top%d_heads" % khead]), flush=True)
    print("  name-movers  A=%s  B=%s  overlap=%s" % (heads["name_movers_a"], heads["name_movers_b"], heads["name_movers_overlap"]), flush=True)
    print("  S-inhibition A=%s  B=%s  overlap=%s" % (heads["s_inhibition_a"], heads["s_inhibition_b"], heads["s_inhibition_overlap"]), flush=True)
    print("  wrote", out_path, flush=True)
    return result

def main():
    if len(sys.argv) > 1 and sys.argv[1] == "compare":
        if len(sys.argv) < 4:
            raise SystemExit("usage: circuit_twin.py compare A.json B.json [out.json]")
        compare(sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else None)
        return
    try:
        run_model()
    except Exception:
        tb = traceback.format_exc()
        print(tb, flush=True)
        try:
            os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
            open(os.path.join(os.path.dirname(OUT_JSON), f"{TAG}_STATUS.txt"), "w").write("ERROR\n" + tb)
        except Exception:
            pass
        raise

if __name__ == "__main__":
    main()
