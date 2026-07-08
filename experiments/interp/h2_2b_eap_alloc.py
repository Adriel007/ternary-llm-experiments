
from __future__ import annotations

import json
import math
import os
import sys
import time
import traceback

ROOT = "/content/PhD-propose"
sys.path[:0] = [ROOT, os.path.join(ROOT, "packages/bitnet_core/src")]

import numpy as np  

from experiments.interp.eap_ig_2b import MID, OUT, DRIVE  
from experiments.interp.h2_2b_correction import (  
    RANK, ALPHA, install_unified_forward, attach_adapters,
)
from experiments.interp.h2_2b_greedy import (  
    REPORT_BUDGETS, SEEDS_FINAL, build_corpora, measure_path_point,
)

SEED = 0
RUN2 = os.path.join(ROOT, "reports/data/h2_2b_correction.json")
GREEDY = os.path.join(ROOT, "reports/data/h2_2b_greedy.json")
PREDICTOR = os.path.join(ROOT, "reports/data/h2_2b_predictor.json")
EAPNODES = os.path.join(ROOT, "reports/data/eap_ig_2b.json")   
EDIR = os.path.join(DRIVE, "h2_2b_eap_alloc_checkpoints")
ECKPT = os.path.join(EDIR, "h2_2b_eap_alloc_state.json")
ELOCAL = os.path.join(OUT, "h2_2b_eap_alloc.json")

CFG = {
    "model": MID, "rank": RANK, "alpha": ALPHA, "report_budgets": REPORT_BUDGETS,
    "seeds_final": SEEDS_FINAL, "seed": SEED,
    "signal": "per-layer |EAP-IG| and |exact-patching| task-causal importance (reused from eap_ig_2b.json), |attn|+|mlp|",
}

def _atomic_dump(obj, path):
    tmp = path + ".tmp"
    json.dump(obj, open(tmp, "w"), indent=2)
    os.replace(tmp, path)

def _load_state():
    for p in (ECKPT, ELOCAL):
        if os.path.exists(p):
            try:
                return json.load(open(p))
            except Exception:
                pass
    return {}

def _save_state(state):
    os.makedirs(EDIR, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)
    _atomic_dump(state, ECKPT)
    _atomic_dump(state, ELOCAL)

def _rankdata(x):
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    ranks[order] = np.arange(1, len(x) + 1, dtype=float)
    _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.zeros(len(cnt)); np.add.at(sums, inv, ranks)
    return (sums / cnt)[inv]

def _spearman(a, b):
    ra, rb = _rankdata(a), _rankdata(b)
    ra = ra - ra.mean(); rb = rb - rb.mean()
    denom = math.sqrt(float((ra ** 2).sum()) * float((rb ** 2).sum()))
    return float((ra * rb).sum() / denom) if denom > 0 else 0.0

def _agg_layers(node_scores, n_layers):
    return np.array([abs(node_scores.get("a%d" % i, 0.0)) + abs(node_scores.get("m%d" % i, 0.0))
                     for i in range(n_layers)])

def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    status = os.path.join(OUT, "H2_2B_EAP_ALLOC_STATUS.txt")
    open(status, "w").write("LOADING\n")
    t0 = time.time()
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        tok = AutoTokenizer.from_pretrained(MID)
        model = AutoModelForCausalLM.from_pretrained(MID, dtype=torch.bfloat16).to(dev).eval()
        model.requires_grad_(False)
        n_layers = len(model.model.layers)
        install_unified_forward()
        plp, extra = attach_adapters(model, RANK, ALPHA, dev, SEED)
        print("loaded", MID, "| layers", n_layers, "| extra/layer=%.4f" % extra)

        train_b, val_ids, test_ids, _score = build_corpora(tok, dev)
        print("corpora: train %d-batches | val %d | test %d" %
              (len(train_b), val_ids.shape[0], test_ids.shape[0]))

        run2 = json.load(open(RUN2)); greedy = json.load(open(GREEDY))
        eapn = json.load(open(EAPNODES))
        c_val = np.array([float(run2["c_l"][str(i)]) for i in range(n_layers)])
        s_trace = np.array([run2["s_l"][str(i)]["trace"] for i in range(n_layers)])
        recov = np.array([float(run2["recovered"][str(i)]) for i in range(n_layers)])
        greedy_order = [d["added"] for d in greedy["greedy_path"]]
        cmp_existing = greedy["comparison"]
        g_ell = None
        if os.path.exists(PREDICTOR):
            g_ell = np.array([json.load(open(PREDICTOR))["g_ell"][str(i)] for i in range(n_layers)])

        c_eap = _agg_layers(eapn["eapig_score_mean"], n_layers)        
        c_exact = _agg_layers(eapn["exact_effect_mean"], n_layers)     
        eap_order = [int(x) for x in np.argsort(-c_eap)]
        exact_order = [int(x) for x in np.argsort(-c_exact)]
        print("EAP-IG ranking (top10): %s" % eap_order[:10])
        print("exact  ranking (top10): %s" % exact_order[:10])
        print("greedy order          : %s" % greedy_order)

        state = _load_state()
        if "config" in state and {k: state["config"].get(k) for k in CFG} != CFG:
            raise RuntimeError("checkpoint config mismatch")
        state["config"] = dict(CFG); state["extra_mem_frac_per_layer"] = extra
        state["eap_order"] = eap_order; state["exact_order"] = exact_order
        state["c_eap"] = {str(i): float(c_eap[i]) for i in range(n_layers)}
        state["c_exact"] = {str(i): float(c_exact[i]) for i in range(n_layers)}

        corr = {
            "spearman_eap_vs_exact": _spearman(c_eap, c_exact),
            "spearman_eap_vs_c_causal": _spearman(c_eap, c_val),
            "spearman_eap_vs_s_hawq": _spearman(c_eap, s_trace),
            "spearman_eap_vs_recovered": _spearman(c_eap, recov),
        }
        if g_ell is not None:
            corr["spearman_eap_vs_g_ell"] = _spearman(c_eap, g_ell)
        e_rank = {li: r for r, li in enumerate(eap_order)}
        x_rank = {li: r for r, li in enumerate(exact_order)}
        add_pos = list(range(len(greedy_order)))
        corr["spearman_greedyorder_vs_eap"] = _spearman(add_pos, [e_rank[l] for l in greedy_order])
        corr["spearman_greedyorder_vs_exact"] = _spearman(add_pos, [x_rank[l] for l in greedy_order])
        corr["topB_overlap_with_greedy"] = {}
        for B in REPORT_BUDGETS:
            grdy = set(greedy_order[:B])
            corr["topB_overlap_with_greedy"][str(B)] = {
                "eap_ig": len(set(eap_order[:B]) & grdy),
                "exact": len(set(exact_order[:B]) & grdy), "of": B}
        state["correlations"] = corr
        _save_state(state)
        print("spearman eap~exact=%.3f eap~c=%.3f eap~HAWQ=%.3f eap~recov=%.3f | greedy-order~eap=%.3f ~exact=%.3f"
              % (corr["spearman_eap_vs_exact"], corr["spearman_eap_vs_c_causal"],
                 corr["spearman_eap_vs_s_hawq"], corr["spearman_eap_vs_recovered"],
                 corr["spearman_greedyorder_vs_eap"], corr["spearman_greedyorder_vs_exact"]))

        open(status, "w").write("FRONT\n")
        state.setdefault("eap_front", {}); state.setdefault("exact_front", {})
        for name, order, fr in (("EAP-IG", eap_order, state["eap_front"]),
                                ("exact", exact_order, state["exact_front"])):
            print("=== %s top-B joint front (%d seeds, identical 512-test) ===" % (name, len(SEEDS_FINAL)))
            for B in REPORT_BUDGETS:
                if str(B) in fr:
                    continue
                t = time.time()
                layers = sorted(order[:B])
                pt = measure_path_point(model, plp, layers, train_b, val_ids, test_ids, SEEDS_FINAL)
                fr[str(B)] = pt; _save_state(state)
                print("  [%s] B=%2d %s -> ppl=%.3f±%.3f (%.0fs)" %
                      (name, B, layers, pt["ppl_mean"], pt["ppl_sd"], time.time() - t))

        open(status, "w").write("VERDICT\n")
        table = {}
        eb_c = eb_rm = eb_rp = eb_g = xb_c = n = 0
        gap_e = gap_x = gap_c = 0.0
        for B in REPORT_BUDGETS:
            ex = cmp_existing[str(B)]
            eP = state["eap_front"][str(B)]["ppl_mean"]
            xP = state["exact_front"][str(B)]["ppl_mean"]
            row = {"eap_ig": eP, "exact": xP, "greedy": ex["greedy"], "c_causal": ex["c_causal"],
                   "s_hawq": ex["s_hawq"], "oracle": ex["oracle"],
                   "random_median": ex["random_median"], "random_p05": ex["random_p05"]}
            if g_ell is not None and os.path.exists(PREDICTOR):
                row["g_ell"] = json.load(open(PREDICTOR))["g_front"][str(B)]["ppl_mean"]
            table[str(B)] = row
            n += 1
            eb_c += eP < ex["c_causal"]; eb_rm += eP < ex["random_median"]
            eb_rp += eP < ex["random_p05"]; eb_g += eP < ex["greedy"]
            xb_c += xP < ex["c_causal"]
            gap_e += eP - ex["greedy"]; gap_x += xP - ex["greedy"]; gap_c += ex["c_causal"] - ex["greedy"]
        verdict = {
            "eap_beats_c_causal": [eb_c, n], "eap_beats_random_median": [eb_rm, n],
            "eap_beats_random_p05": [eb_rp, n], "eap_beats_greedy_ceiling": [eb_g, n],
            "exact_beats_c_causal": [xb_c, n],
            "mean_gap_eap_to_greedy": gap_e / n, "mean_gap_exact_to_greedy": gap_x / n,
            "mean_gap_c_to_greedy": gap_c / n,
        }
        state["table"] = table; state["verdict"] = verdict
        state["wall_s"] = time.time() - t0
        _save_state(state)

        print("\n=== EAP-IG / exact ALLOCATION vs greedy ceiling (test ppl; base %.3f) ===" % greedy["base"]["ppl"])
        print("  B    eap_ig  exact   greedy   c_causal  s_hawq   random med[p05]")
        for B in REPORT_BUDGETS:
            r = table[str(B)]
            print("  %2d   %5.2f   %5.2f   %5.2f    %5.2f     %5.2f    %5.2f[%.2f]" %
                  (B, r["eap_ig"], r["exact"], r["greedy"], r["c_causal"], r["s_hawq"],
                   r["random_median"], r["random_p05"]))
        print("EAP-IG beats c_causal %d/%d | random-med %d/%d | random-p05 %d/%d | greedy %d/%d  ||  exact beats c_causal %d/%d"
              % (*verdict["eap_beats_c_causal"], *verdict["eap_beats_random_median"],
                 *verdict["eap_beats_random_p05"], *verdict["eap_beats_greedy_ceiling"],
                 *verdict["exact_beats_c_causal"]))
        print("mean gap to greedy ceiling: EAP-IG=%.3f  exact=%.3f  c_causal=%.3f ppl"
              % (verdict["mean_gap_eap_to_greedy"], verdict["mean_gap_exact_to_greedy"],
                 verdict["mean_gap_c_to_greedy"]))
        open(status, "w").write("DONE %.0fs\n" % (time.time() - t0))
        print("DONE %.1fs" % (time.time() - t0))
    except Exception:
        tb = traceback.format_exc()
        print(tb); open(status, "w").write("ERROR\n" + tb)
        raise

if __name__ == "__main__":
    main()
