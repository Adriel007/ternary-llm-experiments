
from __future__ import annotations

import json
import math
import os
import sys
import time
import traceback

ROOT = os.environ.get("REPO_ROOT", "/content/PhD-propose")
sys.path[:0] = [ROOT, os.path.join(ROOT, "packages/bitnet_core/src")]

import numpy as np  

from experiments.interp.eap_ig_2b import MID  
from experiments.interp.h2_2b_correction import (  
    RANK, ALPHA, install_unified_forward, attach_adapters, _bitlinears,
)
import experiments.interp.h2_2b_greedy as G  
from experiments.interp.h2_2b_eap_alloc import _agg_layers, _spearman  

MODEL = os.environ.get("MODEL", MID)
SMOKE = os.environ.get("SMOKE", "0") == "1"
DATA = os.path.join(ROOT, "reports", "data")
CORR = os.path.join(DATA, "h2_2b_correction.json")     
GREEDYJ = os.path.join(DATA, "h2_2b_greedy.json")      
PREDICTOR = os.path.join(DATA, "h2_2b_predictor.json")  
EAPNODES = os.path.join(DATA, "eap_ig_2b.json")        
OUT_JSON = os.environ.get("OUT_JSON", os.path.join(ROOT, "artifacts", "poc", "h2_2b_redo_hawq_full.json"))
DONE = OUT_JSON + ".DONE"

def _env_int_list(name, default):
    v = os.environ.get(name)
    if not v:
        return list(default)
    return [int(x) for x in v.split(",") if x.strip() != ""]

if SMOKE:
    
    G.MAX_STEPS, G.EVAL_EVERY, G.PATIENCE = 4, 2, 2
    G.N_TRAIN_SEQS, G.N_VAL_SEQS, G.N_TEST_SEQS, G.N_SCORE_SEQS, G.SKIP_SEQS = 16, 8, 8, 4, 4
    BUDGETS = _env_int_list("BUDGETS", [1, 2])
    SEEDS = _env_int_list("SEEDS", [0])
    N_RANDOM = int(os.environ.get("N_RANDOM", "2"))
else:
    BUDGETS = _env_int_list("BUDGETS", G.REPORT_BUDGETS)   
    SEEDS = _env_int_list("SEEDS", G.SEEDS_FINAL)          
    N_RANDOM = int(os.environ.get("N_RANDOM", "5"))

SEED = 0
CFG = {
    "model": MODEL, "rank": RANK, "alpha": ALPHA, "budgets": BUDGETS, "seeds": SEEDS,
    "n_random": N_RANDOM, "smoke": SMOKE, "seed": SEED,
    "max_steps": G.MAX_STEPS, "n_test_seqs": G.N_TEST_SEQS,
    "hawq_v2_full": "tr(H_l) . ||Q(W_l)-W_l||_F^2 ; trace reused from h2_2b_correction.json (smooth-twin "
                    "Hutchinson), perturbation from WeightQuant.apply (deployed ternary quantizer)",
    "machinery": "measure_path_point (joint per-allocation LoRA correction, identical 512-seq test) for "
                 "ALL signals + re-measured greedy ceiling -> controlled single-machinery comparison",
}

def _atomic_dump(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    json.dump(obj, open(tmp, "w"), indent=2)
    os.replace(tmp, path)

def _load_state():
    if os.path.exists(OUT_JSON):
        try:
            return json.load(open(OUT_JSON))
        except Exception:
            pass
    return {}

def _save(state):
    _atomic_dump(state, OUT_JSON)

def perturbation_per_layer(model, n_layers):
    import torch
    from transformers.integrations.bitnet import WeightQuant
    pert = {}
    with torch.no_grad():
        for li, layer in enumerate(model.model.layers):
            s = 0.0
            for bl in _bitlinears(layer):
                w = bl.weight
                q = WeightQuant.apply(w)                       
                s += float((q.float() - w.float()).pow(2).sum().item())
            pert[li] = s
    return pert

def _measure(model, plp, layers, train_b, val_ids, test_ids, seeds, cache, state):
    key = ",".join(map(str, sorted(int(x) for x in layers)))
    if key in cache:
        return cache[key]
    pt = G.measure_path_point(model, plp, sorted(int(x) for x in layers), train_b, val_ids, test_ids, seeds)
    cache[key] = pt
    state["point_cache"] = cache
    _save(state)
    return pt

def _front_for_order(model, plp, order, budgets, train_b, val_ids, test_ids, seeds, cache, state, tag):
    front = {}
    for B in budgets:
        pt = _measure(model, plp, order[:B], train_b, val_ids, test_ids, seeds, cache, state)
        front[str(B)] = pt
        print("  [%-13s] B=%2d %s -> ppl=%.3f +/- %.3f" %
              (tag, B, sorted(order[:B]), pt["ppl_mean"], pt["ppl_sd"]), flush=True)
    return front

def main() -> None:
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    status = OUT_JSON + ".STATUS"
    open(status, "w").write("LOADING\n")
    t0 = time.time()
    try:
        if os.path.exists(DONE):
            print("already DONE (%s exists) -- nothing to do; delete it to force a re-run." % DONE)
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        tok = AutoTokenizer.from_pretrained(MODEL)
        model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to(dev).eval()
        model.requires_grad_(False)
        n_layers = len(model.model.layers)
        install_unified_forward()
        plp, extra = attach_adapters(model, RANK, ALPHA, dev, SEED)
        print("loaded %s | layers %d | dev %s | extra-mem/layer %.4f | SMOKE=%s" %
              (MODEL, n_layers, dev, extra, SMOKE), flush=True)

        corr = json.load(open(CORR))
        greedy = json.load(open(GREEDYJ))
        eapn = json.load(open(EAPNODES))
        pred = json.load(open(PREDICTOR)) if os.path.exists(PREDICTOR) else None
        for nm, d, sub in (("correction s_l", corr, "s_l"), ("correction c_l", corr, "c_l"),
                           ("correction recovered", corr, "recovered")):
            if len(d[sub]) != n_layers:
                raise RuntimeError("%s has %d layers but model has %d -- ranking JSON / model mismatch"
                                   % (nm, len(d[sub]), n_layers))
        trace = {i: float(corr["s_l"][str(i)]["trace"]) for i in range(n_layers)}
        c_l = {i: float(corr["c_l"][str(i)]) for i in range(n_layers)}
        recovered = {i: float(corr["recovered"][str(i)]) for i in range(n_layers)}
        c_exact = _agg_layers(eapn["exact_effect_mean"], n_layers)          
        g_ell = ({i: float(pred["g_ell"][str(i)]) for i in range(n_layers)} if pred else None)

        state = _load_state()
        if state.get("config") and {k: state["config"].get(k) for k in CFG} != CFG:
            raise RuntimeError("checkpoint config mismatch -- refusing to mix runs.\n loaded %s\n cur %s"
                               % ({k: state["config"].get(k) for k in CFG}, CFG))
        state["config"] = dict(CFG)
        cache = state.get("point_cache", {})
        state["point_cache"] = cache
        _save(state)

        if "perturbation" not in state:
            open(status, "w").write("PERTURBATION\n")
            pert = perturbation_per_layer(model, n_layers)
            state["perturbation"] = {str(i): pert[i] for i in range(n_layers)}
            _save(state)
            print("perturbation ||Q(W)-W||^2 per layer computed (reused WeightQuant.apply)", flush=True)
        pert = {int(k): float(v) for k, v in state["perturbation"].items()}
        hawq_v2_full = {i: trace[i] * pert[i] for i in range(n_layers)}          
        hawq_v2_abs = {i: abs(trace[i]) * pert[i] for i in range(n_layers)}      
        state["trace"] = {str(i): trace[i] for i in range(n_layers)}
        state["hawq_v2_full"] = {str(i): hawq_v2_full[i] for i in range(n_layers)}
        state["hawq_v2_absfull"] = {str(i): hawq_v2_abs[i] for i in range(n_layers)}

        desc = lambda d: sorted(range(n_layers), key=lambda l: d[l], reverse=True)
        rankings = {
            "exact":         [int(x) for x in np.argsort(-c_exact)],
            "c_l":           desc(c_l),
            "hawq_v2_full":  desc(hawq_v2_full),
            "hawq_v2_absfull": desc(hawq_v2_abs),
            "hawq_trace":    desc(trace),
            "oracle":        desc(recovered),
        }
        if g_ell is not None:
            rankings["g_l"] = desc(g_ell)
        state["rankings"] = rankings
        print("hawq_v2_full top10:", rankings["hawq_v2_full"][:10], flush=True)
        print("hawq_trace   top10:", rankings["hawq_trace"][:10], flush=True)
        print("exact        top10:", rankings["exact"][:10], flush=True)

        tr_arr = np.array([trace[i] for i in range(n_layers)])
        hv_arr = np.array([hawq_v2_full[i] for i in range(n_layers)])
        cl_arr = np.array([c_l[i] for i in range(n_layers)])
        rec_arr = np.array([recovered[i] for i in range(n_layers)])
        corrs = {
            "spearman_hawqv2full_vs_exact": _spearman(hv_arr, c_exact),
            "spearman_hawqv2full_vs_hawqtrace": _spearman(hv_arr, tr_arr),
            "spearman_hawqv2full_vs_c_l": _spearman(hv_arr, cl_arr),
            "spearman_hawqv2full_vs_oracle": _spearman(hv_arr, rec_arr),
            "spearman_hawqtrace_vs_exact": _spearman(tr_arr, c_exact),
            "spearman_hawqtrace_vs_oracle": _spearman(tr_arr, rec_arr),
        }
        state["correlations"] = corrs
        _save(state)
        print("Spearman  hawqv2full~exact=%+.3f  ~hawqtrace=%+.3f  ~c_l=%+.3f  ~oracle=%+.3f"
              % (corrs["spearman_hawqv2full_vs_exact"], corrs["spearman_hawqv2full_vs_hawqtrace"],
                 corrs["spearman_hawqv2full_vs_c_l"], corrs["spearman_hawqv2full_vs_oracle"]), flush=True)

        open(status, "w").write("CORPORA\n")
        train_b, val_ids, test_ids, _score = G.build_corpora(tok, dev)
        print("corpora: train %d | val %d | test %d seqs" %
              (len(train_b) * G.BATCH, val_ids.shape[0], test_ids.shape[0]), flush=True)

        open(status, "w").write("CEILING\n")
        gpath = greedy["greedy_path"]
        state.setdefault("greedy_ceiling", {})
        for B in BUDGETS:
            if str(B) in state["greedy_ceiling"]:
                continue
            if B > len(gpath):
                raise RuntimeError("budget %d exceeds stored greedy path length %d" % (B, len(gpath)))
            layers = gpath[B - 1]["layers"]                 
            pt = _measure(model, plp, layers, train_b, val_ids, test_ids, SEEDS, cache, state)
            state["greedy_ceiling"][str(B)] = pt
            _save(state)
            print("  [greedy-ceiling] B=%2d %s -> ppl=%.3f" % (B, sorted(layers), pt["ppl_mean"]), flush=True)

        open(status, "w").write("FRONTS\n")
        state.setdefault("fronts", {})
        det_signals = ["exact", "c_l", "hawq_v2_full", "hawq_trace"]
        det_signals += ["hawq_v2_absfull"]
        if g_ell is not None:
            det_signals += ["g_l"]
        det_signals += ["oracle"]
        for sig in det_signals:
            print("=== front: %s (%d seeds) ===" % (sig, len(SEEDS)), flush=True)
            state["fronts"][sig] = _front_for_order(
                model, plp, rankings[sig], BUDGETS, train_b, val_ids, test_ids, SEEDS, cache, state, sig)
            _save(state)

        open(status, "w").write("RANDOM\n")
        state.setdefault("random_draws", {})     
        for B in BUDGETS:
            state["random_draws"].setdefault(str(B), [])
        for r in range(N_RANDOM):
            rng = np.random.default_rng(1000 + r)
            order = [int(x) for x in rng.permutation(n_layers)]
            for B in BUDGETS:
                if len(state["random_draws"][str(B)]) > r:
                    continue                       
                pt = _measure(model, plp, order[:B], train_b, val_ids, test_ids, SEEDS, cache, state)
                state["random_draws"][str(B)].append(pt["ppl_mean"])
                _save(state)
                print("  [random d%d] B=%2d %s -> ppl=%.3f" % (r, B, sorted(order[:B]), pt["ppl_mean"]), flush=True)
        random_front = {}
        for B in BUDGETS:
            draws = state["random_draws"][str(B)]
            random_front[str(B)] = {"ppl_median": float(np.median(draws)),
                                    "ppl_p05": float(np.percentile(draws, 5)),
                                    "ppl_draws": [float(x) for x in draws]}
        state["fronts"]["random"] = random_front
        _save(state)

        ceil = {B: state["greedy_ceiling"][str(B)]["ppl_mean"] for B in BUDGETS}
        gaps = {}
        for sig in det_signals:
            gaps[sig] = {str(B): float(state["fronts"][sig][str(B)]["ppl_mean"] - ceil[B]) for B in BUDGETS}
        gaps["random"] = {str(B): float(random_front[str(B)]["ppl_median"] - ceil[B]) for B in BUDGETS}
        state["gaps"] = gaps

        mean_gap = {sig: float(np.mean([gaps[sig][str(B)] for B in BUDGETS])) for sig in gaps}
        
        beats_random = sum(gaps["hawq_v2_full"][str(B)] < gaps["random"][str(B)] for B in BUDGETS)
        beats_trace = sum(gaps["hawq_v2_full"][str(B)] < gaps["hawq_trace"][str(B)] for B in BUDGETS)
        order_by_gap = sorted(mean_gap, key=mean_gap.get)
        verdict = {
            "mean_gap_ppl": mean_gap,
            "ordering_best_to_worst": order_by_gap,
            "hawq_v2_full_beats_random_per_budget": [beats_random, len(BUDGETS)],
            "hawq_v2_full_beats_hawq_trace_per_budget": [beats_trace, len(BUDGETS)],
            "curvature_still_worse_than_random": bool(mean_gap["hawq_v2_full"] > mean_gap["random"]),
            "full_metric_helps_over_trace": bool(mean_gap["hawq_v2_full"] < mean_gap["hawq_trace"]),
            "base_ppl": greedy["base"]["ppl"],
        }
        state["verdict"] = verdict
        state["wall_s"] = time.time() - t0
        _save(state)

        print("\n=== mean PPL gap to greedy ceiling (lower = better; base %.3f) ===" % greedy["base"]["ppl"])
        for sig in order_by_gap:
            print("  %-15s %.3f" % (sig, mean_gap[sig]))
        print("hawq_v2_full beats random at %d/%d budgets | beats trace-only at %d/%d"
              % (beats_random, len(BUDGETS), beats_trace, len(BUDGETS)))
        print("=> curvature STILL worse than random: %s | full metric helps over trace-only: %s"
              % (verdict["curvature_still_worse_than_random"], verdict["full_metric_helps_over_trace"]))

        open(DONE, "w").write("DONE %.0fs\n" % (time.time() - t0))
        open(status, "w").write("DONE %.0fs\n" % (time.time() - t0))
        print("DONE %.1fs -> %s" % (time.time() - t0, OUT_JSON), flush=True)
    except Exception:
        tb = traceback.format_exc()
        print(tb)
        open(status, "w").write("ERROR\n" + tb)
        raise

if __name__ == "__main__":
    main()
