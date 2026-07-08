
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
import torch  

from experiments.interp.eap_ig_2b import MID, OUT, DRIVE  
from experiments.interp.h2_2b_correction import (  
    RANK, ALPHA, install_unified_forward, attach_adapters,
    set_layers_active, set_all_prec,
)
from experiments.interp.h2_2b_greedy import (  
    REPORT_BUDGETS, SEEDS_FINAL, build_corpora, measure_path_point,
)

GRAD_SEEDS = [0, 1, 2]     
N_GRAD_BATCHES = 16        
SEED = 0

RUN2 = os.path.join(ROOT, "reports/data/h2_2b_correction.json")   
GREEDY = os.path.join(ROOT, "reports/data/h2_2b_greedy.json")     
PDIR = os.path.join(DRIVE, "h2_2b_predictor_checkpoints")
PCKPT = os.path.join(PDIR, "h2_2b_predictor_state.json")
PLOCAL = os.path.join(OUT, "h2_2b_predictor.json")

CFG = {
    "model": MID, "rank": RANK, "alpha": ALPHA, "grad_seeds": GRAD_SEEDS,
    "n_grad_batches": N_GRAD_BATCHES, "report_budgets": REPORT_BUDGETS,
    "seeds_final": SEEDS_FINAL, "seed": SEED,
    "signal": "g_ell = ||dL/dB_ell|| at B=0 (one backward), full L2 norm summed over layer projections",
}

def _atomic_dump(obj, path):
    tmp = path + ".tmp"
    json.dump(obj, open(tmp, "w"), indent=2)
    os.replace(tmp, path)

def _load_state():
    for p in (PCKPT, PLOCAL):
        if os.path.exists(p):
            try:
                return json.load(open(p))
            except Exception:
                pass
    return {}

def _save_state(state):
    os.makedirs(PDIR, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)
    _atomic_dump(state, PCKPT)
    _atomic_dump(state, PLOCAL)

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

def compute_grad_signal(model, plp, train_b):
    n_layers = len(plp)
    set_all_prec(model, "ternary")
    set_layers_active(model, list(range(n_layers)))     
    for li in range(n_layers):
        for (A, B) in plp[li]:
            A.requires_grad_(False); B.requires_grad_(True); B.grad = None
    nb = min(N_GRAD_BATCHES, len(train_b))
    G = np.zeros((len(GRAD_SEEDS), n_layers))
    for si, s in enumerate(GRAD_SEEDS):
        gen = torch.Generator(device="cpu").manual_seed(1000 + s)
        with torch.no_grad():                            
            for li in range(n_layers):
                for (A, B) in plp[li]:
                    d_in = A.shape[1]
                    A.copy_((torch.randn(A.shape, generator=gen) / math.sqrt(d_in)).to(A))
                    B.zero_(); B.grad = None
        for i in range(nb):
            b = train_b[i]
            model(input_ids=b, labels=b).loss.backward()  
        for li in range(n_layers):
            sq = sum(float((B.grad ** 2).sum().item()) for (A, B) in plp[li])
            G[si, li] = math.sqrt(sq) / nb
        for li in range(n_layers):                       
            for (A, B) in plp[li]:
                B.grad = None
    for li in range(n_layers):                           
        for (A, B) in plp[li]:
            B.requires_grad_(False); B.grad = None; B.zero_()
    set_layers_active(model, [])
    return G

def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    status = os.path.join(OUT, "H2_2B_PREDICTOR_STATUS.txt")
    open(status, "w").write("LOADING\n")
    t0 = time.time()
    try:
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
        s_trace = np.array([run2["s_l"][str(i)]["trace"] for i in range(n_layers)])
        c_val = np.array([float(run2["c_l"][str(i)]) for i in range(n_layers)])
        recov = np.array([float(run2["recovered"][str(i)]) for i in range(n_layers)])
        greedy_order = [d["added"] for d in greedy["greedy_path"]]            
        cmp_existing = greedy["comparison"]                                   

        state = _load_state()
        if "config" in state and {k: state["config"].get(k) for k in CFG} != CFG:
            raise RuntimeError("checkpoint config mismatch")
        state["config"] = dict(CFG); state["extra_mem_frac_per_layer"] = extra

        open(status, "w").write("GRAD_SIGNAL\n")
        if "g_ell" not in state:
            print("=== computing g_ell = ||dL/dB|| at B=0 over %d A-inits x %d batches ===" %
                  (len(GRAD_SEEDS), N_GRAD_BATCHES))
            G = compute_grad_signal(model, plp, train_b)
            g_mean = G.mean(0)
            seed_rankings = [list(np.argsort(-G[i])) for i in range(len(GRAD_SEEDS))]
            stab = float(np.mean([_spearman(G[i], G[j])
                                  for i in range(len(GRAD_SEEDS)) for j in range(i + 1, len(GRAD_SEEDS))]))
            state["g_ell"] = {str(i): float(g_mean[i]) for i in range(n_layers)}
            state["g_ell_per_seed"] = {str(s): [float(x) for x in G[i]] for i, s in enumerate(GRAD_SEEDS)}
            state["g_ell_seed_stability_spearman"] = stab
            _save_state(state)
            print("  g_ell rank-stability across A-inits: spearman=%.3f" % stab)
        g_mean = np.array([state["g_ell"][str(i)] for i in range(n_layers)])
        g_order = [int(x) for x in np.argsort(-g_mean)]
        print("  g_ell ranking (top10): %s" % g_order[:10])
        print("  greedy order         : %s" % greedy_order)

        corr = {
            "spearman_g_vs_c_causal": _spearman(g_mean, c_val),
            "spearman_g_vs_s_hawq": _spearman(g_mean, s_trace),
            "spearman_g_vs_recovered": _spearman(g_mean, recov),
            
            "greedy_layers": greedy_order,
        }
        g_rank = {li: r for r, li in enumerate(g_order)}        
        c_rank = {li: r for r, li in enumerate(np.argsort(-c_val))}
        s_rank = {li: r for r, li in enumerate(np.argsort(-s_trace))}
        add_pos = list(range(len(greedy_order)))                
        corr["spearman_greedyorder_vs_g"] = _spearman(add_pos, [g_rank[l] for l in greedy_order])
        corr["spearman_greedyorder_vs_c"] = _spearman(add_pos, [c_rank[l] for l in greedy_order])
        corr["spearman_greedyorder_vs_s"] = _spearman(add_pos, [s_rank[l] for l in greedy_order])
        
        corr["topB_overlap_with_greedy"] = {}
        for B in REPORT_BUDGETS:
            gset, grdy = set(g_order[:B]), set(greedy_order[:B])
            cset = set(int(x) for x in np.argsort(-c_val)[:B])
            corr["topB_overlap_with_greedy"][str(B)] = {
                "g_ell": len(gset & grdy), "c_causal": len(cset & grdy), "of": B}
        state["correlations"] = corr
        _save_state(state)
        print("  spearman g~c=%.3f g~HAWQ=%.3f g~recovered=%.3f | greedy-order~g=%.3f ~c=%.3f ~HAWQ=%.3f"
              % (corr["spearman_g_vs_c_causal"], corr["spearman_g_vs_s_hawq"],
                 corr["spearman_g_vs_recovered"], corr["spearman_greedyorder_vs_g"],
                 corr["spearman_greedyorder_vs_c"], corr["spearman_greedyorder_vs_s"]))

        open(status, "w").write("FRONT\n")
        print("=== g_ell top-B joint front (%d seeds, identical 512-test) ===" % len(SEEDS_FINAL))
        state.setdefault("g_front", {})
        for B in REPORT_BUDGETS:
            if str(B) in state["g_front"]:
                continue
            t = time.time()
            layers = sorted(g_order[:B])
            pt = measure_path_point(model, plp, layers, train_b, val_ids, test_ids, SEEDS_FINAL)
            state["g_front"][str(B)] = pt; _save_state(state)
            print("  [g-front] B=%2d %s -> ppl=%.3f±%.3f (%.0fs)" %
                  (B, layers, pt["ppl_mean"], pt["ppl_sd"], time.time() - t))

        open(status, "w").write("VERDICT\n")
        table = {}
        g_beats_c = g_beats_grdym = g_beats_grdyp05 = g_beats_greedy = n = 0
        gap_g = gap_c = 0.0
        for B in REPORT_BUDGETS:
            e = cmp_existing[str(B)]
            gP = state["g_front"][str(B)]["ppl_mean"]
            row = {"g_ell": gP, "greedy": e["greedy"], "c_causal": e["c_causal"],
                   "s_hawq": e["s_hawq"], "oracle": e["oracle"],
                   "random_median": e["random_median"], "random_p05": e["random_p05"]}
            table[str(B)] = row
            n += 1
            g_beats_c += gP < e["c_causal"]
            g_beats_grdym += gP < e["random_median"]
            g_beats_grdyp05 += gP < e["random_p05"]
            g_beats_greedy += gP < e["greedy"]
            gap_g += gP - e["greedy"]; gap_c += e["c_causal"] - e["greedy"]
        verdict = {
            "g_beats_c_causal": [g_beats_c, n],
            "g_beats_random_median": [g_beats_grdym, n],
            "g_beats_random_p05": [g_beats_grdyp05, n],
            "g_beats_greedy_ceiling": [g_beats_greedy, n],
            "mean_gap_g_to_greedy": gap_g / n,        
            "mean_gap_c_to_greedy": gap_c / n,        
        }
        state["table"] = table; state["verdict"] = verdict
        state["wall_s"] = time.time() - t0
        _save_state(state)

        print("\n=== CHEAP-PREDICTOR vs greedy ceiling (test ppl; base %.3f) ===" % greedy["base"]["ppl"])
        print("  B    g_ell   greedy   c_causal  s_hawq   oracle   random med[p05]")
        for B in REPORT_BUDGETS:
            r = table[str(B)]
            print("  %2d   %5.2f   %5.2f    %5.2f     %5.2f    %5.2f    %5.2f[%.2f]" %
                  (B, r["g_ell"], r["greedy"], r["c_causal"], r["s_hawq"], r["oracle"],
                   r["random_median"], r["random_p05"]))
        print("g_ell beats c_causal %d/%d | random-med %d/%d | random-p05 %d/%d | greedy %d/%d"
              % (*verdict["g_beats_c_causal"], *verdict["g_beats_random_median"],
                 *verdict["g_beats_random_p05"], *verdict["g_beats_greedy_ceiling"]))
        print("mean gap to greedy ceiling: g_ell=%.3f ppl vs c_causal=%.3f ppl"
              % (verdict["mean_gap_g_to_greedy"], verdict["mean_gap_c_to_greedy"]))
        open(status, "w").write("DONE %.0fs\n" % (time.time() - t0))
        print("DONE %.1fs" % (time.time() - t0))
    except Exception:
        tb = traceback.format_exc()
        print(tb); open(status, "w").write("ERROR\n" + tb)
        raise

if __name__ == "__main__":
    main()
