
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
    LR, WD, RANK, ALPHA, install_unified_forward, attach_adapters,
    set_layers_active, set_all_prec, ppl,
)
from experiments.interp.h2_2b_greedy import (  
    REPORT_BUDGETS, SEEDS_FINAL, SEED, CAND_STEPS, MAX_BUDGET,
    build_corpora, measure_path_point, joint_train, eval_mean_loss,
)
from experiments.interp.h2_2b_predictor import _spearman  
from experiments.env_stamp import env_stamp  

PROBE_MODE = (sys.argv[1] if len(sys.argv) > 1 else "frozen").strip()
assert PROBE_MODE in ("frozen", "jointwarm"), "PROBE_MODE must be frozen|jointwarm"
K_PROBE = 20               
K_PROBE_JW = 40            
N_GRAD_BATCHES = 16        
SEARCH_STEPS = CAND_STEPS  

GREEDY = os.path.join(ROOT, "reports/data/h2_2b_greedy.json")     
PREDICTOR = os.path.join(ROOT, "reports/data/h2_2b_predictor.json")  
_SUF = "" if PROBE_MODE == "frozen" else "_" + PROBE_MODE
CGDIR = os.path.join(DRIVE, "h2_2b_cheapgreedy_checkpoints")
CGCKPT = os.path.join(CGDIR, "h2_2b_cheapgreedy_state%s.json" % _SUF)
CGLOCAL = os.path.join(OUT, "h2_2b_cheapgreedy%s.json" % _SUF)

CFG = {
    "model": MID, "rank": RANK, "alpha": ALPHA, "lr": LR, "wd": WD,
    "probe_mode": PROBE_MODE,
    "k_probe": (K_PROBE if PROBE_MODE == "frozen" else K_PROBE_JW),
    "search_steps": SEARCH_STEPS, "n_grad_batches": N_GRAD_BATCHES,
    "max_budget": MAX_BUDGET, "report_budgets": REPORT_BUDGETS, "seeds_final": SEEDS_FINAL,
    "seed": SEED,
    "signal": "cg-probe(%s): greedy with warm candidate fine-tune (marginal loss gain on SCORE set); "
              "cg-grad byproduct = conditional ||dL/dB_j|| given trained S" % PROBE_MODE,
}

def _atomic_dump(obj, path):
    tmp = path + ".tmp"
    json.dump(obj, open(tmp, "w"), indent=2)
    os.replace(tmp, path)

def _load_state():
    for p in (CGCKPT, CGLOCAL):
        if os.path.exists(p):
            try:
                return json.load(open(p))
            except Exception:
                pass
    return {}

def _save_state(state):
    os.makedirs(CGDIR, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)
    _atomic_dump(state, CGCKPT)
    _atomic_dump(state, CGLOCAL)

def _reset_layer(plp, li):
    for (A, B) in plp[li]:
        with torch.no_grad():
            B.zero_()
        B.requires_grad_(False); B.grad = None

def _set_grad(plp, li, on):
    for (A, B) in plp[li]:
        B.requires_grad_(on); B.grad = None

def cond_grad(model, plp, S, cands, train_b):
    set_all_prec(model, "ternary")
    set_layers_active(model, list(S) + list(cands))
    for li in cands:
        _reset_layer(plp, li); _set_grad(plp, li, True)
    nb = min(N_GRAD_BATCHES, len(train_b))
    for i in range(nb):
        b = train_b[i]
        model(input_ids=b, labels=b).loss.backward()
    g = {}
    for li in cands:
        sq = sum(float((B.grad ** 2).sum().item()) for (A, B) in plp[li])
        g[li] = math.sqrt(sq) / nb
    for li in cands:
        _set_grad(plp, li, False); _reset_layer(plp, li)
    set_layers_active(model, [])
    return g

def probe_gain(model, plp, S, j, train_b, score_ids, base_score):
    set_layers_active(model, list(S) + [j])
    _reset_layer(plp, j)
    if PROBE_MODE == "jointwarm":
        snap = {li: [B.detach().clone() for (A, B) in plp[li]] for li in S}   
        train_layers = list(S) + [j]
        steps = K_PROBE_JW
    else:
        train_layers = [j]
        steps = K_PROBE
    for li in train_layers:
        _set_grad(plp, li, True)
    flat = [B for li in train_layers for (A, B) in plp[li]]
    torch.manual_seed(SEED * 101 + j)
    opt = torch.optim.AdamW(flat, lr=LR, weight_decay=WD)
    nb = len(train_b)
    for step in range(1, steps + 1):
        b = train_b[(step - 1) % nb]
        loss = model(input_ids=b, labels=b).loss
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    sc = eval_mean_loss(model, score_ids)
    for li in train_layers:
        _set_grad(plp, li, False)
    _reset_layer(plp, j)
    if PROBE_MODE == "jointwarm":
        with torch.no_grad():                                                
            for li in S:
                for (A, B), Bsnap in zip(plp[li], snap[li]):
                    B.copy_(Bsnap)
    set_layers_active(model, list(S))
    return base_score - sc, sc

def run_cheap_greedy(model, plp, n_layers, train_b, val_ids, score_ids, state):
    state.setdefault("cg_probe_path", [])     
    state.setdefault("cg_grad_path", [])       
    S = list(state["cg_probe_path"][-1]["layers"]) if state["cg_probe_path"] else []
    Sg = list(state["cg_grad_path"][-1]["layers"]) if state["cg_grad_path"] else []
    while len(S) < MAX_BUDGET:
        k = len(S) + 1
        t = time.time()
        
        if S:
            joint_train(model, plp, S, train_b, val_ids, SEARCH_STEPS, SEED)
            base_score = eval_mean_loss(model, score_ids)
            for li in S:                      
                _set_grad(plp, li, False)
        else:
            set_all_prec(model, "ternary"); set_layers_active(model, [])
            base_score = eval_mean_loss(model, score_ids)
        cands = [j for j in range(n_layers) if j not in S]

        gnorm = cond_grad(model, plp, S, cands, train_b)
        if S:                                  
            set_layers_active(model, S)
        jg = max(gnorm, key=gnorm.get)
        Sg = sorted(Sg + [jg]) if jg not in Sg else Sg
        state["cg_grad_path"].append({"k": k, "added": int(jg), "layers": list(Sg),
                                      "grad_norm": float(gnorm[jg])})

        gains, scores = {}, {}
        for j in cands:
            gains[j], scores[j] = probe_gain(model, plp, S, j, train_b, score_ids, base_score)
        jstar = max(gains, key=gains.get)
        S = sorted(S + [jstar])
        state["cg_probe_path"].append({"k": k, "added": int(jstar), "gain": float(gains[jstar]),
                                       "score_ppl": ppl(scores[jstar]), "layers": list(S)})
        _save_state(state)
        print("  [cg] k=%2d  probe-add L%-2d (gain %.4f, score-ppl %.3f)  | grad-add L%-2d "
              "(||g||=%.3e) | %d cand %.0fs" %
              (k, jstar, gains[jstar], ppl(scores[jstar]), jg, gnorm[jg], len(cands), time.time() - t))
    return [d["layers"] for d in state["cg_probe_path"]]

def _selftest(model, plp, n_layers, train_b, score_ids):
    print("[selftest] running...")
    
    set_all_prec(model, "ternary"); set_layers_active(model, [])
    with torch.no_grad():
        base = float(model(input_ids=train_b[0], labels=train_b[0]).loss)
    set_layers_active(model, list(range(n_layers)))   
    with torch.no_grad():
        wired = float(model(input_ids=train_b[0], labels=train_b[0]).loss)
    set_layers_active(model, [])
    d_noop = abs(base - wired)
    assert d_noop < 1e-4, "no-op invariant broken: |dL|=%.2e (B=0 adapters change forward)" % d_noop
    
    probe = [0, n_layers // 2]
    g_all = cond_grad(model, plp, [], probe, train_b[:1])      
    g_one = {}
    for j in probe:
        set_all_prec(model, "ternary"); set_layers_active(model, [j])
        _reset_layer(plp, j); _set_grad(plp, j, True)
        model(input_ids=train_b[0], labels=train_b[0]).loss.backward()
        g_one[j] = math.sqrt(sum(float((B.grad ** 2).sum().item()) for (A, B) in plp[j]))
        _set_grad(plp, j, False); _reset_layer(plp, j); set_layers_active(model, [])
    rel = max(abs(g_all[j] - g_one[j]) / (g_one[j] + 1e-12) for j in probe)
    assert rel < 1e-3, "cond_grad one-shot != per-candidate: rel=%.2e" % rel
    
    set_all_prec(model, "ternary"); set_layers_active(model, [])
    base_sc = eval_mean_loss(model, score_ids)
    gain, sc = probe_gain(model, plp, [], 0, train_b, score_ids, base_sc)
    print("[selftest] no-op |dL|=%.2e | cond-grad rel=%.2e | probe gain(L0)=%+.4f (%.3f->%.3f)"
          % (d_noop, rel, gain, ppl(base_sc), ppl(sc)))
    print("[selftest] PASS")

def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    status = os.path.join(OUT, "H2_2B_CHEAPGREEDY_STATUS.txt")
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

        train_b, val_ids, test_ids, score_ids = build_corpora(tok, dev)
        print("corpora (disjoint): train %d-b | val %d | test %d | score %d" %
              (len(train_b), val_ids.shape[0], test_ids.shape[0], score_ids.shape[0]))

        greedy = json.load(open(GREEDY))
        greedy_order = [d["added"] for d in greedy["greedy_path"]]
        cmp_existing = greedy["comparison"]
        try:
            pred = json.load(open(PREDICTOR))
            g_init_order = [int(x) for x in np.argsort(
                -np.array([pred["g_ell"][str(i)] for i in range(n_layers)]))]
        except Exception:
            g_init_order = None

        open(status, "w").write("SELFTEST\n")
        _selftest(model, plp, n_layers, train_b, score_ids)

        state = _load_state()
        if "config" in state and {k: state["config"].get(k) for k in CFG} != CFG:
            raise RuntimeError("checkpoint config mismatch")
        state["config"] = dict(CFG); state["extra_mem_frac_per_layer"] = extra
        state["env"] = env_stamp()                          
        _save_state(state)

        open(status, "w").write("SEARCH\n")
        print("=== cheap greedy search (mode=%s, K=%d warm; cg-grad byproduct) ===" %
              (PROBE_MODE, K_PROBE if PROBE_MODE == "frozen" else K_PROBE_JW))
        run_cheap_greedy(model, plp, n_layers, train_b, val_ids, score_ids, state)
        cg_order = [state["cg_probe_path"][i]["added"] for i in range(len(state["cg_probe_path"]))]
        grad_order = [state["cg_grad_path"][i]["added"] for i in range(len(state["cg_grad_path"]))]
        print("  cg-probe path : %s" % cg_order)
        print("  cg-grad  path : %s" % grad_order)
        print("  greedy   path : %s" % greedy_order[:MAX_BUDGET])
        if g_init_order is not None:
            print("  g_ell(init)   : %s" % g_init_order[:MAX_BUDGET])

        ov = {}
        for B in REPORT_BUDGETS:
            grdy = set(greedy_order[:B])
            ov[str(B)] = {"cg_probe": len(set(cg_order[:B]) & grdy),
                          "cg_grad": len(set(grad_order[:B]) & grdy),
                          "g_init": (len(set(g_init_order[:B]) & grdy) if g_init_order else None),
                          "of": B}
        state["topB_overlap_with_greedy"] = ov
        
        cgrank = {li: r for r, li in enumerate(cg_order)}
        state["spearman_greedyorder_vs_cgprobe"] = _spearman(
            list(range(len(greedy_order))),
            [cgrank.get(l, len(cg_order)) for l in greedy_order])
        _save_state(state)

        open(status, "w").write("FRONT\n")
        print("=== cg-probe top-B joint front (%d seeds, identical 512-test) ===" % len(SEEDS_FINAL))
        state.setdefault("cg_front", {})
        for B in REPORT_BUDGETS:
            if str(B) in state["cg_front"]:
                continue
            t = time.time()
            layers = sorted(cg_order[:B])
            pt = measure_path_point(model, plp, layers, train_b, val_ids, test_ids, SEEDS_FINAL)
            state["cg_front"][str(B)] = pt; _save_state(state)
            print("  [cg-front] B=%2d %s -> ppl=%.3f±%.3f (%.0fs)" %
                  (B, layers, pt["ppl_mean"], pt["ppl_sd"], time.time() - t))

        open(status, "w").write("VERDICT\n")
        table = {}
        beats_c = beats_rm = beats_rp = beats_greedy = n = 0
        gap_cg = gap_c = 0.0
        for B in REPORT_BUDGETS:
            e = cmp_existing[str(B)]
            cgP = state["cg_front"][str(B)]["ppl_mean"]
            table[str(B)] = {"cg_probe": cgP, "greedy": e["greedy"], "c_causal": e["c_causal"],
                             "s_hawq": e["s_hawq"], "oracle": e["oracle"],
                             "random_median": e["random_median"], "random_p05": e["random_p05"]}
            n += 1
            beats_c += cgP < e["c_causal"]
            beats_rm += cgP < e["random_median"]
            beats_rp += cgP < e["random_p05"]
            beats_greedy += cgP < e["greedy"]
            gap_cg += cgP - e["greedy"]; gap_c += e["c_causal"] - e["greedy"]
        verdict = {
            "cg_beats_c_causal": [beats_c, n],
            "cg_beats_random_median": [beats_rm, n],
            "cg_beats_random_p05": [beats_rp, n],
            "cg_beats_greedy_ceiling": [beats_greedy, n],
            "mean_gap_cg_to_greedy": gap_cg / n,
            "mean_gap_c_to_greedy": gap_c / n,
            "rescued": bool(beats_c >= 4 and (gap_cg / n) < (gap_c / n)),
        }
        state["table"] = table; state["verdict"] = verdict
        state["wall_s"] = time.time() - t0
        _save_state(state)

        print("\n=== CHEAP-FAITHFUL-GREEDY vs greedy ceiling (test ppl; base %.3f) ===" % greedy["base"]["ppl"])
        print("  B   cg_probe  greedy  c_causal  s_hawq  oracle  random med[p05]")
        for B in REPORT_BUDGETS:
            r = table[str(B)]
            print("  %2d   %6.2f  %6.2f   %6.2f   %6.2f  %6.2f   %5.2f[%.2f]" %
                  (B, r["cg_probe"], r["greedy"], r["c_causal"], r["s_hawq"], r["oracle"],
                   r["random_median"], r["random_p05"]))
        print("cg-probe beats c_causal %d/%d | random-med %d/%d | random-p05 %d/%d | greedy %d/%d"
              % (*verdict["cg_beats_c_causal"], *verdict["cg_beats_random_median"],
                 *verdict["cg_beats_random_p05"], *verdict["cg_beats_greedy_ceiling"]))
        print("mean gap to ceiling: cg-probe=%.3f ppl vs c_causal=%.3f ppl | top-B overlap(greedy) B5: %s"
              % (verdict["mean_gap_cg_to_greedy"], verdict["mean_gap_c_to_greedy"], ov["5"]))
        print("VERDICT: %s" % ("RESCUED — cheap faithful greedy tracks the oracle"
                               if verdict["rescued"] else
                               "NOT rescued — cheap search still can't approximate the optimal allocation"))
        open(status, "w").write("DONE %.0fs\n" % (time.time() - t0))
        print("DONE %.1fs" % (time.time() - t0))
    except Exception:
        tb = traceback.format_exc()
        print(tb); open(status, "w").write("ERROR\n" + tb)
        raise

if __name__ == "__main__":
    main()
