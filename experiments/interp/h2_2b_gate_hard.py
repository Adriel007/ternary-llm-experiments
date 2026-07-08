
from __future__ import annotations

import json
import os
import sys
import time
import traceback

ROOT = os.environ.get("PHD_ROOT", os.environ.get("POC_ROOT", "/content/PhD-propose"))
sys.path[:0] = [ROOT, os.path.join(ROOT, "packages/bitnet_core/src")]

import numpy as np  
import torch  

from experiments.interp.h2_2b_correction import (  
    MID, OUT, _token_pool, eval_loss, ppl, BATCH, SEQ_LEN,
    N_TRAIN_SEQS, N_VAL_SEQS, N_CALIB_SEQS, RANK, ALPHA,
)
from experiments.interp.h2_2b_gate import (  
    install_gated_forward, attach_gated, joint_train_eval_gated, RANKINGS, BUDGETS, VARIANTS,
    GATE_RANK,
)

SEEDS = [0, 1, 2]
N_TEST_HARD = 512
T95_DF2 = 4.302653            

OUT_JSON = os.path.join(OUT, "h2_2b_gate_hard.json")
STATUS = os.path.join(OUT, "H2_2B_GATE_HARD_STATUS.txt")

def _dump(obj):
    os.makedirs(OUT, exist_ok=True)
    tmp = OUT_JSON + ".tmp"
    json.dump(obj, open(tmp, "w"), indent=2)
    os.replace(tmp, OUT_JSON)

def _status(s):
    os.makedirs(OUT, exist_ok=True)
    open(STATUS, "w").write(s + "\n")

def build_corpora_bigtest(tok, device, n_test):
    sizes = [N_TRAIN_SEQS, N_VAL_SEQS, n_test, N_CALIB_SEQS]
    pool = _token_pool(tok, sum(sizes), SEQ_LEN)
    off = np.cumsum([0] + sizes)
    slices = [pool[off[i]:off[i + 1]] for i in range(4)]
    seen = {}
    for nm, sl in zip(("train", "val", "test", "calib"), slices):
        for row in sl.tolist():
            key = tuple(row)
            assert key not in seen, f"leakage: sequence in both {seen.get(key)} and {nm}"
            seen[key] = nm
    mk = lambda ids: [ids[i:i + BATCH].to(device) for i in range(0, ids.shape[0], BATCH)]
    return tuple(mk(sl) for sl in slices)

def _ci95(vals):
    a = np.asarray(vals, dtype=np.float64)
    m = float(a.mean())
    if a.size < 2:
        return m, 0.0
    sem = float(a.std(ddof=1) / np.sqrt(a.size))
    return m, T95_DF2 * sem

def main():
    _status("LOADING")
    t0 = time.time()
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        tok = AutoTokenizer.from_pretrained(MID)
        model = AutoModelForCausalLM.from_pretrained(MID, dtype=torch.bfloat16).to(dev).eval()
        model.requires_grad_(False)
        n_layers = len(model.model.layers)
        install_gated_forward()
        train_b, val_b, test_b, _calib_b = build_corpora_bigtest(tok, dev, N_TEST_HARD)
        n_test_seqs = sum(b.shape[0] for b in test_b)
        print("loaded %s | layers %d | test seqs %d" % (MID, n_layers, n_test_seqs), flush=True)

        order = RANKINGS["c_causal"]
        
        raw = {v: {str(B): [] for B in BUDGETS} for v in VARIANTS}
        frac = None
        results = {
            "model": MID, "rank": RANK, "alpha": ALPHA, "gate_rank": GATE_RANK,
            "seeds": SEEDS, "n_test_seqs": n_test_seqs, "budgets": BUDGETS,
            "ranking": "c_causal", "variants": VARIANTS, "per_seed": {}, "summary": {},
        }
        _dump(results)

        for seed in SEEDS:
            per_layer, frac = attach_gated(model, RANK, ALPHA, dev, seed)
            base_test = eval_loss(model, test_b)
            results["per_seed"].setdefault(str(seed), {"base_test_ppl": ppl(base_test)})
            print("[seed %d] base test ppl %.4f | frac %s"
                  % (seed, ppl(base_test), {k: round(v, 5) for k, v in frac.items()}), flush=True)
            for variant in VARIANTS:
                results["per_seed"][str(seed)].setdefault(variant, {})
                for B in BUDGETS:
                    t = time.time()
                    layers = sorted(order[:B])
                    loss_b = joint_train_eval_gated(model, layers, variant, train_b, val_b, test_b,
                                                    seed * 7 + B)
                    raw[variant][str(B)].append(ppl(loss_b))
                    results["per_seed"][str(seed)][variant][str(B)] = {
                        "test_ppl": ppl(loss_b), "test_loss": loss_b, "layers": layers}
                    _dump(results)
                    print("  [s%d] %-13s B=%2d -> ppl=%.4f (%.1fs)"
                          % (seed, variant, B, ppl(loss_b), time.time() - t), flush=True)

        results["extra_frac_per_layer"] = frac
        
        summ = {v: {} for v in VARIANTS}
        for variant in VARIANTS:
            for B in BUDGETS:
                m, hw = _ci95(raw[variant][str(B)])
                cell = {"mean_ppl": m, "ci95_halfwidth": hw, "per_seed_ppl": raw[variant][str(B)],
                        "extra_mem_frac": B * (frac["lora"] + (0.0 if variant == "static" else frac[{
                            "scalar_gate": "gate_scalar", "channel_gate": "gate_channel",
                            "input_gate": "gate_input"}[variant]]))}
                if variant != "static":
                    deltas = [s - g for s, g in zip(raw["static"][str(B)], raw[variant][str(B)])]
                    dm, dhw = _ci95(deltas)             
                    cell["delta_vs_static_per_seed"] = deltas
                    cell["delta_mean"] = dm
                    cell["delta_ci95_halfwidth"] = dhw
                    cell["wins_of_n"] = int(sum(d > 0 for d in deltas))
                    cell["ci_excludes_zero"] = bool(dm - dhw > 0)
                summ[variant][str(B)] = cell
        results["summary"] = summ
        results["wall_s"] = time.time() - t0
        _dump(results)

        print("\n==== GATE HARDENING SUMMARY (%d seeds, %d-seq test) ===="
              % (len(SEEDS), n_test_seqs), flush=True)
        for B in BUDGETS:
            row = "  B=%2d " % B
            for v in VARIANTS:
                c = summ[v][str(B)]
                tag = v.replace("_gate", "G").replace("static", "stat")
                row += " %s=%.3f±%.3f" % (tag, c["mean_ppl"], c["ci95_halfwidth"])
            print(row, flush=True)
        print("\n  gate beats static (paired, CI excludes 0):", flush=True)
        for v in VARIANTS:
            if v == "static":
                continue
            wins = [(B, summ[v][str(B)]["wins_of_n"], summ[v][str(B)]["ci_excludes_zero"])
                    for B in BUDGETS]
            print("    %-13s %s" % (v, wins), flush=True)
        _status("DONE %.0fs" % (time.time() - t0))
        print("\nDONE %.1fs" % (time.time() - t0), flush=True)
    except Exception:
        tb = traceback.format_exc()
        print(tb, flush=True)
        _status("ERROR\n" + tb)
        raise

if __name__ == "__main__":
    main()
