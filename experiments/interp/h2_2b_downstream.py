
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

from experiments.interp.h2_2b_correction import (  
    MID, OUT,
    install_unified_forward, attach_adapters, set_layers_active, set_all_prec,
    build_corpora, joint_train_eval, ppl,
    RANK, ALPHA, SEED,
)

RANKINGS = {
    "c_causal": [0, 1, 2, 29, 19, 5, 4, 28, 27, 9, 25, 3, 20, 26, 13, 8, 24, 10, 23, 14,
                 21, 7, 15, 12, 22, 16, 18, 11, 6, 17],
    "s_hawq":   [1, 0, 3, 5, 9, 6, 2, 4, 13, 8, 15, 7, 11, 19, 14, 21, 12, 24, 10, 25,
                 26, 16, 18, 17, 20, 27, 22, 23, 28, 29],
    "random":   [2, 11, 26, 21, 10, 4, 28, 16, 23, 6, 18, 25, 3, 29, 8, 0, 19, 12, 20, 13,
                 7, 5, 17, 14, 22, 9, 27, 24, 1, 15],
    "greedy":   [28, 10, 29, 26, 1, 0, 27, 18],   
}

PROBE_TASKS = ["hellaswag", "arc_challenge", "winogrande", "piqa"]

CONFIRM_TASKS = ["hellaswag", "arc_challenge", "arc_easy", "winogrande", "piqa", "openbookqa"]
FULL_TASKS = CONFIRM_TASKS

NAMED_TASKS = ["mmlu", "gsm8k"]
NAMED_FEWSHOT = 5
PROBE_LIMIT = 500            
FULL_LIMIT = None            
FULL_BUDGETS = [2, 5, 8]
EVAL_BSZ = 16
NAMED_BSZ = "auto"          

STATUS = os.path.join(OUT, "H2_2B_DOWNSTREAM_STATUS.txt")
OUT_JSON = os.path.join(OUT, "h2_2b_downstream.json")
DRIVE_JSON = "/content/drive/MyDrive/PhD_PoC/h2_2b_downstream.json"

def _dump(obj):
    os.makedirs(OUT, exist_ok=True)
    for p in (OUT_JSON, DRIVE_JSON):
        try:
            tmp = p + ".tmp"
            json.dump(obj, open(tmp, "w"), indent=2)
            os.replace(tmp, p)
        except Exception:
            pass

def _status(s):
    os.makedirs(OUT, exist_ok=True)
    open(STATUS, "w").write(s + "\n")

def train_and_activate(model, per_layer_params, layers, train_b, val_b, tag):
    if not layers:
        set_all_prec(model, "ternary"); set_layers_active(model, [])
        return None
    t = time.time()

    val_loss = joint_train_eval(model, per_layer_params, sorted(layers), train_b, val_b, val_b,
                                SEED * 13 + len(layers))
    set_all_prec(model, "ternary"); set_layers_active(model, sorted(layers))
    print("  [train] %-14s layers %s -> val ppl %.3f (%.1fs)"
          % (tag, sorted(layers), ppl(val_loss), time.time() - t), flush=True)
    return ppl(val_loss)

def run_lm_eval(lm, tasks, limit, num_fewshot=None):
    from lm_eval import simple_evaluate
    res = simple_evaluate(model=lm, tasks=tasks, limit=limit, num_fewshot=num_fewshot,
                          bootstrap_iters=0, verbosity="ERROR")
    out = {}
    
    rows = dict(res.get("results", {}))
    rows.update(res.get("groups", {}) or {})
    for task, d in rows.items():
        rec = {}
        for k, v in d.items():
            if isinstance(v, (int, float)) and ("acc" in k or "exact_match" in k):
                rec[k] = float(v)
        if rec:
            out[task] = rec
    return out

def _primary_acc(task_metrics):
    prim = {}
    for task, rec in task_metrics.items():
        for cand in ("acc_norm,none", "acc,none",
                     "exact_match,strict-match", "exact_match,flexible-extract", "exact_match,none",
                     "acc_norm", "acc", "exact_match"):
            if cand in rec:
                prim[task] = (cand, rec[cand]); break
    return prim

def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "probe"
    assert mode in ("probe", "confirm", "full", "named"), "mode must be probe|confirm|full|named"
    tasks = {"probe": PROBE_TASKS, "confirm": CONFIRM_TASKS,
             "full": FULL_TASKS, "named": NAMED_TASKS}[mode]
    limit = PROBE_LIMIT if mode == "probe" else FULL_LIMIT
    fewshot = NAMED_FEWSHOT if mode == "named" else None
    if mode == "named":  
        global OUT_JSON, DRIVE_JSON
        OUT_JSON = os.path.join(OUT, "h2_2b_downstream_named.json")
        DRIVE_JSON = "/content/drive/MyDrive/PhD_PoC/h2_2b_downstream_named.json"
    _status("LOADING %s" % mode)
    t0 = time.time()
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from lm_eval.models.huggingface import HFLM
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        tok = AutoTokenizer.from_pretrained(MID)
        model = AutoModelForCausalLM.from_pretrained(MID, dtype=torch.bfloat16).to(dev).eval()
        model.requires_grad_(False)
        n_layers = len(model.model.layers)
        install_unified_forward()
        per_layer_params, extra_frac = attach_adapters(model, RANK, ALPHA, dev, SEED)
        set_all_prec(model, "ternary"); set_layers_active(model, [])
        print("loaded %s | layers %d | extra-mem/layer %.4f (rank %d) | mode %s"
              % (MID, n_layers, extra_frac, RANK, mode), flush=True)

        
        train_b, val_b, _test_b, _calib_b = build_corpora(tok, dev)
        print("corpora: train %d | val %d seqs" % (len(train_b) * 4, len(val_b) * 4), flush=True)

        lm = HFLM(pretrained=model, tokenizer=tok,
                  batch_size=NAMED_BSZ if mode == "named" else EVAL_BSZ)

        if mode in ("probe", "confirm", "named"):
            allocs = [("base", []), ("greedy@8", RANKINGS["greedy"][:8])]
        else:
            allocs = [("base", [])]
            for B in FULL_BUDGETS:
                for name in ("c_causal", "s_hawq", "random", "greedy"):
                    allocs.append(("%s@%d" % (name, B), RANKINGS[name][:B]))

        results = {"mode": mode, "model": MID, "rank": RANK, "alpha": ALPHA,
                   "extra_mem_frac_per_layer": extra_frac, "tasks": tasks, "limit": limit,
                   "num_fewshot": fewshot,
                   "budgets": FULL_BUDGETS if mode == "full" else None,
                   "subsampled": limit is not None, "allocations": {}}
        _dump(results)

        for tag, layers in allocs:
            _status("%s : %s" % (mode, tag))
            vppl = train_and_activate(model, per_layer_params, layers, train_b, val_b, tag)
            te = time.time()
            tm = run_lm_eval(lm, tasks, limit, num_fewshot=fewshot)
            prim = _primary_acc(tm)

            
            prim = {t: prim[t] for t in tasks if t in prim}
            mean_acc = float(np.mean([v for _, v in prim.values()])) if prim else None
            results["allocations"][tag] = {
                "layers": sorted(layers), "n_layers": len(layers),
                "extra_mem_frac": len(layers) / n_layers * extra_frac,
                "val_ppl": vppl, "metrics": tm,
                "primary": {k: {"metric": m, "value": v} for k, (m, v) in prim.items()},
                "mean_primary_acc": mean_acc,
                "eval_s": time.time() - te,
            }
            _dump(results)
            print("  [eval] %-14s mean-acc=%.4f | %s (%.0fs)"
                  % (tag, mean_acc if mean_acc is not None else float("nan"),
                     " ".join("%s=%.3f" % (t, v) for t, (_, v) in prim.items()),
                     time.time() - te), flush=True)

        results["wall_s"] = time.time() - t0
        _dump(results)

        A = results["allocations"]
        base = A.get("base", {}).get("mean_primary_acc")
        print("\n==== DOWNSTREAM ACCURACY SUMMARY (%s) ====" % mode, flush=True)
        print("base mean-primary-acc = %.4f" % (base if base is not None else float("nan")))
        if mode in ("probe", "confirm", "named"):
            ceil = A.get("greedy@8", {}).get("mean_primary_acc")
            delta = (ceil - base) if (ceil is not None and base is not None) else None
            print("greedy@8 (ceiling)    = %.4f  | delta vs base = %+.4f"
                  % (ceil if ceil else float("nan"), delta if delta is not None else float("nan")))
            print("PER-TASK base vs ceiling (delta; ~2*SE flags significance):")
            for task in tasks:
                bp = A["base"]["primary"].get(task, {})
                cp = A["greedy@8"]["primary"].get(task, {})
                b, c = bp.get("value"), cp.get("value")
                if b is None or c is None:
                    continue
                
                mname = bp.get("metric", "")
                se_key = mname.replace("acc", "acc_stderr").replace("exact_match", "exact_match_stderr")
                se_b = A["base"]["metrics"].get(task, {}).get(se_key)
                se_c = A["greedy@8"]["metrics"].get(task, {}).get(se_key)
                se = (se_b ** 2 + se_c ** 2) ** 0.5 if (se_b and se_c) else None
                sig = ("  [%.1f SE]" % (abs(c - b) / se)) if se else ""
                print("  %-16s %.4f -> %.4f  (%+.4f)%s" % (task, b, c, c - b, sig))
            print("\nGATE: a ceiling gain significant on some task (>~2 SE) => the metric "
                  "discriminates there -> run the c-vs-HAWQ sweep on it. If every task washes "
                  "out even on full sets, the LM-loss correction does NOT transfer to accuracy "
                  "(honest C3 floor: ternary 2B already near-FP downstream).")
        else:
            print("Pareto (mean-primary-acc by allocation):")
            for tag in sorted(A.keys()):
                a = A[tag]
                print("  %-14s B=%2d acc=%.4f mem+%.4f" %
                      (tag, a["n_layers"], a["mean_primary_acc"] or float("nan"),
                       a["extra_mem_frac"]))
        results["verdict_base_acc"] = base
        _dump(results)
        _status("DONE %s %.0fs" % (mode, time.time() - t0))
        print("\nDONE %s %.1fs" % (mode, time.time() - t0), flush=True)
    except Exception:
        tb = traceback.format_exc()
        print(tb, flush=True)
        _status("ERROR\n" + tb)
        raise

if __name__ == "__main__":
    main()
