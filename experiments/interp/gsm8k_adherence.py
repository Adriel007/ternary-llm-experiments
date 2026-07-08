
from __future__ import annotations

import json
import os
import re
import sys
import time
import traceback

ROOT = os.environ.get("PHD_ROOT", "/content/PhD-propose")
sys.path[:0] = [ROOT, os.path.join(ROOT, "packages/bitnet_core/src")]

import numpy as np  
import torch  

from experiments.interp.h2_2b_correction import (  
    MID, OUT, install_unified_forward, attach_adapters, set_layers_active, set_all_prec,
    build_corpora, joint_train_eval, ppl, RANK, ALPHA, SEED,
)
from experiments.interp.h2_2b_downstream import RANKINGS  

STRICT_RE = re.compile(r"####\s*-?[\d][\d,\.]*")
NUM_RE = re.compile(r"-?[\d][\d,\.]*")
FEWSHOT = 5

ARM_SEL = os.environ.get("ARM", "both")
SH_I, SH_N = (int(x) for x in os.environ.get("SHARD", "0/1").split("/"))
assert 0 <= SH_I < SH_N, "SHARD must be i/N with 0<=i<N"
_suffix = ("" if ARM_SEL == "both" else "_" + ARM_SEL.replace("@", "")) + ("" if SH_N == 1 else "_s%dof%d" % (SH_I, SH_N))
OUT_JSON = os.path.join(OUT, "gsm8k_adherence%s.json" % _suffix)
STATUS = os.path.join(OUT, "GSM8K_ADHERENCE%s_STATUS.txt" % _suffix)

def _dump(obj):
    os.makedirs(OUT, exist_ok=True)
    tmp = OUT_JSON + ".tmp"
    json.dump(obj, open(tmp, "w"), indent=2)
    os.replace(tmp, OUT_JSON)

def _status(s):
    os.makedirs(OUT, exist_ok=True)
    open(STATUS, "w").write(s + "\n")

def _gen_text(sample):
    for key in ("resps", "filtered_resps"):
        v = sample.get(key)
        if isinstance(v, list) and v:
            x = v[0]
            while isinstance(x, list) and x:
                x = x[0]
            if isinstance(x, str):
                return x
    return ""

def eval_gsm8k_shard(lm):
    if SH_N == 1:
        from lm_eval import simple_evaluate
        res = simple_evaluate(model=lm, tasks=["gsm8k"], num_fewshot=FEWSHOT,
                              log_samples=True, bootstrap_iters=0, verbosity="ERROR")
        rows = dict(res.get("results", {})).get("gsm8k", {})
        samples = res.get("samples", {}).get("gsm8k", [])
        return rows, samples
    from lm_eval.tasks import TaskManager, get_task_dict
    from lm_eval.evaluator import evaluate
    td = get_task_dict(["gsm8k"], TaskManager())
    task = td["gsm8k"]
    split = task.config.test_split
    docs = task.test_docs()
    full = len(docs)
    sharded = docs.select(range(SH_I, full, SH_N))
    assert 0 < len(sharded) < full, "shard did not reduce the doc set"
    task.test_docs = lambda: sharded            
    res = evaluate(lm=lm, task_dict=td, limit=None, bootstrap_iters=0, log_samples=True,
                   verbosity="ERROR")
    rows = dict(res.get("results", {})).get("gsm8k", {})
    samples = res.get("samples", {}).get("gsm8k", [])
    return rows, samples

def measure_adherence(samples):
    n = len(samples)
    n_strict_tmpl = 0
    for s in samples:
        gen = _gen_text(s)
        if STRICT_RE.search(gen):
            n_strict_tmpl += 1
    return {"n": n, "n_with_strict_template": n_strict_tmpl,
            "adherence_rate": n_strict_tmpl / n if n else None}

def main():
    _status("LOADING")
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
        train_b, val_b, _t, _c = build_corpora(tok, dev)
        lm = HFLM(pretrained=model, tokenizer=tok, batch_size="auto")
        print("loaded %s | layers %d" % (MID, n_layers), flush=True)

        allocs = [("base", []), ("greedy@8", RANKINGS["greedy"][:8])]
        if ARM_SEL != "both":
            allocs = [a for a in allocs if a[0] == ARM_SEL]
            assert allocs, "ARM must be base | greedy@8 | both"
        results = {"model": MID, "rank": RANK, "alpha": ALPHA, "num_fewshot": FEWSHOT,
                   "arm_sel": ARM_SEL, "shard": [SH_I, SH_N],
                   "greedy_layers": RANKINGS["greedy"][:8], "allocations": {}}
        _dump(results)

        for tag, layers in allocs:
            _status(tag)
            if layers:
                set_all_prec(model, "ternary"); set_layers_active(model, [])
                joint_train_eval(model, per_layer_params, sorted(layers), train_b, val_b, val_b,
                                 SEED * 13 + len(layers))
                set_all_prec(model, "ternary"); set_layers_active(model, sorted(layers))
            else:
                set_all_prec(model, "ternary"); set_layers_active(model, [])
            te = time.time()
            rows, samples = eval_gsm8k_shard(lm)
            adh = measure_adherence(samples)
            results["allocations"][tag] = {
                "layers": sorted(layers),
                "strict_match": rows.get("exact_match,strict-match"),
                "flexible_extract": rows.get("exact_match,flexible-extract"),
                "adherence": adh,
                "eval_s": time.time() - te,
            }
            _dump(results)
            print("  [%s] strict=%.4f flex=%.4f adherence=%.4f (n=%d) (%.0fs)"
                  % (tag, rows.get("exact_match,strict-match") or float("nan"),
                     rows.get("exact_match,flexible-extract") or float("nan"),
                     adh["adherence_rate"] or float("nan"), adh["n"], time.time() - te), flush=True)

        
        A = results["allocations"]
        if not ("base" in A and "greedy@8" in A):
            results["wall_s"] = time.time() - t0
            _dump(results)
            _status("DONE %s %.0fs" % (ARM_SEL, time.time() - t0))
            print("\nDONE arm=%s %.1fs -> %s" % (ARM_SEL, time.time() - t0, OUT_JSON), flush=True)
            return
        b, g = A["base"], A["greedy@8"]
        d_strict = (g["strict_match"] or 0) - (b["strict_match"] or 0)
        d_flex = (g["flexible_extract"] or 0) - (b["flexible_extract"] or 0)
        d_adh = (g["adherence"]["adherence_rate"] or 0) - (b["adherence"]["adherence_rate"] or 0)
        results["verdict"] = {
            "delta_strict": d_strict, "delta_flexible": d_flex, "delta_adherence": d_adh,
            "adherence_explains_strict": abs(d_strict - d_adh) < abs(d_strict) * 0.5 + 0.01,
        }
        results["wall_s"] = time.time() - t0
        _dump(results)
        print("\n==== TEMPLATE-ADHERENCE VERDICT ====", flush=True)
        print("  Δstrict=%+.4f  Δflexible=%+.4f  Δadherence=%+.4f" % (d_strict, d_flex, d_adh),
              flush=True)
        print("  adherence explains the strict gain:", results["verdict"]["adherence_explains_strict"],
              flush=True)
        _status("DONE %.0fs" % (time.time() - t0))
        print("\nDONE %.1fs" % (time.time() - t0), flush=True)
    except Exception:
        tb = traceback.format_exc()
        print(tb, flush=True)
        _status("ERROR\n" + tb)
        raise

if __name__ == "__main__":
    main()
