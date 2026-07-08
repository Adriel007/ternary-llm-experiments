
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
    MID, OUT, install_unified_forward, attach_adapters, set_layers_active, set_all_prec,
    joint_train_eval, eval_loss, ppl, _node_modules, _out0,
    RANK, ALPHA, SEED, BATCH, SEQ_LEN, N_TRAIN_SEQS, N_VAL_SEQS, N_TEST_SEQS, N_CALIB_SEQS,
)

RANKINGS = {
    "c_causal": [0, 1, 2, 29, 19, 5, 4, 28, 27, 9, 25, 3, 20, 26, 13, 8, 24, 10, 23, 14,
                 21, 7, 15, 12, 22, 16, 18, 11, 6, 17],
    "s_hawq":   [1, 0, 3, 5, 9, 6, 2, 4, 13, 8, 15, 7, 11, 19, 14, 21, 12, 24, 10, 25,
                 26, 16, 18, 17, 20, 27, 22, 23, 28, 29],
    "random":   [2, 11, 26, 21, 10, 4, 28, 16, 23, 6, 18, 25, 3, 29, 8, 0, 19, 12, 20, 13,
                 7, 5, 17, 14, 22, 9, 27, 24, 1, 15],
}
C6_BUDGETS = [2, 5, 8]
SEQ_LONG = 1024
N_TEST_LONG = 48               
N_BOOT = 500                   

OUT_JSON = os.path.join(OUT, "h2_2b_robust.json")
STATUS = os.path.join(OUT, "H2_2B_ROBUST_STATUS.txt")

def _dump(obj):
    os.makedirs(OUT, exist_ok=True)
    tmp = OUT_JSON + ".tmp"
    json.dump(obj, open(tmp, "w"), indent=2)
    os.replace(tmp, OUT_JSON)

def _status(s):
    os.makedirs(OUT, exist_ok=True)
    open(STATUS, "w").write(s + "\n")

def build_robust_corpora(tok, device):
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train", streaming=True)
    eos = tok.eos_token_id
    n128 = N_TRAIN_SEQS + N_VAL_SEQS + N_TEST_SEQS + N_CALIB_SEQS
    need = n128 * SEQ_LEN + N_TEST_LONG * SEQ_LONG
    buf = []
    for ex in ds:
        t = ex["text"].strip()
        if not t:
            continue
        buf.extend(tok(t, add_special_tokens=False).input_ids)
        buf.append(eos)
        if len(buf) >= need:
            break
    buf = buf[:need]
    p128 = torch.tensor(buf[:n128 * SEQ_LEN], dtype=torch.long).view(n128, SEQ_LEN)
    p1024 = torch.tensor(buf[n128 * SEQ_LEN:], dtype=torch.long).view(N_TEST_LONG, SEQ_LONG)
    sizes = [N_TRAIN_SEQS, N_VAL_SEQS, N_TEST_SEQS, N_CALIB_SEQS]
    off = np.cumsum([0] + sizes)
    sl = [p128[off[i]:off[i + 1]] for i in range(4)]
    mk = lambda ids, bs: [ids[i:i + bs].to(device) for i in range(0, ids.shape[0], bs)]
    train, val, test, calib = [mk(s, BATCH) for s in sl]
    test_long = mk(p1024, 1)   
    return train, val, test, calib, test_long

@torch.no_grad()
def perseq_ablation_deltas(model, calib_seqs, n_layers):
    set_all_prec(model, "ternary")
    set_layers_active(model, [])
    nodes = dict(_node_modules(model))

    def mk_hook():
        def hook(m, i, o):
            o0 = _out0(o)
            mean = o0.float().mean(dim=(0, 1), keepdim=True).to(o0.dtype)
            rep = mean.expand_as(o0).clone()
            return (rep,) + tuple(o[1:]) if isinstance(o, tuple) else rep
        return hook

    base = np.array([float(model(input_ids=b, labels=b).loss.item()) for b in calib_seqs])
    deltas = np.zeros((n_layers, len(calib_seqs)), dtype=np.float64)
    for l in range(n_layers):
        h = [nodes[f"a{l}"].register_forward_hook(mk_hook()),
             nodes[f"m{l}"].register_forward_hook(mk_hook())]
        for s, b in enumerate(calib_seqs):
            deltas[l, s] = float(model(input_ids=b, labels=b).loss.item()) - base[s]
        for hh in h:
            hh.remove()
    return deltas

def _spearman(a, b):
    from scipy.stats import spearmanr
    return float(spearmanr(a, b).correlation)

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
        install_unified_forward()
        per_layer_params, extra_frac = attach_adapters(model, RANK, ALPHA, dev, SEED)
        
        train_b, val_b, test128_b, calib_b4, test1024_b = build_robust_corpora(tok, dev)
        calib_seqs = [b[i:i + 1] for b in calib_b4 for i in range(b.shape[0])]
        set_all_prec(model, "ternary"); set_layers_active(model, [])
        base128 = eval_loss(model, test128_b)
        base1024 = eval_loss(model, test1024_b)
        print("loaded %s | layers %d | base ppl: 128=%.3f 1024=%.3f | test1024 seqs=%d"
              % (MID, n_layers, ppl(base128), ppl(base1024), len(test1024_b)), flush=True)

        results = {"model": MID, "rank": RANK, "alpha": ALPHA, "seed": SEED,
                   "base_ppl_128": ppl(base128), "base_ppl_1024": ppl(base1024),
                   "n_test1024_seqs": len(test1024_b), "c6_budgets": C6_BUDGETS,
                   "c6": {}, "c7": {}}
        _dump(results)

        _status("C6")
        for name, order in RANKINGS.items():
            results["c6"].setdefault(name, {})
            for B in C6_BUDGETS:
                t = time.time()
                layers = sorted(order[:B])
                loss128 = joint_train_eval(model, per_layer_params, layers, train_b, val_b,
                                           test128_b, SEED * 7 + B)
                
                set_all_prec(model, "ternary"); set_layers_active(model, layers)
                loss1024 = eval_loss(model, test1024_b)
                set_layers_active(model, [])
                results["c6"][name][str(B)] = {
                    "layers": layers, "ppl_128": ppl(loss128), "ppl_1024": ppl(loss1024),
                    "extra_mem_frac": B / n_layers * extra_frac}
                _dump(results)
                print("  [C6] %-9s B=%d -> ppl128=%.3f ppl1024=%.3f (%.0fs)"
                      % (name, B, ppl(loss128), ppl(loss1024), time.time() - t), flush=True)

        order_pres = {}
        for B in C6_BUDGETS:
            o128 = sorted(RANKINGS, key=lambda nm: results["c6"][nm][str(B)]["ppl_128"])
            o1024 = sorted(RANKINGS, key=lambda nm: results["c6"][nm][str(B)]["ppl_1024"])
            order_pres[str(B)] = {"order_128": o128, "order_1024": o1024, "preserved": o128 == o1024}
        results["c6_order_preserved"] = order_pres

        _status("C7")
        deltas = perseq_ablation_deltas(model, calib_seqs, n_layers)   
        full_c = deltas.mean(axis=1)
        full_rank = list(np.argsort(full_c)[::-1])
        ref_rank_vec = full_c                                    
        rng = np.random.default_rng(SEED)
        n_seqs = deltas.shape[1]
        rhos, top5_jac, top8_jac = [], [], []
        full_top5, full_top8 = set(full_rank[:5]), set(full_rank[:8])
        for _ in range(N_BOOT):
            idx = rng.integers(0, n_seqs, n_seqs)
            c = deltas[:, idx].mean(axis=1)
            rhos.append(_spearman(ref_rank_vec, c))
            r = list(np.argsort(c)[::-1])
            top5_jac.append(len(full_top5 & set(r[:5])) / len(full_top5 | set(r[:5])))
            top8_jac.append(len(full_top8 & set(r[:8])) / len(full_top8 | set(r[:8])))
        rhos = np.array(rhos)
        results["c7"] = {
            "n_calib_seqs": n_seqs, "n_boot": N_BOOT,
            "full_calib_top8": [int(x) for x in full_rank[:8]],
            "spearman_median": float(np.median(rhos)),
            "spearman_p05": float(np.percentile(rhos, 5)),
            "spearman_p95": float(np.percentile(rhos, 95)),
            "spearman_min": float(rhos.min()),
            "top5_jaccard_median": float(np.median(top5_jac)),
            "top8_jaccard_median": float(np.median(top8_jac)),
            "note": "per-sequence mean-ablation approximation; bootstrap assesses ranking stability",
        }
        results["wall_s"] = time.time() - t0
        _dump(results)

        print("\n==== C6 FRONT ORDER (128 vs 1024) ====", flush=True)
        for B in C6_BUDGETS:
            op = order_pres[str(B)]
            print("  B=%d  128:%s  1024:%s  preserved=%s"
                  % (B, op["order_128"], op["order_1024"], op["preserved"]), flush=True)
        print("\n==== C7 c_ℓ BOOTSTRAP STABILITY ====", flush=True)
        c7 = results["c7"]
        print("  Spearman vs full-calib: median=%.3f  [p05=%.3f, p95=%.3f]  min=%.3f"
              % (c7["spearman_median"], c7["spearman_p05"], c7["spearman_p95"], c7["spearman_min"]),
              flush=True)
        print("  top5 Jaccard median=%.3f | top8 Jaccard median=%.3f"
              % (c7["top5_jaccard_median"], c7["top8_jaccard_median"]), flush=True)
        _status("DONE %.0fs" % (time.time() - t0))
        print("\nDONE %.1fs" % (time.time() - t0), flush=True)
    except Exception:
        tb = traceback.format_exc()
        print(tb, flush=True)
        _status("ERROR\n" + tb)
        raise

if __name__ == "__main__":
    main()
