
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

from experiments.interp.eap_ig_2b import MID, OUT, DRIVE  
from experiments.interp.h2_2b_correction import (  
    SEQ_LEN, BATCH, N_TRAIN_SEQS, N_VAL_SEQS, RANK, ALPHA, MAX_STEPS as _MAXSTEPS,
    install_unified_forward, attach_adapters, set_layers_active, set_all_prec, ppl,
)
from experiments.interp.h2_2b_greedy import (  
    SEED, build_corpora, joint_train, eval_mean_loss, per_seq_losses, _point,
)
from experiments.env_stamp import env_stamp  

TRAIN_DOMAIN = (sys.argv[1] if len(sys.argv) > 1 else "fineweb").strip().lower()
assert TRAIN_DOMAIN in ("fineweb", "mix"), "TRAIN_DOMAIN must be 'fineweb' or 'mix'"

N_TEST_SEQS = 512                 
SKIP_FW = 96                      
REPORT_BUDGETS = [2, 5, 8]
PRIMARY_B = 5                     
SEEDS_FINAL = [0, 1, 2]
N_RANDOM_DRAWS = 3
FINEWEB = ("HuggingFaceFW/fineweb-edu", "sample-10BT")
BASE_PPL_TOL = 0.20               

FW_BASE_REF = 21.522909983057094
FW_BASE_TOL = 0.05

RUN2 = os.path.join(ROOT, "reports/data/h2_2b_correction.json")
GREEDY = os.path.join(ROOT, "reports/data/h2_2b_greedy.json")
TDIR = os.path.join(DRIVE, "h2_2b_traindomain_%s_checkpoints" % TRAIN_DOMAIN)
TCKPT = os.path.join(TDIR, "h2_2b_traindomain_%s_state.json" % TRAIN_DOMAIN)
TLOCAL = os.path.join(OUT, "h2_2b_traindomain_%s.json" % TRAIN_DOMAIN)

CFG = {
    "model": MID, "seq_len": SEQ_LEN, "rank": RANK, "alpha": ALPHA, "batch": BATCH,
    "n_train_seqs": N_TRAIN_SEQS, "n_val_seqs": N_VAL_SEQS, "n_test_seqs": N_TEST_SEQS,
    "skip_fw": SKIP_FW, "report_budgets": REPORT_BUDGETS, "primary_b": PRIMARY_B,
    "seeds_final": SEEDS_FINAL, "n_random_draws": N_RANDOM_DRAWS,
    "fineweb": list(FINEWEB), "seed": SEED, "train_domain": TRAIN_DOMAIN,
    "purpose": "train-domain control: swap the adapter TRAINING corpus (fineweb or wiki+fineweb mix) "
               "while keeping the cross-domain test slices fixed; reciprocal of the cross-domain run, "
               "separates domain-adaptation from quantization-correction without a second model",
}

def _atomic_dump(obj, path):
    tmp = path + ".tmp"
    json.dump(obj, open(tmp, "w"), indent=2)
    os.replace(tmp, path)

def _load_state():
    for p in (TCKPT, TLOCAL):
        if os.path.exists(p):
            try:
                return json.load(open(p))
            except Exception:
                pass
    return {}

def _save_state(state):
    os.makedirs(TDIR, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)
    _atomic_dump(state, TCKPT)
    _atomic_dump(state, TLOCAL)

def _fineweb_pool(tok, device):
    from datasets import load_dataset
    name, cfg = FINEWEB
    ds = load_dataset(name, cfg, split="train", streaming=True)
    eos = tok.eos_token_id
    sizes = [N_TEST_SEQS, SKIP_FW, N_TRAIN_SEQS, N_VAL_SEQS]
    need = sum(sizes) * SEQ_LEN
    buf = []
    for ex in ds:
        t = ex["text"].strip()
        if not t:
            continue
        buf.extend(tok(t, add_special_tokens=False).input_ids)
        buf.append(eos)
        if len(buf) >= need:
            break
    pool = torch.tensor(buf[:need], dtype=torch.long).view(sum(sizes), SEQ_LEN)
    off = np.cumsum([0] + sizes)
    fw_test, _, fw_train, fw_val = (pool[off[i]:off[i + 1]] for i in range(4))
    
    seen = {}
    for nm, sl in (("test", fw_test), ("train", fw_train), ("val", fw_val)):
        for row in sl.tolist():
            key = tuple(row)
            assert key not in seen, f"fineweb leakage: sequence in both {seen.get(key)} and {nm}"
            seen[key] = nm
    return fw_test.to(device), fw_train.to(device), fw_val.to(device)

def _build_train(wiki_train_b, val_wiki, fw_train, fw_val, device):
    fw_train_b = [fw_train[i:i + BATCH] for i in range(0, fw_train.shape[0], BATCH)]
    if TRAIN_DOMAIN == "fineweb":
        return fw_train_b, fw_val
    
    half = len(wiki_train_b) // 2                       
    train_b = []
    for w, f in zip(wiki_train_b[:half], fw_train_b[:half]):
        train_b.append(w)
        train_b.append(f)
    nv = val_wiki.shape[0] // 2
    val_ids = torch.cat([val_wiki[:nv], fw_val[:nv]], dim=0).to(device)
    return train_b, val_ids

def measure_cross(model, plp, layers, train_b, val_ids, test_wiki, test_fw, seeds):
    pw, pf = [], []
    for s in seeds:
        joint_train(model, plp, layers, train_b, val_ids, _MAXSTEPS, SEED * 31 + s)
        pw.append(_point(per_seq_losses(model, test_wiki)))
        pf.append(_point(per_seq_losses(model, test_fw)))
        set_layers_active(model, [])

    def agg(pts):
        ppls = [p["ppl"] for p in pts]
        return {"ppl_mean": float(np.mean(ppls)),
                "ppl_sd": float(np.std(ppls, ddof=1)) if len(ppls) > 1 else 0.0,
                "ppl_seeds": [float(x) for x in ppls]}
    return {"layers": sorted(layers), "wikitext": agg(pw), "fineweb": agg(pf)}

def _selftest(model, test_wiki, test_fw, ref_ppl):
    print("[selftest] running...")
    set_all_prec(model, "ternary"); set_layers_active(model, [])
    base_w = ppl(eval_mean_loss(model, test_wiki))
    base_f = ppl(eval_mean_loss(model, test_fw))
    dw = abs(base_w - ref_ppl)
    df = abs(base_f - FW_BASE_REF)
    assert dw < BASE_PPL_TOL, ("forward-faithfulness gate FAILED: wiki base ppl %.4f vs greedy.json "
                               "base %.4f (|Δ|=%.3f)" % (base_w, ref_ppl, dw))
    assert df < FW_BASE_TOL, ("continuity gate FAILED: fineweb base ppl %.4f vs cross-domain %.4f "
                              "(|Δ|=%.3f) -> test slice diverged" % (base_f, FW_BASE_REF, df))
    print("[selftest] wiki base ppl=%.4f (ref %.4f, Δ%.3f) | fineweb base ppl=%.4f (xdomain %.4f, Δ%.3f)"
          " | PASS" % (base_w, ref_ppl, dw, base_f, FW_BASE_REF, df))
    return base_w, base_f

def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    status = os.path.join(OUT, "H2_2B_TRAINDOMAIN_%s_STATUS.txt" % TRAIN_DOMAIN.upper())
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
        print("loaded", MID, "| layers", n_layers, "| train_domain", TRAIN_DOMAIN,
              "| extra/layer=%.4f" % extra)

        wiki_train_b, val_wiki, test_wiki, _score = build_corpora(tok, dev)   
        open(status, "w").write("FINEWEB\n")
        test_fw, fw_train, fw_val = _fineweb_pool(tok, dev)
        train_b, val_ids = _build_train(wiki_train_b, val_wiki, fw_train, fw_val, dev)
        print("corpora: train %d-seq (%s) | val %d | wiki-test %d | fineweb-test %d" %
              (len(train_b) * BATCH, TRAIN_DOMAIN, val_ids.shape[0], test_wiki.shape[0], test_fw.shape[0]))

        run2 = json.load(open(RUN2)); greedy = json.load(open(GREEDY))
        c_val = np.array([float(run2["c_l"][str(i)]) for i in range(n_layers)])
        s_trace = np.array([float(run2["s_l"][str(i)]["trace"]) for i in range(n_layers)])
        greedy_order = [d["added"] for d in greedy["greedy_path"]]
        c_order = [int(x) for x in np.argsort(-c_val)]
        s_order = [int(x) for x in np.argsort(-s_trace)]
        rankings = {"greedy": greedy_order, "c_causal": c_order, "s_hawq": s_order}

        open(status, "w").write("SELFTEST\n")
        base_w, base_f = _selftest(model, test_wiki, test_fw, float(greedy["base"]["ppl"]))

        state = _load_state()
        if "config" in state and {k: state["config"].get(k) for k in CFG} != CFG:
            raise RuntimeError("checkpoint config mismatch")
        state["config"] = dict(CFG); state["extra_mem_frac_per_layer"] = extra
        state["env"] = env_stamp()
        state.setdefault("base", {"wikitext_ppl": base_w, "fineweb_ppl": base_f})
        state.setdefault("fronts", {})
        state.setdefault("random", {})
        _save_state(state)

        open(status, "w").write("FRONTS\n")
        for name, order in rankings.items():
            for B in REPORT_BUDGETS:
                key = "%s@%d" % (name, B)
                if key in state["fronts"]:
                    continue
                t = time.time()
                layers = sorted(order[:B])
                state["fronts"][key] = measure_cross(model, plp, layers, train_b, val_ids,
                                                     test_wiki, test_fw, SEEDS_FINAL)
                _save_state(state)
                r = state["fronts"][key]
                print("  [%s] %s -> wiki %.3f±%.3f | fineweb %.3f±%.3f (%.0fs)" %
                      (key, layers, r["wikitext"]["ppl_mean"], r["wikitext"]["ppl_sd"],
                       r["fineweb"]["ppl_mean"], r["fineweb"]["ppl_sd"], time.time() - t))

        open(status, "w").write("RANDOM\n")
        rng = np.random.default_rng(1234)
        for d in range(N_RANDOM_DRAWS):
            key = "draw%d@%d" % (d, PRIMARY_B)
            if key in state["random"]:
                continue
            layers = sorted(int(x) for x in rng.choice(n_layers, PRIMARY_B, replace=False))
            t = time.time()
            state["random"][key] = measure_cross(model, plp, layers, train_b, val_ids,
                                                 test_wiki, test_fw, [SEEDS_FINAL[0]])
            _save_state(state)
            r = state["random"][key]
            print("  [%s] %s -> wiki %.3f | fineweb %.3f (%.0fs)" %
                  (key, layers, r["wikitext"]["ppl_mean"], r["fineweb"]["ppl_mean"], time.time() - t))

        open(status, "w").write("VERDICT\n")
        def ppl_at(name, B, corpus):
            return state["fronts"]["%s@%d" % (name, B)][corpus]["ppl_mean"]
        rand_med = {c: float(np.median([state["random"]["draw%d@%d" % (d, PRIMARY_B)][c]["ppl_mean"]
                                        for d in range(N_RANDOM_DRAWS)])) for c in ("wikitext", "fineweb")}
        verdict = {"primary_b": PRIMARY_B, "train_domain": TRAIN_DOMAIN, "random_median@primary": rand_med}
        for corpus in ("wikitext", "fineweb"):
            bppl = state["base"][corpus + "_ppl"]
            g8 = ppl_at("greedy", 8, corpus)
            verdict[corpus] = {
                "base_ppl": bppl,
                "greedy8_ppl": g8,
                "frac_recovered_greedy8": (bppl - g8) / bppl,
                "order_greedy_lt_c@primary": ppl_at("greedy", PRIMARY_B, corpus) < ppl_at("c_causal", PRIMARY_B, corpus),
                "order_c_lt_hawq@primary": ppl_at("c_causal", PRIMARY_B, corpus) < ppl_at("s_hawq", PRIMARY_B, corpus),
                "order_c_lt_randmed@primary": ppl_at("c_causal", PRIMARY_B, corpus) < rand_med[corpus],
            }
        state["verdict"] = verdict; state["wall_s"] = time.time() - t0
        _save_state(state)

        print("\n=== TRAIN-DOMAIN=%s (test ppl; primary B=%d) ===" % (TRAIN_DOMAIN, PRIMARY_B))
        print("  ranking@B        wikitext        fineweb-edu")
        for name in ("greedy", "c_causal", "s_hawq"):
            for B in REPORT_BUDGETS:
                r = state["fronts"]["%s@%d" % (name, B)]
                print("  %-12s   %6.3f±%.3f   %6.3f±%.3f" %
                      ("%s@%d" % (name, B), r["wikitext"]["ppl_mean"], r["wikitext"]["ppl_sd"],
                       r["fineweb"]["ppl_mean"], r["fineweb"]["ppl_sd"]))
        print("  %-12s   %6.3f          %6.3f          (no correction)" %
              ("base", state["base"]["wikitext_ppl"], state["base"]["fineweb_ppl"]))
        print("  %-12s   %6.3f          %6.3f          (median of %d draws)" %
              ("random@%d" % PRIMARY_B, rand_med["wikitext"], rand_med["fineweb"], N_RANDOM_DRAWS))
        for c in ("wikitext", "fineweb"):
            v = verdict[c]
            print("Q1 [%s] base->greedy@8 recovers %.1f%% of base ppl (%.2f->%.2f)"
                  % (c, 100 * v["frac_recovered_greedy8"], v["base_ppl"], v["greedy8_ppl"]))
            print("Q2 [%s] @B=%d: greedy<c %s | c<HAWQ %s | c<rand-med %s"
                  % (c, PRIMARY_B, v["order_greedy_lt_c@primary"], v["order_c_lt_hawq@primary"],
                     v["order_c_lt_randmed@primary"]))
        open(status, "w").write("DONE %.0fs\n" % (time.time() - t0))
        print("DONE %.1fs" % (time.time() - t0))
    except Exception:
        tb = traceback.format_exc()
        print(tb); open(status, "w").write("ERROR\n" + tb)
        raise

if __name__ == "__main__":
    main()
