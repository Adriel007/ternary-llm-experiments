
from __future__ import annotations

import json
import math
import os
import sys
import time
import traceback

ROOT = os.environ.get("PHD_ROOT", "/content/PhD-propose")
sys.path[:0] = [ROOT, os.path.join(ROOT, "packages/bitnet_core/src")]

import numpy as np  
import torch  
from torch import autocast  

from experiments.dynamics.experiment import (  
    build_model, train_one, _lr_at, _param_groups,
)
from experiments.dynamics.data import TokenLoader, build_tinystories_tokens  
from experiments.dynamics.soft_quant_h4 import (  
    CFG, N_HUTCH, DRIVE, set_soft_tau, eval_hard, _trace, _selftest,
)

SEEDS = (0, 1, 2)

SCHEDULES = [
    ("exp_fast",     "exp",      1.0, 0.02),   
    ("exp_slowhold", "hold_exp", 1.0, 0.02),   
    ("linear",       "linear",   1.0, 0.02),   
    ("cosine",       "cosine",   1.0, 0.02),   
    ("exp_lowmin",   "exp",      1.0, 0.005),  
    ("exp_highmin",  "exp",      1.0, 0.10),   
]

def tau_at(shape, frac, tau0, tau_min):
    frac = min(1.0, max(0.0, frac))
    if shape == "exp":
        return tau0 * (tau_min / tau0) ** frac
    if shape == "linear":
        return tau0 + (tau_min - tau0) * frac
    if shape == "cosine":
        return tau_min + 0.5 * (tau0 - tau_min) * (1.0 + math.cos(math.pi * frac))
    if shape == "hold_exp":
        if frac < 0.25:
            return tau0
        return tau0 * (tau_min / tau0) ** ((frac - 0.25) / 0.75)
    raise ValueError(shape)

def train_soft_sched(seed, cfg, loaders, shape, tau0, tau_min):
    train_loader, _ = loaders
    device = cfg.device
    model = build_model(True, seed, device)
    model.train()
    opt = torch.optim.AdamW(_param_groups(model, cfg.weight_decay), lr=cfg.lr, betas=cfg.betas)
    data_gen = torch.Generator().manual_seed(seed)
    hist = {"step": [], "train_loss": [], "tau": []}
    run_loss = None
    for step in range(cfg.steps):
        frac = step / max(1, cfg.steps - 1)
        tau = tau_at(shape, frac, tau0, tau_min)
        set_soft_tau(model, tau)
        lr = _lr_at(step, cfg)
        for pg in opt.param_groups:
            pg["lr"] = lr
        x = train_loader.batch(cfg.batch_size, data_gen)
        with autocast(device_type="cuda", dtype=cfg.amp_dtype, enabled=cfg.amp and device == "cuda"):
            loss = model(input_ids=x, labels=x)["loss"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        lv = loss.item(); run_loss = lv if run_loss is None else 0.9 * run_loss + 0.1 * lv
        if step % cfg.log_every == 0 or step == cfg.steps - 1:
            hist["step"].append(step); hist["train_loss"].append(run_loss); hist["tau"].append(tau)
    set_soft_tau(model, None)                              
    return model.eval(), hist

OUT_JSON = os.path.join(CFG.out_dir, "soft_quant_h4_tuned.json")
DRIVE_JSON = os.path.join(DRIVE, "soft_quant_h4_tuned.json")
STATUS = os.path.join(CFG.out_dir, "SOFTQ_TUNED_STATUS.txt")

def _save(obj):
    os.makedirs(CFG.out_dir, exist_ok=True)
    json.dump(obj, open(OUT_JSON, "w"), indent=2)
    if os.path.isdir(DRIVE):
        json.dump(obj, open(DRIVE_JSON, "w"), indent=2)

def _load():
    if os.path.exists(OUT_JSON):
        try:
            return json.load(open(OUT_JSON))
        except Exception:
            pass
    return {"refs": {}, "soft": {}}

def _shard_path(i):
    return OUT_JSON.replace(".json", ".shard%d.json" % i)

def _setup():
    _selftest()
    tr_ids, va_ids = build_tinystories_tokens(CFG.data_dir, CFG.n_train_tokens, CFG.n_val_tokens)
    loaders = (TokenLoader(tr_ids, CFG.seq_len, CFG.device),
               TokenLoader(va_ids, CFG.seq_len, CFG.device))
    xb = next(iter(loaders[1].iter_eval(CFG.hessian_bsz, 1)))
    return loaders, xb

def phase_refs(loaders, xb):
    state = _load()
    for seed in SEEDS:
        sk = str(seed)
        if sk in state["refs"]:
            continue
        open(STATUS, "w").write("REFS seed %d\n" % seed)
        rfp = train_one(False, seed, CFG, loaders); fp_val = float(rfp["final_val_loss"]); del rfp
        rste = train_one(True, seed, CFG, loaders); m_ste = rste["model"]
        ste_tr, ste_va = eval_hard(m_ste, loaders, CFG)
        ste_lam, ste_trace, ste_sd = _trace(m_ste, xb)
        del m_ste, rste; torch.cuda.empty_cache()
        state["refs"][sk] = {"fp_val": fp_val,
                             "ste": {"val": ste_va, "train": ste_tr, "tax": ste_va - fp_val,
                                     "gap": ste_va - ste_tr, "lambda_max": ste_lam,
                                     "trace": ste_trace, "trace_sd": ste_sd}}
        _save(state)
        print("[refs seed %d] FP val=%.4f | STE tax=%+.4f trace=%.0f gap=%+.4f"
              % (seed, fp_val, state["refs"][sk]["ste"]["tax"], ste_trace,
                 state["refs"][sk]["ste"]["gap"]), flush=True)
    print("REFS DONE: seeds %s" % sorted(state["refs"].keys()), flush=True)

def _flat_arms():
    return [(name, shape, t0, tm, seed)
            for (name, shape, t0, tm) in SCHEDULES for seed in SEEDS]

def phase_soft_shard(loaders, xb, shard_i, shard_n):
    refs = _load().get("refs", {})
    assert len(refs) == len(SEEDS), "refs incomplete (%s) -- run phase 'refs' first" % sorted(refs)
    arms = [a for k, a in enumerate(_flat_arms()) if k % shard_n == shard_i]
    out = {}
    if os.path.exists(_shard_path(shard_i)):
        try:
            out = json.load(open(_shard_path(shard_i)))
        except Exception:
            out = {}
    for name, shape, tau0, tau_min, seed in arms:
        sk = str(seed)
        if out.get(name, {}).get(sk):
            continue
        open(STATUS, "w").write("SHARD %d/%d SOFT %s seed %d\n" % (shard_i, shard_n, name, seed))
        t = time.time()
        fp_val = refs[sk]["fp_val"]
        m, shist = train_soft_sched(seed, CFG, loaders, shape, tau0, tau_min)
        s_tr, s_va = eval_hard(m, loaders, CFG)
        s_lam, s_trace, s_sd = _trace(m, xb)
        del m; torch.cuda.empty_cache()
        out.setdefault(name, {})[sk] = {"val": s_va, "train": s_tr, "tax": s_va - fp_val,
                                        "gap": s_va - s_tr, "lambda_max": s_lam,
                                        "trace": s_trace, "trace_sd": s_sd,
                                        "shape": shape, "tau0": tau0, "tau_min": tau_min, "hist": shist}
        json.dump(out, open(_shard_path(shard_i) + ".tmp", "w"), indent=2)
        os.replace(_shard_path(shard_i) + ".tmp", _shard_path(shard_i))
        print("[shard%d %s seed %d] tax=%+.4f trace=%.0f gap=%+.4f (%.0fs)"
              % (shard_i, name, seed, out[name][sk]["tax"], s_trace, out[name][sk]["gap"],
                 time.time() - t), flush=True)
    print("SHARD %d DONE (%d arms)" % (shard_i, len(arms)), flush=True)

def phase_merge(shard_n):
    state = _load()
    state.setdefault("soft", {})
    for i in range(shard_n):
        p = _shard_path(i)
        if not os.path.exists(p):
            continue
        sh = json.load(open(p))
        for name, byseed in sh.items():
            state["soft"].setdefault(name, {}).update(byseed)
    
    missing = []
    for name, *_ in SCHEDULES:
        for seed in SEEDS:
            if str(seed) not in state["soft"].get(name, {}):
                missing.append((name, seed))
    if missing:
        print("WARNING: incomplete soft results, missing %s" % missing, flush=True)

    def arr(d, key):
        return np.array([d[str(s)][key] for s in SEEDS if str(s) in d], float)
    ste_tax = float(np.mean([state["refs"][str(s)]["ste"]["tax"] for s in SEEDS]))
    ste_trace = float(np.mean([state["refs"][str(s)]["ste"]["trace"] for s in SEEDS]))
    summary = {"ste_tax": ste_tax, "ste_trace": ste_trace, "schedules": {}, "incomplete": missing}
    print("\n==== H4 TUNED (tau-schedule sweep) — STE: tax=%+.4f trace=%.0f ====" % (ste_tax, ste_trace))
    for name, *_ in SCHEDULES:
        d = state["soft"].get(name, {})
        if not d:
            continue
        tax = arr(d, "tax"); trace = arr(d, "trace")
        dtax = tax - np.array([state["refs"][str(s)]["ste"]["tax"] for s in SEEDS if str(s) in d])
        dtax_se = float(np.std(dtax, ddof=1) / math.sqrt(len(dtax))) if len(dtax) > 1 else float("nan")
        traceratio = float(np.mean(trace) / ste_trace)
        beats = bool(np.mean(dtax) < 0 and traceratio < 1.0)
        summary["schedules"][name] = {
            "tax_mean": float(np.mean(tax)), "tax_sd": float(np.std(tax, ddof=1)),
            "trace_mean": float(np.mean(trace)), "trace_over_ste": traceratio,
            "dtax_vs_ste_mean": float(np.mean(dtax)), "dtax_vs_ste_se": dtax_se,
            "beats_ste_both": beats}
        print("  %-12s tax=%+.4f (Δvs STE %+.4f ± %.4f) trace=%.0f (×STE %.2f)  beats-both=%s"
              % (name, np.mean(tax), np.mean(dtax), dtax_se, np.mean(trace), traceratio, beats))
    any_beat = [n for n in summary["schedules"] if summary["schedules"][n]["beats_ste_both"]]
    summary["any_schedule_beats_ste"] = any_beat
    state["summary"] = summary
    _save(state)
    verdict = ("RESCUED by " + ",".join(any_beat)) if any_beat else              "REFUTED across all %d schedules (naive negative is NOT a schedule artefact)" % len(SCHEDULES)
    print("\nH4 TUNED VERDICT: %s" % verdict, flush=True)
    open(STATUS, "w").write("DONE | %s\n" % verdict)

def main():
    os.makedirs(CFG.out_dir, exist_ok=True)
    argv = sys.argv[1:]
    mode = argv[0] if argv else "full"
    t0 = time.time()
    try:
        if mode == "merge":
            phase_merge(int(argv[1]) if len(argv) > 1 else 6)
        elif mode == "refs":
            loaders, xb = _setup(); phase_refs(loaders, xb)
        elif mode == "soft":
            i, n = (int(x) for x in argv[1].split("/"))   
            loaders, xb = _setup(); phase_soft_shard(loaders, xb, i, n)
        else:  
            loaders, xb = _setup(); phase_refs(loaders, xb)
            phase_soft_shard(loaders, xb, 0, 1); phase_merge(1)
        print("phase '%s' DONE %.0fs" % (mode, time.time() - t0), flush=True)
    except Exception:
        tb = traceback.format_exc(); print(tb, flush=True)
        open(STATUS, "w").write("ERROR\n" + tb)
        raise

if __name__ == "__main__":
    main()
