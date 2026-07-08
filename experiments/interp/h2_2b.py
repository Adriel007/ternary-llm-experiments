
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
import torch.nn.functional as TF  
from torch.nn.attention import SDPBackend, sdpa_kernel  

from experiments.interp.eap_ig_2b import MID, OUT, DRIVE, _node_modules, _out0  

SEQ_LEN = 128
BATCH = 4
N_CALIB_BATCHES = 8        
N_HELDOUT_BATCHES = 16     
HESSIAN_BSZ = 2            
N_HUTCH = 8                
INT8_LEVELS = 127          
BUDGETS = [1, 2, 3, 5, 8, 11, 15]   
TERNARY_BITS, INT8_BITS = 1.58, 8.0
SEED = 0

CKPT_DIR = os.path.join(DRIVE, "h2_2b_checkpoints")
CKPT_PATH = os.path.join(CKPT_DIR, "h2_2b_state.json")
LOCAL_PATH = os.path.join(OUT, "h2_2b.json")

CFG = {
    "model": MID, "seq_len": SEQ_LEN, "batch": BATCH,
    "n_calib_batches": N_CALIB_BATCHES, "n_heldout_batches": N_HELDOUT_BATCHES,
    "hessian_bsz": HESSIAN_BSZ, "n_hutch": N_HUTCH, "int8_levels": INT8_LEVELS,
    "budgets": BUDGETS, "seed": SEED,
    "calib_corpus": "Salesforce/wikitext wikitext-103-raw-v1 (streaming, train split)",
    "upgrade": "ternary(1.58b) -> int8 re-quant over same absmean range [-g,g]",
    "c_l": "mean-ablation of decoder-layer residual write on LM loss (task-general, calib)",
    "s_l": "HAWQ Hessian trace (Hutchinson) on the all-smooth twin (calib)",
}

def _atomic_dump(obj, path):
    tmp = path + ".tmp"
    json.dump(obj, open(tmp, "w"), indent=2)
    os.replace(tmp, path)

def _load_state():
    for p in (CKPT_PATH, LOCAL_PATH):
        if os.path.exists(p):
            try:
                return json.load(open(p))
            except Exception:
                pass
    return {}

def _save_state(state):
    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)
    _atomic_dump(state, CKPT_PATH)
    _atomic_dump(state, LOCAL_PATH)

def _bitlinears(layer):
    from transformers.integrations.bitnet import AutoBitLinear
    return [m for m in layer.modules() if isinstance(m, AutoBitLinear)]

def install_precision_modes():
    from transformers.integrations.bitnet import AutoBitLinear, WeightQuant, ActQuant

    def patched_forward(self, input):
        if self.rms_norm is not None:
            input = self.rms_norm(input)
        prec = getattr(self, "_prec", "ternary")
        if prec == "smooth":

            w = self.weight.float()
            scale = 1.0 / w.abs().mean().clamp_(min=1e-5)
            weight = ((w * scale).clamp(-1, 1) / scale).to(self.weight.dtype)
            return TF.linear(input, weight, self.bias)
        if prec == "int8":
            
            w = self.weight.float()
            scale = 1.0 / w.abs().mean().clamp_(min=1e-5)
            q = (w * scale * INT8_LEVELS).round().clamp(-INT8_LEVELS, INT8_LEVELS)
            weight = (q / INT8_LEVELS / scale).to(self.weight.dtype)
        else:  
            weight = WeightQuant.apply(self.weight)
        input = ActQuant.apply(input)
        return TF.linear(input, weight, self.bias)

    AutoBitLinear.forward = patched_forward

def set_prec(model, idxs, mode, base="ternary"):
    sel = set(int(i) for i in idxs)
    for i, layer in enumerate(model.model.layers):
        m = mode if i in sel else base
        for bl in _bitlinears(layer):
            bl._prec = m

def set_all_prec(model, mode):
    for layer in model.model.layers:
        for bl in _bitlinears(layer):
            bl._prec = mode

@torch.no_grad()
def selftest_harness(model, held_batches, n_layers):
    probe = held_batches[0][:1]
    lg_pristine = model(input_ids=probe).logits.clone()
    install_precision_modes()
    set_all_prec(model, "ternary")
    lg_tern = model(input_ids=probe).logits
    identical_ternary = torch.equal(lg_pristine, lg_tern)
    loss_tern = eval_loss(model, held_batches)
    set_all_prec(model, "int8")
    lg_int8 = model(input_ids=probe).logits
    loss_int8 = eval_loss(model, held_batches)
    set_all_prec(model, "ternary")
    return {
        "identical_ternary": bool(identical_ternary),
        "int8_engages": bool((lg_int8 - lg_tern).abs().max().item() > 0.0),
        "loss_ternary": loss_tern, "ppl_ternary": ppl(loss_tern),
        "loss_allint8": loss_int8, "ppl_allint8": ppl(loss_int8),
        "int8_improves": bool(loss_int8 <= loss_tern + 1e-6),
    }

def _token_pool(tok, n_seqs, seq_len):
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train", streaming=True)
    eos = tok.eos_token_id
    need = n_seqs * seq_len
    buf = []
    for ex in ds:
        t = ex["text"].strip()
        if not t:
            continue
        buf.extend(tok(t, add_special_tokens=False).input_ids)
        buf.append(eos)
        if len(buf) >= need:
            break
    return torch.tensor(buf[:need], dtype=torch.long).view(n_seqs, seq_len)

def _batches(ids, batch, device):
    return [ids[i:i + batch].to(device) for i in range(0, ids.shape[0], batch)]

def build_corpora(tok, device):
    n_calib = N_CALIB_BATCHES * BATCH
    n_held = N_HELDOUT_BATCHES * BATCH
    pool = _token_pool(tok, n_calib + n_held, SEQ_LEN)
    calib_ids, held_ids = pool[:n_calib], pool[n_calib:]
    overlap = set(map(tuple, calib_ids.tolist())) & set(map(tuple, held_ids.tolist()))
    assert not overlap, f"calib/held-out leakage: {len(overlap)} shared sequences"
    return _batches(calib_ids, BATCH, device), _batches(held_ids, BATCH, device)

@torch.no_grad()
def eval_loss(model, batches):
    losses = [float(model(input_ids=b, labels=b).loss.item()) for b in batches]
    return float(np.mean(losses))

def ppl(loss):
    return math.exp(loss)

def eff_bits(n_int8, n_layers):
    return (n_int8 * INT8_BITS + (n_layers - n_int8) * TERNARY_BITS) / n_layers

def _hessian_trace_one_layer(grads_l, params_l, n_hutch, seed_l):
    g = torch.Generator(device="cpu").manual_seed(seed_l)
    ests = []
    for _ in range(n_hutch):
        v = [(torch.randint(0, 2, p.shape, generator=g).to(p) * 2 - 1) for p in params_l]  
        dot = sum((gr * vv).sum() for gr, vv in zip(grads_l, v))
        Hv = torch.autograd.grad(dot, params_l, retain_graph=True)
        ests.append(float(sum((hv * vv).sum() for hv, vv in zip(Hv, v)).item()))
    return float(np.mean(ests)), float(np.std(ests) / np.sqrt(len(ests)))

def compute_s_l(model, calib_batches, n_layers, state):
    if "s_l" not in state:
        state["s_l"] = {}
    done = set(int(k) for k in state["s_l"])
    if len(done) == n_layers:
        print("  [s_l] already complete (%d/%d) -- skipping" % (len(done), n_layers))
        return
    set_all_prec(model, "smooth")   
    groups = {i: [m.weight for m in _bitlinears(layer)] for i, layer in enumerate(model.model.layers)}
    all_params = [p for i in range(n_layers) for p in groups[i]]
    for p in all_params:
        p.requires_grad_(True)
    xb = calib_batches[0][:HESSIAN_BSZ]
    t0 = time.time()
    with sdpa_kernel(SDPBackend.MATH):   
        loss = model(input_ids=xb, labels=xb).loss
        grads = torch.autograd.grad(loss, all_params, create_graph=True, retain_graph=True)
        idx, grad_groups = 0, {}
        for i in range(n_layers):
            grad_groups[i] = grads[idx:idx + len(groups[i])]
            idx += len(groups[i])
        print("  [s_l] shared backward: smooth-twin loss=%.4f (%.1fs) | resuming at %d/%d" %
              (float(loss.item()), time.time() - t0, len(done), n_layers))
        for l in range(n_layers):
            if l in done:
                continue
            tl = time.time()
            mean_est, sem_est = _hessian_trace_one_layer(grad_groups[l], groups[l], N_HUTCH, SEED * 1000 + l)
            
            if not state["s_l"] and abs(mean_est) < 1e-9:
                raise RuntimeError("smooth-twin Hessian still degenerate (trace~0 on layer %d) -- "
                                   "ActQuant/STE not fully removed; aborting" % l)
            state["s_l"][str(l)] = {"trace": mean_est, "sem": sem_est}
            _save_state(state)
            print("  [s_l] layer %2d: trace = %+.4g +/- %.2g  (%.1fs)" % (l, mean_est, sem_est, time.time() - tl))
    for p in all_params:
        p.requires_grad_(False)
    del loss, grads, grad_groups
    set_all_prec(model, "ternary")
    torch.cuda.empty_cache()

@torch.no_grad()
def compute_c_l(model, calib_batches, n_layers, base_loss_ternary_calib, state):
    if "c_l" not in state:
        state["c_l"] = {}
    done = set(int(k) for k in state["c_l"])
    if len(done) == n_layers:
        print("  [c_l] already complete (%d/%d) -- skipping" % (len(done), n_layers))
        return
    set_all_prec(model, "ternary")   
    nodes = dict(_node_modules(model))
    for l in range(n_layers):
        if l in done:
            continue
        t0 = time.time()

        def mk_hook():
            def hook(m, i, o):
                o0 = _out0(o)
                mean = o0.float().mean(dim=(0, 1), keepdim=True).to(o0.dtype)
                rep = mean.expand_as(o0).clone()
                return (rep,) + tuple(o[1:]) if isinstance(o, tuple) else rep
            return hook

        handles = [nodes[f"a{l}"].register_forward_hook(mk_hook()),
                   nodes[f"m{l}"].register_forward_hook(mk_hook())]
        abl = eval_loss(model, calib_batches)
        for h in handles:
            h.remove()
        state["c_l"][str(l)] = float(abl - base_loss_ternary_calib)
        _save_state(state)
        print("  [c_l] layer %2d: Delta-loss = %+.4f  (%.1fs)" % (l, state["c_l"][str(l)], time.time() - t0))

@torch.no_grad()
def compute_oracle(model, held_batches, n_layers, base_loss_ternary_held, state):
    if "oracle_dloss" not in state:
        state["oracle_dloss"] = {}
    done = set(int(k) for k in state["oracle_dloss"])
    if len(done) == n_layers:
        print("  [oracle] already complete (%d/%d) -- skipping" % (len(done), n_layers))
        return
    for l in range(n_layers):
        if l in done:
            continue
        t0 = time.time()
        set_prec(model, [l], "int8")
        loss_l = eval_loss(model, held_batches)
        state["oracle_dloss"][str(l)] = float(base_loss_ternary_held - loss_l)   
        _save_state(state)
        print("  [oracle] layer %2d -> int8: Delta-loss = %+.4f  (%.1fs)" %
              (l, state["oracle_dloss"][str(l)], time.time() - t0))
    set_all_prec(model, "ternary")

def build_rankings(s_l, c_l, oracle_dloss, n_layers, seed):
    rng = np.random.default_rng(seed)
    return {
        "s_hawq":   sorted(range(n_layers), key=lambda l: s_l[l], reverse=True),       
        "c_causal": sorted(range(n_layers), key=lambda l: c_l[l], reverse=True),       
        "oracle":   sorted(range(n_layers), key=lambda l: oracle_dloss[l], reverse=True),
        "random":   [int(x) for x in rng.permutation(n_layers)],
    }

@torch.no_grad()
def compute_pareto(model, held_batches, rankings, budgets, n_layers, state):
    if "pareto" not in state:
        state["pareto"] = {}
    for name, order in rankings.items():
        state["pareto"].setdefault(name, {})
        for B in budgets:
            key = str(B)
            if key in state["pareto"][name]:
                continue
            t0 = time.time()
            set_prec(model, order[:B], "int8")
            loss_b = eval_loss(model, held_batches)
            state["pareto"][name][key] = {"loss": float(loss_b), "ppl": ppl(loss_b),
                                          "eff_bits": eff_bits(B, n_layers)}
            _save_state(state)
            print("  [pareto] %-9s B=%2d (eff %.2fb) layers %s -> ppl=%.3f  (%.1fs)" %
                  (name, B, eff_bits(B, n_layers), sorted(order[:B]), ppl(loss_b), time.time() - t0))
    set_all_prec(model, "ternary")

def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    status = os.path.join(OUT, "H2_2B_STATUS.txt")
    open(status, "w").write("LOADING\n")
    t0 = time.time()
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        tok = AutoTokenizer.from_pretrained(MID)
        model = AutoModelForCausalLM.from_pretrained(MID, dtype=torch.bfloat16).to(dev).eval()
        model.requires_grad_(False)
        n_layers = len(model.model.layers)
        print("loaded", MID, "| layers", n_layers, "| dev", dev)

        state = _load_state()
        if "config" in state and state["config"] != CFG:
            raise RuntimeError("checkpoint config mismatch -- refusing to mix incompatible runs."
                               "\n  loaded:  %s\n  current: %s" % (state["config"], CFG))
        state.setdefault("config", CFG)
        _save_state(state)
        have = {k: (len(v) if isinstance(v, dict) else v) for k, v in state.items() if k != "config"}
        print("checkpoint:", CKPT_PATH, "| resumed:", have)

        calib_batches, held_batches = build_corpora(tok, dev)
        print("corpora: calib %d seqs | held-out %d seqs | disjoint: verified" %
              (len(calib_batches) * BATCH, len(held_batches) * BATCH))

        open(status, "w").write("SELFTEST\n")
        st = selftest_harness(model, held_batches, n_layers)
        state["selftest"] = st
        _save_state(state)
        print("SELFTEST | identical_ternary=%s | int8_engages=%s | ternary ppl=%.3f | all-int8 ppl=%.3f | int8_improves=%s"
              % (st["identical_ternary"], st["int8_engages"], st["ppl_ternary"], st["ppl_allint8"], st["int8_improves"]))
        assert st["identical_ternary"], "ternary mode differs from pristine -- monkeypatch is not faithful"
        assert st["int8_engages"], "int8 mode does not change logits -- mode inert"
        assert st["int8_improves"], ("GATE FAILED: all-int8 PPL (%.3f) > ternary PPL (%.3f) -- the upgrade does "
                                     "not help, so there is no Pareto front. Aborting honestly."
                                     % (st["ppl_allint8"], st["ppl_ternary"]))
        open(status, "w").write("RUNNING\n")

        if "base" not in state:
            set_all_prec(model, "ternary")
            base_held = eval_loss(model, held_batches)
            base_calib = eval_loss(model, calib_batches)
            state["base"] = {"loss_ternary_held": base_held, "ppl_ternary_held": ppl(base_held),
                             "loss_ternary_calib": base_calib}
            _save_state(state)
        b = state["base"]
        print("base | ternary held-out: loss=%.4f ppl=%.3f | calib loss=%.4f"
              % (b["loss_ternary_held"], b["ppl_ternary_held"], b["loss_ternary_calib"]))

        print("=== s_ell : HAWQ Hessian trace (Hutchinson, all-smooth twin, calib) ===")
        compute_s_l(model, calib_batches, n_layers, state)

        print("=== c_ell : causal importance (mean-ablation, deployed ternary, calib) ===")
        compute_c_l(model, calib_batches, n_layers, state["base"]["loss_ternary_calib"], state)

        print("=== oracle Delta-loss (per layer ternary->int8, deployed model, held-out) ===")
        compute_oracle(model, held_batches, n_layers, state["base"]["loss_ternary_held"], state)

        s_l = {int(k): v["trace"] for k, v in state["s_l"].items()}
        c_l = {int(k): v for k, v in state["c_l"].items()}
        oracle_dloss = {int(k): v for k, v in state["oracle_dloss"].items()}
        layers = list(range(n_layers))
        rankings = build_rankings(s_l, c_l, oracle_dloss, n_layers, SEED)
        state["rankings"] = rankings

        from scipy.stats import spearmanr
        r_sc = spearmanr([s_l[l] for l in layers], [c_l[l] for l in layers])
        r_so = spearmanr([s_l[l] for l in layers], [oracle_dloss[l] for l in layers])
        r_co = spearmanr([c_l[l] for l in layers], [oracle_dloss[l] for l in layers])
        state["spearman"] = {"s_vs_c": [float(r_sc.correlation), float(r_sc.pvalue)],
                             "s_vs_oracle": [float(r_so.correlation), float(r_so.pvalue)],
                             "c_vs_oracle": [float(r_co.correlation), float(r_co.pvalue)]}
        _save_state(state)
        print("ranking s_hawq  :", rankings["s_hawq"])
        print("ranking c_causal:", rankings["c_causal"])
        print("ranking oracle  :", rankings["oracle"])
        print("Spearman  s~c = %+.3f (p=%.3g) | s~oracle = %+.3f (p=%.3g) | c~oracle = %+.3f (p=%.3g)"
              % (r_sc.correlation, r_sc.pvalue, r_so.correlation, r_so.pvalue, r_co.correlation, r_co.pvalue))
        for k in BUDGETS:
            print("  top-%2d overlap | s&c=%d | s&oracle=%d | c&oracle=%d (of %d)" % (
                k, len(set(rankings["s_hawq"][:k]) & set(rankings["c_causal"][:k])),
                len(set(rankings["s_hawq"][:k]) & set(rankings["oracle"][:k])),
                len(set(rankings["c_causal"][:k]) & set(rankings["oracle"][:k])), k))

        print("=== Pareto sweep (top-B by ranking -> int8, deployed model, held-out PPL) ===")
        compute_pareto(model, held_batches, rankings, BUDGETS, n_layers, state)

        state["wall_s"] = time.time() - t0
        _save_state(state)
        open(status, "w").write("DONE %.0fs\n" % (time.time() - t0))
        print("DONE %.1fs" % (time.time() - t0))
    except Exception:
        tb = traceback.format_exc()
        print(tb)
        open(status, "w").write("ERROR\n" + tb)
        raise

if __name__ == "__main__":
    main()
