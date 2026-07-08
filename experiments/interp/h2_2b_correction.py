
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

N_TRAIN_SEQS = 1024        
N_VAL_SEQS = 32            
N_TEST_SEQS = 64           
N_CALIB_SEQS = 32          
HESSIAN_BSZ = 1            
N_HUTCH = 8
RANK = 8                   
ALPHA = 16                 
MAX_STEPS = 400            
EVAL_EVERY = 40            
PATIENCE = 3               
LR = 2e-4                  
WD = 0.01                  
BUDGETS = [1, 2, 3, 5, 8, 11, 15]
COMPOSE_CHECK_B = 5        
SEED = 0

CKPT_DIR = os.path.join(DRIVE, "h2_2b_corr_checkpoints")
CKPT_PATH = os.path.join(CKPT_DIR, "h2_2b_corr_state.json")
LOCAL_PATH = os.path.join(OUT, "h2_2b_correction.json")

CFG = {
    "model": MID, "seq_len": SEQ_LEN, "batch": BATCH,
    "n_train_seqs": N_TRAIN_SEQS, "n_val_seqs": N_VAL_SEQS, "n_test_seqs": N_TEST_SEQS,
    "n_calib_seqs": N_CALIB_SEQS,
    "rank": RANK, "alpha": ALPHA, "max_steps": MAX_STEPS, "eval_every": EVAL_EVERY,
    "patience": PATIENCE, "lr": LR, "wd": WD, "n_hutch": N_HUTCH, "budgets": BUDGETS, "seed": SEED,
    "mechanism": "trained low-rank FP16 correction (LoRA) per layer; backbone frozen; early-stop on val",
    "calib_corpus": "Salesforce/wikitext wikitext-103-raw-v1 (streaming, train split)",
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

def install_unified_forward():
    from transformers.integrations.bitnet import AutoBitLinear, WeightQuant, ActQuant

    def patched_forward(self, input):
        x = self.rms_norm(input) if self.rms_norm is not None else input
        prec = getattr(self, "_prec", "ternary")
        if prec == "smooth":

            
            with torch.no_grad():
                scale = (1.0 / self.weight.float().abs().mean().clamp_(min=1e-5)).to(self.weight.dtype)
            weight = (self.weight * scale).clamp(-1, 1) / scale
            out = TF.linear(x, weight, self.bias)
        else:
            if prec == "int8":
                w = self.weight.float()
                scale = 1.0 / w.abs().mean().clamp_(min=1e-5)
                q = (w * scale * 127).round().clamp(-127, 127)
                weight = (q / 127 / scale).to(self.weight.dtype)
            else:
                weight = WeightQuant.apply(self.weight)
            out = TF.linear(ActQuant.apply(x), weight, self.bias)
        ad = getattr(self, "_adapter", None)
        if ad is not None and getattr(self, "_adapter_active", False):
            A, B, sc = ad
            corr = TF.linear(TF.linear(x.to(A.dtype), A), B) * sc
            out = out + corr.to(out.dtype)
        return out

    AutoBitLinear.forward = patched_forward

def attach_adapters(model, rank, alpha, device, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    sc = alpha / rank
    per_layer_params, backbone_per_layer, extra_per_layer = {}, None, None
    for li, layer in enumerate(model.model.layers):
        params, n_extra, n_back = [], 0, 0
        for bl in _bitlinears(layer):
            d_out, d_in = bl.weight.shape
            A = (torch.randn(rank, d_in, generator=g) / math.sqrt(d_in)).to(device=device, dtype=torch.float32)
            B = torch.zeros(d_out, rank, device=device, dtype=torch.float32)
            A.requires_grad_(False); B.requires_grad_(False)
            bl._adapter = (A, B, sc)
            bl._adapter_active = False
            params.append((A, B))
            n_extra += A.numel() + B.numel()
            n_back += bl.weight.numel()
        per_layer_params[li] = params
        if extra_per_layer is None:
            extra_per_layer, backbone_per_layer = n_extra, n_back
    return per_layer_params, extra_per_layer / backbone_per_layer

def set_layers_active(model, idxs):
    sel = set(int(i) for i in idxs)
    for li, layer in enumerate(model.model.layers):
        for bl in _bitlinears(layer):
            bl._adapter_active = li in sel

def set_all_prec(model, mode):
    for layer in model.model.layers:
        for bl in _bitlinears(layer):
            bl._prec = mode

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

def build_corpora(tok, device):
    sizes = [N_TRAIN_SEQS, N_VAL_SEQS, N_TEST_SEQS, N_CALIB_SEQS]
    pool = _token_pool(tok, sum(sizes), SEQ_LEN)
    off = np.cumsum([0] + sizes)
    slices = [pool[off[i]:off[i + 1]] for i in range(4)]
    
    seen = {}
    for nm, sl in zip(("train", "val", "test", "calib"), slices):
        for row in sl.tolist():
            key = tuple(row)
            assert key not in seen, f"leakage: a sequence appears in both {seen.get(key)} and {nm}"
            seen[key] = nm
    mk = lambda ids: [ids[i:i + BATCH].to(device) for i in range(0, ids.shape[0], BATCH)]
    return tuple(mk(sl) for sl in slices)   

@torch.no_grad()
def eval_loss(model, batches):
    return float(np.mean([float(model(input_ids=b, labels=b).loss.item()) for b in batches]))

def ppl(loss):
    return math.exp(loss)

def train_one_layer(model, layer_params, train_batches, val_batches, seed):
    torch.manual_seed(seed)
    flat = []
    for (A, B) in layer_params:
        A.requires_grad_(True); B.requires_grad_(True)
        flat += [A, B]
    opt = torch.optim.AdamW(flat, lr=LR, weight_decay=WD)
    set_all_prec(model, "ternary")
    nb = len(train_batches)
    best_val, best_state, bad, stopped = float("inf"), None, 0, MAX_STEPS
    for step in range(1, MAX_STEPS + 1):
        b = train_batches[(step - 1) % nb]
        loss = model(input_ids=b, labels=b).loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % EVAL_EVERY == 0 or step == MAX_STEPS:
            vl = eval_loss(model, val_batches)
            if vl < best_val - 1e-4:
                best_val, bad = vl, 0
                best_state = [(A.detach().clone(), B.detach().clone()) for (A, B) in layer_params]
            else:
                bad += 1
                if bad >= PATIENCE:
                    stopped = step
                    break
    if best_state is not None:          
        with torch.no_grad():
            for (A, B), (Ab, Bb) in zip(layer_params, best_state):
                A.copy_(Ab); B.copy_(Bb)
    for (A, B) in layer_params:
        A.requires_grad_(False); B.requires_grad_(False)
    return best_val, stopped

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
    state.setdefault("s_l", {})
    done = set(int(k) for k in state["s_l"])
    if len(done) == n_layers:
        print("  [s_l] complete -- skip"); return
    set_all_prec(model, "smooth")
    set_layers_active(model, [])                 
    groups = {i: [m.weight for m in _bitlinears(layer)] for i, layer in enumerate(model.model.layers)}
    xb = calib_batches[0][:HESSIAN_BSZ]
    for l in range(n_layers):
        if l in done:
            continue
        t = time.time()
        for p in groups[l]:
            p.requires_grad_(True)
        with sdpa_kernel(SDPBackend.MATH):       
            loss = model(input_ids=xb, labels=xb).loss
            grads_l = torch.autograd.grad(loss, groups[l], create_graph=True, retain_graph=True)
            m, s = _hessian_trace_one_layer(grads_l, groups[l], N_HUTCH, SEED * 1000 + l)
        for p in groups[l]:
            p.requires_grad_(False)
        del loss, grads_l
        torch.cuda.empty_cache()
        if not state["s_l"] and abs(m) < 1e-9:
            raise RuntimeError("smooth-twin Hessian degenerate (trace~0 on L%d) -- aborting" % l)
        state["s_l"][str(l)] = {"trace": m, "sem": s}; _save_state(state)
        print("  [s_l] L%2d trace=%+.4g +/- %.2g (%.1fs)" % (l, m, s, time.time() - t))
    set_all_prec(model, "ternary"); torch.cuda.empty_cache()

@torch.no_grad()
def compute_c_l(model, calib_batches, n_layers, base_calib, state):
    state.setdefault("c_l", {})
    done = set(int(k) for k in state["c_l"])
    if len(done) == n_layers:
        print("  [c_l] complete -- skip"); return
    set_all_prec(model, "ternary")
    set_layers_active(model, [])
    nodes = dict(_node_modules(model))
    for l in range(n_layers):
        if l in done:
            continue
        t = time.time()

        def mk():
            def hook(m, i, o):
                o0 = _out0(o)
                mean = o0.float().mean(dim=(0, 1), keepdim=True).to(o0.dtype)
                rep = mean.expand_as(o0).clone()
                return (rep,) + tuple(o[1:]) if isinstance(o, tuple) else rep
            return hook

        h = [nodes[f"a{l}"].register_forward_hook(mk()), nodes[f"m{l}"].register_forward_hook(mk())]
        abl = eval_loss(model, calib_batches)
        for hh in h:
            hh.remove()
        state["c_l"][str(l)] = float(abl - base_calib); _save_state(state)
        print("  [c_l] L%2d Delta=%+.4f (%.1fs)" % (l, state["c_l"][str(l)], time.time() - t))

def _adapter_path(li):
    return os.path.join(CKPT_DIR, f"adapter_L{li:02d}.pt")

def train_all_layers(model, per_layer_params, train_batches, val_batches, test_batches,
                     n_layers, base_held, state):
    state.setdefault("recovered", {})       
    state.setdefault("val_loss", {})
    for li in range(n_layers):
        if str(li) in state["recovered"] and os.path.exists(_adapter_path(li)):
            continue
        t = time.time()
        set_layers_active(model, [li])           
        set_all_prec(model, "ternary")
        vl, stopped = train_one_layer(model, per_layer_params[li], train_batches, val_batches, SEED * 100 + li)
        loss_li = eval_loss(model, test_batches)
        set_layers_active(model, [])
        
        sd = {f"{k}_{j}": p.detach().cpu() for j, (A, B) in enumerate(per_layer_params[li])
              for k, p in (("A", A), ("B", B))}
        torch.save(sd, _adapter_path(li))
        state["val_loss"][str(li)] = vl
        state["recovered"][str(li)] = float(base_held - loss_li)     
        _save_state(state)
        print("  [adapter] L%2d best_val=%.4f stop@%d | test ppl %.3f->%.3f | recovered Delta-loss=%+.4f (%.1fs)"
              % (li, vl, stopped, ppl(base_held), ppl(loss_li), state["recovered"][str(li)], time.time() - t))

def load_all_adapters(model, per_layer_params, n_layers):
    for li in range(n_layers):
        sd = torch.load(_adapter_path(li), map_location="cpu")
        for j, (A, B) in enumerate(per_layer_params[li]):
            A.copy_(sd[f"A_{j}"].to(A.device, A.dtype))
            B.copy_(sd[f"B_{j}"].to(B.device, B.dtype))

def build_rankings(s_l, c_l, recovered, n_layers, seed):
    rng = np.random.default_rng(seed)
    return {
        "s_hawq":   sorted(range(n_layers), key=lambda l: s_l[l], reverse=True),
        "c_causal": sorted(range(n_layers), key=lambda l: c_l[l], reverse=True),
        "oracle":   sorted(range(n_layers), key=lambda l: recovered[l], reverse=True),
        "random":   [int(x) for x in rng.permutation(n_layers)],
    }

@torch.no_grad()
def compute_pareto(model, held_batches, rankings, budgets, extra_frac, state):
    state.setdefault("pareto", {})
    set_all_prec(model, "ternary")
    for name, order in rankings.items():
        state["pareto"].setdefault(name, {})
        for B in budgets:
            if str(B) in state["pareto"][name]:
                continue
            t = time.time()
            set_layers_active(model, order[:B])
            loss_b = eval_loss(model, held_batches)
            set_layers_active(model, [])
            state["pareto"][name][str(B)] = {"loss": float(loss_b), "ppl": ppl(loss_b),
                                             "extra_mem_frac": B / len(model.model.layers) * extra_frac}
            _save_state(state)
            print("  [pareto] %-9s B=%2d layers %s -> ppl=%.3f (%.1fs)" %
                  (name, B, sorted(order[:B]), ppl(loss_b), time.time() - t))
    set_all_prec(model, "ternary"); set_layers_active(model, [])

def joint_train_eval(model, per_layer_params, layers, train_b, val_b, test_b, seed):
    for li in layers:
        for (A, B) in per_layer_params[li]:
            B.zero_()
    flat = [p for li in layers for AB in per_layer_params[li] for p in AB]
    for p in flat:
        p.requires_grad_(True)
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(flat, lr=LR, weight_decay=WD)
    set_all_prec(model, "ternary"); set_layers_active(model, layers)
    nb, best, bad, best_state = len(train_b), float("inf"), 0, None
    for step in range(1, MAX_STEPS + 1):
        b = train_b[(step - 1) % nb]
        loss = model(input_ids=b, labels=b).loss
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if step % EVAL_EVERY == 0 or step == MAX_STEPS:
            vl = eval_loss(model, val_b)
            if vl < best - 1e-4:
                best, bad, best_state = vl, 0, [p.detach().clone() for p in flat]
            else:
                bad += 1
                if bad >= PATIENCE:
                    break
    if best_state is not None:
        with torch.no_grad():
            for p, pb in zip(flat, best_state):
                p.copy_(pb)
    test_loss = eval_loss(model, test_b)
    for p in flat:
        p.requires_grad_(False)
    set_layers_active(model, [])
    return float(test_loss)

def compute_pareto_joint(model, train_b, val_b, test_b, rankings, budgets, per_layer_params,
                         extra_frac, n_layers, state):
    state.setdefault("pareto_joint", {})
    cache = {}
    for nm in state["pareto_joint"]:
        for B, d in state["pareto_joint"][nm].items():
            cache[frozenset(d["layers"])] = d["loss"]
    for name, order in rankings.items():
        state["pareto_joint"].setdefault(name, {})
        for B in budgets:
            if str(B) in state["pareto_joint"][name]:
                continue
            t = time.time()
            layers = sorted(order[:B]); key = frozenset(layers)
            if key in cache:
                loss_b = cache[key]
            else:
                loss_b = joint_train_eval(model, per_layer_params, layers, train_b, val_b, test_b, SEED * 7 + B)
                cache[key] = loss_b
            state["pareto_joint"][name][str(B)] = {"loss": loss_b, "ppl": ppl(loss_b),
                "layers": layers, "extra_mem_frac": B / n_layers * extra_frac}
            _save_state(state)
            print("  [pareto-joint] %-9s B=%2d layers %s -> ppl=%.3f (%.1fs)" %
                  (name, B, layers, ppl(loss_b), time.time() - t))
    set_all_prec(model, "ternary"); set_layers_active(model, [])

def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    status = os.path.join(OUT, "H2_2B_CORR_STATUS.txt")
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
        per_layer_params, extra_frac = attach_adapters(model, RANK, ALPHA, dev, SEED)
        print("loaded", MID, "| layers", n_layers, "| extra-mem/layer = %.4f (rank %d)" % (extra_frac, RANK))

        state = _load_state()
        if "config" in state:   
            saved = {k: state["config"].get(k) for k in CFG}
            if saved != CFG:
                raise RuntimeError("checkpoint config mismatch.\n loaded %s\n cur %s" % (saved, CFG))
        state["config"] = dict(CFG)
        state["extra_mem_frac_per_layer"] = extra_frac
        _save_state(state)
        print("checkpoint:", CKPT_PATH, "| resumed:",
              {k: (len(v) if isinstance(v, dict) else v) for k, v in state.items() if k != "config"})

        train_batches, val_batches, test_batches, calib_batches = build_corpora(tok, dev)
        print("corpora (disjoint verified): train %d | val %d | test %d | calib %d seqs" %
              (len(train_batches) * BATCH, len(val_batches) * BATCH,
               len(test_batches) * BATCH, len(calib_batches) * BATCH))

        open(status, "w").write("BASE\n")
        if "base" not in state:
            set_all_prec(model, "ternary"); set_layers_active(model, [])
            base_held = eval_loss(model, test_batches)
            base_calib = eval_loss(model, calib_batches)
            
            state["base"] = {"loss_ternary_held": base_held, "ppl_ternary_held": ppl(base_held),
                             "loss_ternary_calib": base_calib}
            _save_state(state)
        b = state["base"]
        print("base | ternary held-out loss=%.4f ppl=%.3f | calib loss=%.4f"
              % (b["loss_ternary_held"], b["ppl_ternary_held"], b["loss_ternary_calib"]))

        open(status, "w").write("S_L\n")
        print("=== s_ell : HAWQ Hessian trace (smooth twin, calib) ===")
        compute_s_l(model, calib_batches, n_layers, state)

        open(status, "w").write("C_L\n")
        print("=== c_ell : mean-ablation (deployed ternary, calib) ===")
        compute_c_l(model, calib_batches, n_layers, state["base"]["loss_ternary_calib"], state)

        open(status, "w").write("TRAIN\n")
        print("=== per-layer adapter training (train pool, early-stop on val) -> recovered Delta-loss (oracle) ===")
        train_all_layers(model, per_layer_params, train_batches, val_batches, test_batches, n_layers,
                         state["base"]["loss_ternary_held"], state)

        s_l = {int(k): v["trace"] for k, v in state["s_l"].items()}
        c_l = {int(k): v for k, v in state["c_l"].items()}
        recovered = {int(k): v for k, v in state["recovered"].items()}
        layers = list(range(n_layers))
        rankings = build_rankings(s_l, c_l, recovered, n_layers, SEED)
        state["rankings"] = rankings

        from scipy.stats import spearmanr
        r_sc = spearmanr([s_l[l] for l in layers], [c_l[l] for l in layers])
        r_sr = spearmanr([s_l[l] for l in layers], [recovered[l] for l in layers])
        r_cr = spearmanr([c_l[l] for l in layers], [recovered[l] for l in layers])
        state["spearman"] = {"s_vs_c": [float(r_sc.correlation), float(r_sc.pvalue)],
                             "s_vs_recovered": [float(r_sr.correlation), float(r_sr.pvalue)],
                             "c_vs_recovered": [float(r_cr.correlation), float(r_cr.pvalue)]}
        _save_state(state)
        print("ranking oracle  :", rankings["oracle"])
        print("ranking c_causal:", rankings["c_causal"])
        print("ranking s_hawq  :", rankings["s_hawq"])
        print("Spearman  s~c=%+.3f (p=%.2g) | s~recovered=%+.3f (p=%.2g) | c~recovered=%+.3f (p=%.2g)"
              % (r_sc.correlation, r_sc.pvalue, r_sr.correlation, r_sr.pvalue, r_cr.correlation, r_cr.pvalue))

        open(status, "w").write("PARETO\n")
        print("=== Pareto sweep (JOINT per-allocation training of top-B layers, test PPL) ===")
        compute_pareto_joint(model, train_batches, val_batches, test_batches, rankings, BUDGETS,
                             per_layer_params, extra_frac, n_layers, state)

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
