from __future__ import annotations

import json
import math
import os
import random
import re
import sys

ROOT = os.environ.get("POC_ROOT", "/content/PhD-propose")
sys.path[:0] = [ROOT, os.path.join(ROOT, "sasori/src")]

import torch  

MODEL = os.environ.get("NR_MODEL", "Qwen/Qwen2.5-7B-Instruct")
LENGTHS = [int(x) for x in os.environ.get("NR_LENGTHS", "2000,8000").split(",") if x.strip()]
DEPTHS = [float(x) for x in os.environ.get("NR_DEPTHS", "0.1,0.5,0.9").split(",") if x.strip()]
TOPHEADS = int(os.environ.get("NR_TOPHEADS", "20"))
MAXNEW = int(os.environ.get("NR_MAXNEW", "32"))
KVALS = [int(k.strip()) for k in os.environ.get("NR_KVALS", "0,2,3").split(",") if k.strip() != ""]
GROUP = int(os.environ.get("NR_GROUP", "256"))
TRIALS = int(os.environ.get("NR_TRIALS", "2"))
HEAD_LEN = int(os.environ.get("NR_HEAD_LEN", str(min(LENGTHS)))) if LENGTHS else 2000
HEAD_TRIALS = int(os.environ.get("NR_HEAD_TRIALS", "3"))
K_LOW = int(os.environ.get("NR_LOW", "2"))
K_HIGH = int(os.environ.get("NR_HIGH", "3"))
ATTN_IMPL = os.environ.get("NR_ATTN_IMPL", "eager")
SEED = int(os.environ.get("NR_SEED", "0"))
MAXPROMPT = int(os.environ.get("NR_MAXPROMPT", str((max(LENGTHS) if LENGTHS else 8000) + 512)))
TAG = os.environ.get("NR_TAG", "niah_retrieval_k")
OUT_JSON = os.environ.get("NR_OUT_JSON") or os.path.join(ROOT, "sasori/bench", f"{TAG}.json")

_SKIP_QUANT = ("lm_head", "embed_tokens")
_ROUTER_LEAVES = ("gate", "router", "gating")
_EXPERT_LEAVES = ("gate_up_proj", "down_proj", "gate_proj", "up_proj")
_ATTN_PROJ = ("q_proj", "k_proj", "v_proj", "o_proj")
_MIN_DIM = 64  

_LAYER_RE = re.compile(r"\.layers\.(\d+)\.self_attn$")

_FILLER = (
    "The committee reviewed the quarterly notes and adjourned without further remarks. "
    "Sunlight moved slowly across the empty corridor as the afternoon wore on. "
    "A steady breeze carried the sound of distant traffic through the open window. "
    "The manual described each step in careful, deliberate, and unremarkable detail. "
    "Rows of identical shelves stretched toward the far wall of the storage room. "
    "The lecture continued at an even pace, covering the usual introductory material. "
    "Water dripped from the eaves long after the rain itself had finally stopped. "
    "The report summarized the findings and recommended no immediate changes at all. "
)

def _filler_ids(tok, n_tokens: int) -> list[int]:
    base = tok(_FILLER, add_special_tokens=False).input_ids
    if not base:
        base = [tok.encode(" the", add_special_tokens=False)[0]]
    reps = (n_tokens // len(base)) + 1
    ids = (base * reps)[:n_tokens]
    return ids

def _make_needle(rng: random.Random) -> dict:
    key = rng.choice(["the vault", "warehouse 7", "the north gate", "server room B",
                      "the archive", "dock 12", "the west wing", "unit 9"])
    value = str(rng.randint(1000000, 9999999))
    sentence = f"The secret access code for {key} is {value}."
    return {"key": key, "value": value, "sentence": sentence,
            "question": f"What is the secret access code for {key}? Answer with only the number.",
            
            "copy_question": "Copy the exact sentence about the secret access code, word for word."}

def _build_prompt_ids(tok, dev, needle: dict, length: int, depth: float, question_key: str):
    hay = _filler_ids(tok, length)
    needle_ids = tok(needle["sentence"], add_special_tokens=False).input_ids
    at = int(depth * len(hay))
    ctx_ids = hay[:at] + needle_ids + hay[at:]
    ctx_str = tok.decode(ctx_ids)
    user = (f"{ctx_str}\n\n{needle[question_key]}")
    msgs = [{"role": "user", "content": user}]
    try:  
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False)
    except TypeError:
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(prompt, return_tensors="pt", truncation=True, max_length=MAXPROMPT).input_ids.to(dev)
    flat = ids[0].tolist()
    start = _find_subseq(flat, needle_ids)
    if start < 0:  
        val_ids = tok(needle["value"], add_special_tokens=False).input_ids
        start = _find_subseq(flat, val_ids)
        span = set(range(start, start + len(val_ids))) if start >= 0 else set()
        ntoks = set(val_ids)
    else:
        span = set(range(start, start + len(needle_ids)))
        ntoks = set(needle_ids)
    return ids, span, ntoks

def _find_subseq(hay: list[int], needle: list[int]) -> int:
    if not needle:
        return -1
    n = len(needle)
    for i in range(len(hay) - n + 1):
        if hay[i:i + n] == needle:
            return i
    return -1

def load(dev):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, attn_implementation=ATTN_IMPL).to(dev).eval()
    m.requires_grad_(False)
    return m, tok

def _head_dims(model):
    cfg = model.config
    H = cfg.num_attention_heads
    KV = getattr(cfg, "num_key_value_heads", H) or H
    hd = getattr(cfg, "head_dim", None) or (cfg.hidden_size // H)
    return H, KV, hd

@torch.no_grad()
def fake_quant_k_(model, k_planes: int, group: int = GROUP) -> dict:
    import torch.nn as nn
    from sasori.reconstruct import quantize_matrix_k
    st = {"linear": 0, "expert_mats": 0}
    for _name, mod in model.named_modules():
        for leaf, child in list(mod.named_children()):
            if not isinstance(child, nn.Linear):
                continue
            if leaf in _SKIP_QUANT or leaf in _ROUTER_LEAVES:
                continue
            if min(child.weight.shape) < _MIN_DIM:
                continue
            kp = quantize_matrix_k(child.weight.data, k_planes, group=group, row_chunk=65536)
            child.weight.data.copy_(kp.dequantize(child.weight.dtype))
            st["linear"] += 1
    for name, p in model.named_parameters():
        if p.dim() == 3 and "experts" in name and name.split(".")[-1] in _EXPERT_LEAVES:
            for e in range(p.shape[0]):
                kp = quantize_matrix_k(p.data[e], k_planes, group=group, row_chunk=65536)
                p.data[e] = kp.dequantize(p.dtype)
                st["expert_mats"] += 1
    return st

@torch.no_grad()
def _selective_matrix(weight, protected_slices, group, k_low, k_high):
    from sasori.reconstruct import quantize_matrix_k
    lo = quantize_matrix_k(weight, k_low, group=group, row_chunk=65536).dequantize(weight.dtype)
    if not protected_slices:
        return lo, 0
    hi = quantize_matrix_k(weight, k_high, group=group, row_chunk=65536).dequantize(weight.dtype)
    out = lo.clone()
    mask = torch.zeros(out.shape, dtype=torch.bool, device=out.device)
    for dim, a, b in protected_slices:
        if dim == 0:
            out[a:b, :] = hi[a:b, :]
            mask[a:b, :] = True
        else:
            out[:, a:b] = hi[:, a:b]
            mask[:, a:b] = True
    return out, int(mask.sum().item())

@torch.no_grad()
def fake_quant_selective_(model, protected, group, k_low, k_high) -> dict:
    import torch.nn as nn
    from sasori.reconstruct import quantize_matrix_k
    H, KV, hd = _head_dims(model)
    q_per_kv = max(1, H // KV)
    n_hi = 0
    n_total = 0
    st = {"linear": 0, "attn_selective": 0}
    for name, mod in model.named_modules():
        m = _LAYER_RE.search(name)
        layer = int(m.group(1)) if m else None
        for leaf, child in list(mod.named_children()):
            if not isinstance(child, nn.Linear):
                continue
            if leaf in _SKIP_QUANT or leaf in _ROUTER_LEAVES:
                continue
            if min(child.weight.shape) < _MIN_DIM:
                continue
            heads = protected.get(layer, set()) if layer is not None else set()
            if layer is not None and leaf in _ATTN_PROJ and heads:
                slices = []
                if leaf == "q_proj":
                    slices = [(0, h * hd, (h + 1) * hd) for h in heads]
                elif leaf == "o_proj":
                    slices = [(1, h * hd, (h + 1) * hd) for h in heads]
                elif leaf in ("k_proj", "v_proj"):
                    kvg = sorted({h // q_per_kv for h in heads})
                    slices = [(0, g * hd, (g + 1) * hd) for g in kvg]
                w_new, hi = _selective_matrix(child.weight.data, slices, group, k_low, k_high)
                child.weight.data.copy_(w_new)
                n_hi += hi
                st["attn_selective"] += 1
            else:
                kp = quantize_matrix_k(child.weight.data, k_low, group=group, row_chunk=65536)
                child.weight.data.copy_(kp.dequantize(child.weight.dtype))
                st["linear"] += 1
            n_total += child.weight.numel()
    st["n_hi_elems"] = n_hi
    st["n_total_elems"] = n_total
    return st

def _bpw(K: int, group: int) -> float:
    return K * math.log2(3.0) + K * 32.0 / group

def _bpw_note(stats: dict, group: int, k_low: int, k_high: int) -> str:
    n_hi = stats.get("n_hi_elems", 0)
    n_tot = max(1, stats.get("n_total_elems", 1))
    frac = n_hi / n_tot
    eff = (n_hi * _bpw(k_high, group) + (n_tot - n_hi) * _bpw(k_low, group)) / n_tot
    base = _bpw(k_low, group)
    return (f"effective ~{eff:.4f} bpw vs uniform K{k_low} {base:.4f} bpw "
            f"(delta +{eff - base:.5f}); {100.0 * frac:.4f}% of quantized weights upgraded to "
            f"K{k_high}; formula bpw(K)=K*log2(3)+K*32/g, g={group}; excludes FP lm_head/embeds; "
            f"analytic estimate, not packed-kernel measured")

@torch.no_grad()
def niah_accuracy(model, tok, dev, rng: random.Random) -> dict:
    acc = {}
    by_depth = {}
    for length in LENGTHS:
        hits = 0
        tot = 0
        by_depth[str(length)] = {}
        for depth in DEPTHS:
            d_hits = 0
            d_tot = 0
            for t in range(TRIALS):
                needle = _make_needle(rng)
                ids, _span, _nt = _build_prompt_ids(tok, dev, needle, length, depth, "question")
                plen = ids.shape[1]
                gen = model.generate(ids, max_new_tokens=MAXNEW, do_sample=False,
                                     pad_token_id=tok.eos_token_id)
                txt = tok.decode(gen[0, plen:], skip_special_tokens=True)
                ok = needle["value"] in txt.replace(",", "")
                d_hits += int(ok)
                d_tot += 1
            by_depth[str(length)][str(depth)] = d_hits / d_tot if d_tot else 0.0
            hits += d_hits
            tot += d_tot
        acc[str(length)] = hits / tot if tot else 0.0
    acc["by_depth"] = by_depth
    return acc

@torch.no_grad()
def _copy_paste_scores(model, tok, dev, needle, length, depth, n_layers, H):
    ids, span, ntoks = _build_prompt_ids(tok, dev, needle, length, depth, "copy_question")
    input_ids = ids[0].tolist()
    counts = torch.zeros(n_layers, H)
    n_ops = 0
    eos = tok.eos_token_id

    out = model(input_ids=ids, use_cache=True)          
    past = out.past_key_values
    cur = int(out.logits[0, -1].argmax().item())         
    seq = list(input_ids)                                

    for _step in range(MAXNEW):
        seq.append(cur)
        tin = torch.tensor([[cur]], device=dev)
        o = model(input_ids=tin, past_key_values=past, use_cache=True, output_attentions=True)
        past = o.past_key_values
        nxt = int(o.logits[0, -1].argmax().item())       
        if nxt in ntoks:                                  
            n_ops += 1
            for L in range(n_layers):
                a = o.attentions[L][0, :, -1, :]          
                amax = a.argmax(dim=-1)                    
                for h in range(H):
                    p = int(amax[h].item())
                    if p in span and p < len(seq) and seq[p] == nxt:
                        counts[L, h] += 1.0
        cur = nxt
        if nxt == eos:
            break
    scores = counts / n_ops if n_ops > 0 else counts
    return scores, n_ops

@torch.no_grad()
def detect_retrieval_heads(model, tok, dev, rng: random.Random):
    n_layers = len(model.model.layers)
    H, _KV, _hd = _head_dims(model)
    agg = torch.zeros(n_layers, H)
    used = 0
    per_inst = []
    depth = DEPTHS[len(DEPTHS) // 2] if DEPTHS else 0.5   
    for t in range(HEAD_TRIALS):
        needle = _make_needle(rng)
        scores, n_ops = _copy_paste_scores(model, tok, dev, needle, HEAD_LEN, depth, n_layers, H)
        per_inst.append({"n_copy_ops": n_ops})
        if n_ops > 0:
            agg += scores
            used += 1
    if used > 0:
        agg /= used
    flat = agg.flatten()
    k = min(TOPHEADS, flat.numel())
    top = torch.topk(flat, k)
    heads = []
    for val, idx in zip(top.values.tolist(), top.indices.tolist()):
        heads.append({"layer": idx // H, "head": idx % H, "score": round(val, 5)})
    return heads, {"head_len": HEAD_LEN, "depth": depth, "instances": per_inst, "used": used}

def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    import transformers

    print("env: torch %s | tf %s | dev %s | MODEL %s | lengths %s | depths %s | K %s | "
          "top-heads %d | max_new %d | group %d | trials %d | head_len %d | attn %s"
          % (torch.__version__, transformers.__version__, dev, MODEL, LENGTHS, DEPTHS, KVALS,
             TOPHEADS, MAXNEW, GROUP, TRIALS, HEAD_LEN, ATTN_IMPL), flush=True)

    meta = {"model": MODEL, "lengths": LENGTHS, "depths": DEPTHS, "kvals": KVALS,
            "topheads": TOPHEADS, "max_new_tokens": MAXNEW, "group": GROUP, "trials": TRIALS,
            "head_len": HEAD_LEN, "head_trials": HEAD_TRIALS, "k_low": K_LOW, "k_high": K_HIGH,
            "attn_impl": ATTN_IMPL, "seed": SEED,
            "question": ("does K2 kill the retrieval heads, and does per-head K3 on just the "
                         "retrieval heads restore NIAH at ~K2 bit-cost? (#72)"),
            "retrieval_protocol": ("Wu et al. 2024 arXiv:2404.15574 copy-paste score: per decode "
                                   "step, a head retrieves iff the emitted token is a needle "
                                   "token AND the head's top-attended context position is in the "
                                   "needle span AND the context token there equals the emitted "
                                   "token; head score = copies / needle-copy-steps; decode-only "
                                   "attention, detected at HEAD_LEN, applied at all lengths")}
    R = {"_meta": meta, "niah": {}, "retrieval_heads": [], "selective_k3": {}}

    def _flush():
        os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
        json.dump(R, open(OUT_JSON, "w"), indent=2)

    heads_info = {}
    retrieval_heads = []
    if ATTN_IMPL == "eager":
        m, tok = load(dev)
        retrieval_heads, heads_info = detect_retrieval_heads(m, tok, dev, random.Random(SEED))
        R["retrieval_heads"] = retrieval_heads
        R["_meta"]["head_detection"] = heads_info
        print("retrieval heads (top %d): %s" % (
            len(retrieval_heads), [(h["layer"], h["head"], h["score"]) for h in retrieval_heads[:8]]),
            flush=True)
        
        if 0 in KVALS:
            R["niah"]["FP"] = niah_accuracy(m, tok, dev, random.Random(SEED))
            print("VARIANT FP  niah %s" % {k: v for k, v in R["niah"]["FP"].items() if k != "by_depth"},
                  flush=True)
        del m
        if dev == "cuda":
            torch.cuda.empty_cache()
        _flush()
    else:
        print("NR_ATTN_IMPL=%s (not eager): skipping retrieval-head detection + selective variant "
              "(attentions unavailable)." % ATTN_IMPL, flush=True)

    for k_planes in KVALS:
        if k_planes == 0:
            if "FP" in R["niah"]:
                continue  
            m, tok = load(dev)
            R["niah"]["FP"] = niah_accuracy(m, tok, dev, random.Random(SEED))
            print("VARIANT FP  niah %s" % {k: v for k, v in R["niah"]["FP"].items() if k != "by_depth"},
                  flush=True)
            del m
            if dev == "cuda":
                torch.cuda.empty_cache()
            _flush()
            continue
        label = f"K{k_planes}"
        m, tok = load(dev)
        st = fake_quant_k_(m, k_planes, GROUP)
        print("[%s] fake-quant reached: %s (group=%d)" % (label, st, GROUP), flush=True)
        R["niah"][label] = niah_accuracy(m, tok, dev, random.Random(SEED))
        print("VARIANT %-4s niah %s" % (label, {k: v for k, v in R["niah"][label].items() if k != "by_depth"}),
              flush=True)
        del m
        if dev == "cuda":
            torch.cuda.empty_cache()
        _flush()

    if retrieval_heads:
        protected = {}
        for h in retrieval_heads:
            protected.setdefault(h["layer"], set()).add(h["head"])
        label = f"K{K_LOW}+K{K_HIGH}heads"
        m, tok = load(dev)
        st = fake_quant_selective_(m, protected, GROUP, K_LOW, K_HIGH)
        note = _bpw_note(st, GROUP, K_LOW, K_HIGH)
        print("[%s] selective fake-quant: %s | %s" % (label, st, note), flush=True)
        sel = niah_accuracy(m, tok, dev, random.Random(SEED))
        R["niah"][label] = sel
        R["selective_k3"] = {
            "acc": {k: v for k, v in sel.items() if k != "by_depth"},
            "by_depth": sel.get("by_depth", {}),
            "bpw_note": note,
            "protected": {str(l): sorted(hs) for l, hs in protected.items()},
            "n_protected_heads": sum(len(hs) for hs in protected.values()),
            "quant_stats": st,
        }
        print("VARIANT %-10s niah %s" % (label, R["selective_k3"]["acc"]), flush=True)
        del m
        if dev == "cuda":
            torch.cuda.empty_cache()
        _flush()

    print("\n=== SUMMARY %s (lengths=%s) ===" % (MODEL, LENGTHS), flush=True)
    for label, acc in R["niah"].items():
        print("  %-12s %s" % (label, {k: round(v, 3) for k, v in acc.items() if k != "by_depth"}),
              flush=True)
    if R["selective_k3"]:
        print("  selective bpw: %s" % R["selective_k3"]["bpw_note"], flush=True)
    print("WROTE", OUT_JSON, flush=True)

if __name__ == "__main__":
    main()
