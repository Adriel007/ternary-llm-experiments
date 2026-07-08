from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter

ROOT = os.environ.get("POC_ROOT", "/content/PhD-propose")
sys.path[:0] = [ROOT, os.path.join(ROOT, "sasori/src")]

import torch  

MODEL = os.environ.get("SC_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")

KVALS = [int(k.strip()) for k in os.environ.get("SC_K", "0,2,3").split(",") if k.strip() != ""]
T = float(os.environ.get("SC_T", "0.7"))
TOPP = float(os.environ.get("SC_TOPP", "0.95"))
K_SAMPLES = int(os.environ.get("SC_K_SAMPLES", "8"))
N = int(os.environ.get("SC_N", "40"))
MAX_NEW_TOKENS = int(os.environ.get("SC_MAX_NEW_TOKENS", "400"))   
GROUP = int(os.environ.get("SC_GROUP", "256"))                      
SEED = int(os.environ.get("SC_SEED", "0"))
DUMP = int(os.environ.get("SC_DUMP", "2"))
TAG = os.environ.get("SC_TAG", "self_consistency")
OUT_JSON = os.environ.get("SC_OUT_JSON") or os.path.join(ROOT, "sasori/bench", f"{TAG}.json")

_SKIP_QUANT = ("lm_head", "embed_tokens")
_ROUTER_LEAVES = ("gate", "router", "gating")
_EXPERT_LEAVES = ("gate_up_proj", "down_proj", "gate_proj", "up_proj")

_NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")

def _k_schedule(k_samples: int) -> list[int]:
    ks = {1, k_samples} | {k for k in (1, 4, 8, 16, 32, 64) if k <= k_samples}
    return sorted(k for k in ks if 1 <= k <= k_samples)

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
            if min(child.weight.shape) < 64:
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

def load(dev):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to(dev).eval()
    m.requires_grad_(False)
    return m, tok

def _prompt_ids(tok, question, dev):
    msgs = [{"role": "user", "content": question +
             "\nReason step by step, then give the final answer after ####."}]
    try:                                                      
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return tok(prompt, return_tensors="pt", truncation=True, max_length=1024).input_ids.to(dev)

def _gold(ex) -> str:
    return ex["answer"].split("####")[-1].strip().replace(",", "")

def _extract(txt: str):
    nums = _NUM_RE.findall(txt.replace(",", ""))
    return nums[-1] if nums else None

def _majority(preds):
    votes = [p for p in preds if p is not None]
    if not votes:
        return None
    return Counter(votes).most_common(1)[0][0]

@torch.no_grad()
def run_variant(model, tok, dev, ds, n, ks, dump=0):
    torch.manual_seed(SEED)          
    greedy_correct = 0
    correct_at = {k: 0 for k in ks}
    total = 0
    samples = []
    for ex in ds:
        if total >= n:
            break
        total += 1
        ids = _prompt_ids(tok, ex["question"], dev)
        plen = ids.shape[1]
        gold = _gold(ex)

        gg = model.generate(ids, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                            pad_token_id=tok.eos_token_id)
        greedy_txt = tok.decode(gg[0, plen:], skip_special_tokens=True)
        greedy_pred = _extract(greedy_txt)
        greedy_ok = greedy_pred is not None and greedy_pred == gold
        greedy_correct += int(greedy_ok)

        gs = model.generate(ids, max_new_tokens=MAX_NEW_TOKENS, do_sample=True,
                            temperature=T, top_p=TOPP, num_return_sequences=K_SAMPLES,
                            pad_token_id=tok.eos_token_id)
        sample_preds = [_extract(tok.decode(gs[i, plen:], skip_special_tokens=True))
                        for i in range(K_SAMPLES)]

        for k in ks:
            mp = _majority(sample_preds[:k])
            correct_at[k] += int(mp is not None and mp == gold)

        if len(samples) < dump:
            samples.append({"q": ex["question"][:120], "gold": gold,
                            "greedy_pred": greedy_pred, "greedy_ok": greedy_ok,
                            "sample_preds": sample_preds,
                            "maj": {str(k): _majority(sample_preds[:k]) for k in ks}})
    return {
        "n": total,
        "greedy_acc": (greedy_correct / total if total else 0.0),
        "maj_at_k": {str(k): (correct_at[k] / total if total else 0.0) for k in ks},
        "samples": samples,
    }

def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    import transformers
    from datasets import load_dataset

    ks = _k_schedule(K_SAMPLES)
    print("env: torch %s | tf %s | dev %s | MODEL %s | K %s | T %.2f top_p %.2f | "
          "k_samples %d | N %d | ks %s | max_new %d | group %d"
          % (torch.__version__, transformers.__version__, dev, MODEL, KVALS, T, TOPP,
             K_SAMPLES, N, ks, MAX_NEW_TOKENS, GROUP), flush=True)

    ds = load_dataset("openai/gsm8k", "main", split="test")

    meta = {"model": MODEL, "kvals": KVALS, "T": T, "top_p": TOPP, "k_samples": K_SAMPLES,
            "n": N, "ks": ks, "max_new_tokens": MAX_NEW_TOKENS, "group": GROUP, "seed": SEED,
            "question": "does temperature+maj@k rescue K2 as a 3rd lever (vs K3-weights / distill)?"}
    R = {"_meta": meta, "variants": {}}

    for k_planes in KVALS:
        label = "FP" if k_planes == 0 else f"K{k_planes}"
        m, tok = load(dev)
        quant = None
        if k_planes > 0:
            quant = fake_quant_k_(m, k_planes, GROUP)
            print("[%s] fake-quant reached: %s (group=%d)" % (label, quant, GROUP), flush=True)
        out = run_variant(m, tok, dev, ds, N, ks, dump=DUMP)
        out["quant"] = quant
        R["variants"][label] = out
        print("VARIANT %-4s greedy %.3f | maj@k %s"
              % (label, out["greedy_acc"],
                 {k: round(v, 3) for k, v in out["maj_at_k"].items()}), flush=True)
        del m
        if dev == "cuda":
            torch.cuda.empty_cache()
        os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
        json.dump(R, open(OUT_JSON, "w"), indent=2)

    print("\n=== SUMMARY %s (N=%d, T=%.2f, k_samples=%d) ===" % (MODEL, N, T, K_SAMPLES), flush=True)
    for label, out in R["variants"].items():
        maj = out["maj_at_k"]
        print("  %-4s greedy %.3f | maj@1 %.3f | maj@%d %.3f"
              % (label, out["greedy_acc"], maj.get("1", float("nan")),
                 K_SAMPLES, maj.get(str(ks[-1]), float("nan"))), flush=True)
    print("WROTE", OUT_JSON, flush=True)

if __name__ == "__main__":
    main()
