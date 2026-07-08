from __future__ import annotations

import json
import os
import re
import sys
import time
import traceback
from statistics import median

ROOT = os.environ.get("POC_ROOT", "/content/PhD-propose")
sys.path[:0] = [ROOT, os.path.join(ROOT, "sasori/src")]

import torch  

from sasori.reconstruct import quantize_matrix_k  

MODEL = os.environ.get("MER_MODEL", "Qwen/Qwen1.5-MoE-A2.7B")
N_CALIB = int(os.environ.get("MER_N", "16"))          
SEQLEN = int(os.environ.get("MER_SEQLEN", "256"))     
GROUP = int(os.environ.get("MER_GROUP", "256"))       
K = int(os.environ.get("MER_K", "2"))                 
BATCH = int(os.environ.get("MER_BATCH", "4"))         
CHUNK = int(os.environ.get("MER_CHUNK", "512"))       
SVD_MAXDIM = int(os.environ.get("MER_SVD_MAXDIM", "4096"))  
_LAYERS_ENV = os.environ.get("MER_LAYERS", "").strip()      
LAYERS = [int(x) for x in _LAYERS_ENV.split(",") if x != ""] if _LAYERS_ENV else None
TAG = os.environ.get("MER_TAG", "")
OUT = os.path.join(ROOT, "artifacts/poc")
DRIVE = os.environ.get("MER_DRIVE", "/content/drive/MyDrive/PhD_PoC")
DO_SMOKE = os.environ.get("MER_SMOKE", "1") == "1"

_ROUTER_LEAVES = ("gate", "router", "gating")
_EXPERT_LEAVES = ("gate_up_proj", "down_proj", "gate_proj", "up_proj")
_EXPERT_MARKER = "experts"   

def _layer_index(name: str):
    m = re.search(r"layers\.(\d+)", name)
    return int(m.group(1)) if m else None

def find_moe_blocks(model, layers=None):
    want = set(layers) if layers is not None else None
    blocks = []
    for name, mod in model.named_modules():
        if not any(cn == "experts" for cn, _ in mod.named_children()):
            continue
        li = _layer_index(name)
        if want is not None and li not in want:
            continue
        key = f"block_{li}" if li is not None else name.replace(".", "_")
        blocks.append((key, mod, li))
    return blocks

@torch.no_grad()
def quant_routed_experts_(block, K: int, group: int) -> dict:
    import torch.nn as nn
    st = {"linear": 0, "expert_mats": 0}
    
    for mname, mod in block.named_modules():
        if _EXPERT_MARKER not in mname:
            continue
        for leaf, child in list(mod.named_children()):
            if not isinstance(child, nn.Linear):
                continue
            if leaf in _ROUTER_LEAVES:
                continue
            if min(child.weight.shape) < 64:
                continue
            kp = quantize_matrix_k(child.weight.data, K, group=group, row_chunk=65536)
            child.weight.data.copy_(kp.dequantize(child.weight.dtype))
            st["linear"] += 1
    
    for pname, p in block.named_parameters():
        if p.dim() == 3 and _EXPERT_MARKER in pname and pname.split(".")[-1] in _EXPERT_LEAVES:
            for e in range(p.shape[0]):
                kp = quantize_matrix_k(p.data[e], K, group=group, row_chunk=65536)
                p.data[e] = kp.dequantize(p.dtype)
                st["expert_mats"] += 1
    return st

def _extract(o):
    return o[0] if isinstance(o, tuple) else o

@torch.no_grad()
def collect_block_io(model, ids, blocks, device):
    store = {k: {"in": [], "out": []} for k, _, _ in blocks}

    def mk(k):
        def hook(m, i, o):
            xin = i[0]
            out = _extract(o)
            store[k]["in"].append(xin.reshape(-1, xin.shape[-1]).bfloat16().cpu())
            store[k]["out"].append(out.reshape(-1, out.shape[-1]).bfloat16().cpu())
        return hook

    handles = [mod.register_forward_hook(mk(k)) for k, mod, _ in blocks]
    try:
        for b in range(0, ids.shape[0], BATCH):
            model(input_ids=ids[b:b + BATCH].to(device), use_cache=False)
    finally:
        for h in handles:
            h.remove()
    return {k: {"x_in": torch.cat(v["in"], 0), "out_fp": torch.cat(v["out"], 0)}
            for k, v in store.items()}

@torch.no_grad()
def block_output(block, x_in, device):
    outs = []
    for s in range(0, x_in.shape[0], CHUNK):
        xb = x_in[s:s + CHUNK].to(device=device, dtype=next(block.parameters()).dtype).unsqueeze(1)  
        ob = _extract(block(xb))
        outs.append(ob.reshape(-1, ob.shape[-1]).bfloat16().cpu())
    return torch.cat(outs, 0)

@torch.no_grad()
def _spectrum(M, max_dim=SVD_MAXDIM):
    n, d = M.shape
    m = min(n, d)
    if m <= max_dim:
        G = (M @ M.transpose(0, 1)) if n <= d else (M.transpose(0, 1) @ M)
        ev = torch.linalg.eigvalsh(G.double())          
        ev = ev.clamp_min(0).flip(0)                     
        return ev, ev.sum(), False
    q = min(max_dim, m - 1)
    _, S, _ = torch.svd_lowrank(M.float(), q=q)
    return (S.double() ** 2), (M.double() ** 2).sum(), True

def _rank_metrics(ev, fro2):
    total = float(fro2)
    if ev.numel() == 0 or total <= 0:
        return 0.0, None, None, 0.0, total
    smax2 = float(ev[0])
    stable = total / smax2 if smax2 > 0 else 0.0
    csum = torch.cumsum(ev, 0)
    csum_last = float(csum[-1])

    def erank(frac):
        thr = frac * total
        if csum_last < thr - 1e-9:
            return None
        i = int(torch.searchsorted(csum, torch.tensor(thr, dtype=csum.dtype, device=csum.device)).item())
        return min(i + 1, int(ev.numel()))

    return stable, erank(0.90), erank(0.99), smax2, total

@torch.no_grad()
def rank_entry(E, S, device, qstat):
    dev = device if torch.cuda.is_available() else "cpu"    
    Ed, Sd = E.to(dev), S.to(dev)

    ev_e, fro_e, ap_e = _spectrum(Ed)
    ev_s, fro_s, ap_s = _spectrum(Sd)
    st_e, e90, e99, smax_e, tot_e = _rank_metrics(ev_e, fro_e)
    st_s, s90, s99, _, tot_s = _rank_metrics(ev_s, fro_s)

    Ec = Ed - Ed.mean(0, keepdim=True)
    ev_ec, fro_ec, _ = _spectrum(Ec)
    st_ec, ec90, _, _, _ = _rank_metrics(ev_ec, fro_ec)

    d, n = int(E.shape[1]), int(E.shape[0])

    def ratio(x):
        return None if x is None else round(x / d, 4)

    return {
        "dmodel": d,
        "n_tokens": n,
        "stable_rank": round(st_e, 3),
        "erank90": e90,
        "erank99": e99,
        "erank90_over_dmodel": ratio(e90),
        "erank99_over_dmodel": ratio(e99),
        "stable_rank_centered": round(st_ec, 3),
        "erank90_centered": ec90,
        "sigma_max": round(smax_e ** 0.5, 6),
        "err_fro": round(tot_e ** 0.5, 6),
        "rel_err_energy": (round((tot_e / tot_s) ** 0.5, 6) if tot_s > 0 else None),
        "approx_svd": bool(ap_e or ap_s),
        "signal_rank": {
            "stable_rank": round(st_s, 3),
            "erank90": s90,
            "erank99": s99,
            "erank90_over_dmodel": ratio(s90),
            "erank99_over_dmodel": ratio(s99),
        },
        "top_sigma": [round(float(x) ** 0.5, 6) for x in ev_e[:16].tolist()],
        "quant": qstat,
    }

def measure_blocks(model, ids, blocks, device):
    io = collect_block_io(model, ids, blocks, device)
    results = {}
    for key, block, li in blocks:
        qstat = quant_routed_experts_(block, K, group=GROUP)
        out_k2 = block_output(block, io[key]["x_in"], device)
        E = io[key]["out_fp"].float() - out_k2.float()
        Sfp = io[key]["out_fp"].float()
        results[key] = rank_entry(E, Sfp, device, qstat)
        results[key]["layer"] = li
        io[key] = None            
        del out_k2, E, Sfp
        r = results[key]
        print("[%s] d=%d n=%d  err stable=%.2f erank90=%s (%.3f d)  signal stable=%.2f "
              "erank90=%s  rel|E|=%.4f  q=%s" % (
                  key, r["dmodel"], r["n_tokens"], r["stable_rank"], r["erank90"],
                  (r["erank90_over_dmodel"] or 0.0), r["signal_rank"]["stable_rank"],
                  r["signal_rank"]["erank90"], (r["rel_err_energy"] or 0.0), qstat), flush=True)
    return results

def load_corpus_ids(tok, n_seq, seqlen):
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="train")
    eos = tok.eos_token_id if tok.eos_token_id is not None else (tok.pad_token_id or 0)
    buf, need = [], n_seq * seqlen
    for i in range(len(ds)):
        text = f"Question: {ds[i]['question']}\nAnswer: {ds[i]['answer']}"
        buf.extend(tok(text, add_special_tokens=False).input_ids)
        buf.append(eos)
        if len(buf) >= need:
            break
    n = min(len(buf) // seqlen, n_seq)
    if n < 1:
        raise RuntimeError(f"corpus too small: {len(buf)} tokens < seqlen {seqlen}")
    return torch.tensor(buf[: n * seqlen], dtype=torch.long).view(n, seqlen)

def load_fp():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, trust_remote_code=True, low_cpu_mem_usage=True)
    model = model.to(dev).eval()
    return model, tok, dev

def smoke():
    from transformers import Qwen2MoeConfig, Qwen2MoeForCausalLM
    torch.manual_seed(0)
    cfg = Qwen2MoeConfig(
        vocab_size=256, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
        moe_intermediate_size=64, shared_expert_intermediate_size=128,
        num_experts=8, num_experts_per_tok=2, max_position_embeddings=64,
    )
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tiny = Qwen2MoeForCausalLM(cfg).to(dev).eval()
    for p in tiny.parameters():
        p.requires_grad = False
    blocks = find_moe_blocks(tiny)
    assert len(blocks) == cfg.num_hidden_layers, f"expected {cfg.num_hidden_layers} MoE blocks, got {len(blocks)}"

    ids = torch.randint(0, cfg.vocab_size, (2, 16), device=dev)
    res = measure_blocks(tiny, ids, blocks, dev)
    for k, r in res.items():
        assert r["quant"]["linear"] >= cfg.num_experts, f"{k}: routed experts not ternarized ({r['quant']})"
        assert r["dmodel"] == cfg.hidden_size
        for f in ("stable_rank", "err_fro", "rel_err_energy"):
            v = r[f]
            assert v is not None and v == v, f"{k}: bad {f}={v}"           
        assert 1.0 <= r["stable_rank"] <= r["dmodel"] + 1e-6, f"{k}: stable_rank out of range {r['stable_rank']}"
        if r["erank90"] is not None:
            assert 1 <= r["erank90"] <= min(r["dmodel"], r["n_tokens"]), f"{k}: erank90 {r['erank90']}"
    del tiny
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"[smoke] OK — hook+rerun+rank path validated on {len(blocks)} tiny MoE blocks.", flush=True)

def _summary(results):
    def med(key, sub=None):
        vals = []
        for r in results.values():
            v = r[sub][key] if sub else r[key]
            if v is not None:
                vals.append(v)
        return round(median(vals), 4) if vals else None
    return {
        "n_blocks": len(results),
        "median_err_stable_rank": med("stable_rank"),
        "median_err_erank90": med("erank90"),
        "median_err_erank90_over_dmodel": med("erank90_over_dmodel"),
        "median_err_erank99_over_dmodel": med("erank99_over_dmodel"),
        "median_err_stable_rank_centered": med("stable_rank_centered"),
        "median_signal_stable_rank": med("stable_rank", sub="signal_rank"),
        "median_signal_erank90_over_dmodel": med("erank90_over_dmodel", sub="signal_rank"),
        "median_rel_err_energy": med("rel_err_energy"),
    }

def main():
    os.makedirs(OUT, exist_ok=True)
    status = os.path.join(OUT, f"MER_STATUS{TAG}.txt")
    open(status, "w").write("RUNNING\n")
    t0 = time.time()
    try:
        if DO_SMOKE:
            smoke()

        model, tok, dev = load_fp()
        print(f"[main] loaded {MODEL} on {dev}", flush=True)
        blocks = find_moe_blocks(model, LAYERS)
        if not blocks:
            raise RuntimeError("no MoE blocks found (module with an `experts` child)")
        print(f"[main] {len(blocks)} MoE blocks: {[k for k, _, _ in blocks]}", flush=True)

        ids = load_corpus_ids(tok, N_CALIB, SEQLEN)
        print(f"[main] calib ids {tuple(ids.shape)} (~{ids.numel()} tokens), K={K}, group={GROUP}", flush=True)

        results = measure_blocks(model, ids, blocks, dev)
        summary = _summary(results)

        out = {
            "_meta": {
                "experiment": "#77 MoE routed-expert error effective-rank in the block-output space",
                "date_utc_note": "stamp after run",
                "model": MODEL, "K": K, "group": GROUP,
                "n_calib": int(ids.shape[0]), "seqlen": int(ids.shape[1]),
                "n_tokens": int(ids.numel()), "device": str(dev),
                "svd_maxdim": SVD_MAXDIM, "layers_subsample": LAYERS,
                "torch": torch.__version__,
                "minutes": round((time.time() - t0) / 60, 2),
            },
            "summary": summary,
            "blocks": results,
        }

        fn = os.path.join(OUT, f"moe_error_rank{TAG}.json")
        with open(fn, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[main] wrote {fn}", flush=True)
        if os.path.isdir(DRIVE):
            dfn = os.path.join(DRIVE, f"moe_error_rank{TAG}.json")
            with open(dfn, "w") as f:
                json.dump(out, f, indent=2)
            print(f"[main] wrote {dfn}", flush=True)

        open(status, "w").write("DONE %.0fs\n%s\n" % (time.time() - t0, json.dumps(summary)))
        print("[main] SUMMARY " + json.dumps(summary, indent=2), flush=True)
    except Exception:
        tb = traceback.format_exc()
        print(tb)
        open(status, "w").write("ERROR\n" + tb)
        raise

if __name__ == "__main__":
    main()
