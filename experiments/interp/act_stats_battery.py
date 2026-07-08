from __future__ import annotations

import json
import math
import os
import sys
import time
import traceback

_THIS = os.path.abspath(__file__)
ROOT = os.environ.get("POC_ROOT") or os.path.dirname(os.path.dirname(os.path.dirname(_THIS)))
for _p in (ROOT, os.path.join(ROOT, "sasori", "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  

from sasori.reconstruct import quantize_matrix_k  
from sasori.predict_k import (  
    _measure_sqnr,
    _find_decoder_layers,
    _target_linears,
    _PROMPTS,
)

def _envf(name, default):
    return float(os.environ.get(name, default))

def _envi(name, default):
    return int(os.environ.get(name, default))

CFG = dict(
    model=os.environ.get("AS_MODEL", "Qwen/Qwen2.5-1.5B-Instruct"),
    layers=os.environ.get("AS_LAYERS", "auto"),   
    corpus=os.environ.get("AS_CORPUS", "wikitext"),  
    K=_envi("AS_K", 2),
    n=_envi("AS_N", 24),                
    seq_len=_envi("AS_SEQ_LEN", 256),   
    batch=_envi("AS_BATCH", 4),
    group=_envi("AS_GROUP", 256),       
    niter=_envi("AS_NITER", 25),        
    lam=_envf("AS_LAM", 1e-2),
    outlier_k=_envf("AS_OUTLIER_K", 6.0),      
    max_svd_dim=_envi("AS_MAX_SVD_DIM", 1024),  
    stat_cap=_envi("AS_STAT_CAP", 4_000_000),   
    act_svd_rows=_envi("AS_ACT_SVD_ROWS", 20_000),  
    late_frac=_envf("AS_LATE_FRAC", 0.25),      
    sqnr_gamma=_envi("AS_SQNR_GAMMA", 1),       
    energy_lo=_envf("AS_ENERGY_LO", 0.90),
    energy_hi=_envf("AS_ENERGY_HI", 0.99),
    seed=_envi("AS_SEED", 0),
    device=os.environ.get("AS_DEVICE", ""),     
    out=os.environ.get("AS_OUT", ""),           
)

def _subsample_flat(x: torch.Tensor, cap: int, gen: torch.Generator) -> torch.Tensor:
    f = x.reshape(-1).float()
    if f.numel() > cap:
        idx = torch.randint(0, f.numel(), (cap,), generator=gen)
        f = f[idx]
    return f

def excess_kurtosis(x: torch.Tensor, cap: int, gen: torch.Generator) -> float:
    f = _subsample_flat(x, cap, gen)
    if f.numel() < 4:
        return float("nan")
    mu = f.mean()
    d = f - mu
    var = d.pow(2).mean()
    if var <= 0:
        return float("nan")
    return float((d.pow(4).mean() / var.pow(2) - 3.0).item())

def outlier_frac(x: torch.Tensor, k: float, cap: int, gen: torch.Generator) -> float:
    f = _subsample_flat(x, cap, gen)
    if f.numel() == 0:
        return float("nan")
    std = f.std()
    if std <= 0:
        return 0.0
    return float((f.abs() > k * std).float().mean().item())

def massive_ratio(x: torch.Tensor, cap: int, gen: torch.Generator) -> float:
    f = _subsample_flat(x, cap, gen).abs()
    if f.numel() == 0:
        return float("nan")
    med = f.median()
    if med <= 0:
        return float("inf")
    return float((f.max() / med).item())

def effrank_from_sv(sv: torch.Tensor, energy_lo: float, energy_hi: float) -> dict:
    sv = sv.float().clamp_min(0)
    sv = sv[sv > 0]
    n = int(sv.numel())
    if n == 0:
        return {"stable_rank": float("nan"), "erank_entropy": float("nan"),
                "r_energy_lo": 0, "r_energy_hi": 0, "n_sv": 0}
    sv_sorted, _ = torch.sort(sv, descending=True)
    energy = sv_sorted.pow(2)
    total_e = energy.sum()
    cum = torch.cumsum(energy, 0) / total_e
    r_lo = int((cum >= energy_lo).float().argmax().item()) + 1
    r_hi = int((cum >= energy_hi).float().argmax().item()) + 1
    stable_rank = float((total_e / sv_sorted[0].pow(2)).item())
    p = sv_sorted / sv_sorted.sum()
    ent = float(-(p * (p.clamp_min(1e-12)).log()).sum().item())
    erank = math.exp(ent)
    return {
        "stable_rank": stable_rank,
        "erank_entropy": erank,
        "erank_entropy_norm": erank / n,
        "r_energy_lo": r_lo,
        "r_energy_hi": r_hi,
        "r_energy_hi_norm": r_hi / n,
        "n_sv": n,
    }

def _svdvals_capped(M: torch.Tensor, cap: int, gen: torch.Generator) -> tuple[torch.Tensor, bool]:
    Mf = M.float()
    m, n = Mf.shape
    if min(m, n) <= cap:
        return torch.linalg.svdvals(Mf), False

    omega = torch.randn(n, cap, generator=gen, dtype=Mf.dtype).to(Mf.device)
    Y = Mf @ omega
    return torch.linalg.svdvals(Y), True

def build_examples(tok, cfg) -> list:
    corpus = cfg["corpus"]
    n = cfg["n"]
    if corpus == "wikitext":
        from datasets import load_dataset
        stream = os.environ.get("POC_STREAM", "1") == "1"
        ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train",
                          streaming=stream)
        eos = tok.eos_token_id if tok.eos_token_id is not None else 0
        buf, need = [], (n + 1) * cfg["seq_len"]
        for ex in ds:
            t = ex["text"].strip()
            if not t:
                continue
            buf.extend(tok(t, add_special_tokens=False).input_ids)
            buf.append(eos)
            if len(buf) >= need:
                break
        L = cfg["seq_len"]
        nseq = min(n, len(buf) // L)
        return [torch.tensor(buf[i * L:(i + 1) * L], dtype=torch.long).unsqueeze(0)
                for i in range(nseq)]
    if corpus == "gsm8k":
        from datasets import load_dataset
        ds = load_dataset("openai/gsm8k", "main", split="test")
        prompts = [ds[i]["question"] for i in range(min(n, len(ds)))]
    elif corpus in _PROMPTS:
        bank = _PROMPTS[corpus]
        prompts = [bank[i % len(bank)] for i in range(n)]
    else:
        raise ValueError(f"unknown AS_CORPUS {corpus!r}; use wikitext|gsm8k|"
                         + "|".join(_PROMPTS))
    return [tok(p, return_tensors="pt", truncation=True,
               max_length=cfg["seq_len"]).input_ids for p in prompts]

@torch.no_grad()
def collect_activations(model, layer_list, examples, layers, device) -> dict:
    stores = {l: [] for l in layers}

    def mk(l):
        def hook(_m, _i, o):
            h = o[0] if isinstance(o, (tuple, list)) else o
            
            stores[l].append(h.reshape(-1, h.shape[-1]).bfloat16().cpu())
        return hook

    handles = [layer_list[l].register_forward_hook(mk(l)) for l in layers]
    try:
        for ids in examples:
            model(input_ids=ids.to(device), use_cache=False)
    finally:
        for h in handles:
            h.remove()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return {l: torch.cat(s, 0).contiguous() for l, s in stores.items()}

def _resolve_layers(spec: str, n_layers: int) -> list:
    if spec == "all":
        return list(range(n_layers))
    if spec == "auto":
        
        cand = sorted({int(0.2 * n_layers), int(0.5 * n_layers),
                       min(n_layers - 1, int(0.8 * n_layers))})
        return [c for c in cand if 0 <= c < n_layers] or [n_layers - 1]
    return sorted({int(x) for x in spec.split(",") if x.strip() != "" and 0 <= int(x) < n_layers})

def _mean(vals):
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return float(sum(vals) / len(vals)) if vals else float("nan")

@torch.no_grad()
def weight_stats(model, layer_list, layers, cfg, gen) -> list:
    
    by_layer = {l: [] for l in layers}
    lidset = set(layers)
    for name, mod in _target_linears(model):
        
        parts = name.split(".")
        li = None
        for i, p in enumerate(parts[:-1]):
            if p in ("layers", "h", "blocks") and parts[i + 1].isdigit():
                li = int(parts[i + 1])
                break
        if li in lidset:
            by_layer[li].append((name, mod))

    out = []
    K = cfg["K"]
    for l in layers:
        mods = by_layer.get(l, [])
        kurt, ofrac, nrmse, srank, erank, r_hi = [], [], [], [], [], []
        approx_any = False
        for name, mod in mods:
            W = mod.weight.data
            kurt.append(excess_kurtosis(W, cfg["stat_cap"], gen))
            ofrac.append(outlier_frac(W, cfg["outlier_k"], cfg["stat_cap"], gen))
            
            Wf = W.float()
            kp = quantize_matrix_k(Wf, K, group=cfg["group"], niter=cfg["niter"], lam=cfg["lam"])
            E = Wf - kp.dequantize(Wf.dtype)
            nrmse.append(float((E.norm() / Wf.norm().clamp_min(1e-9)).item()))
            sv, approx = _svdvals_capped(Wf, cfg["max_svd_dim"], gen)
            approx_any = approx_any or approx
            er = effrank_from_sv(sv, cfg["energy_lo"], cfg["energy_hi"])
            srank.append(er["stable_rank"])
            erank.append(er["erank_entropy_norm"])
            r_hi.append(er["r_energy_hi"])
        out.append({
            "layer": l, "n_linears": len(mods),
            "w_kurtosis": _mean(kurt),
            "w_outlier_frac": _mean(ofrac),
            "w_nrmse_k%d" % K: _mean(nrmse),
            "w_stable_rank": _mean(srank),
            "w_erank_entropy_norm": _mean(erank),
            "w_r_energy_hi": _mean(r_hi),
            "svd_approx": approx_any,
        })
    return out

@torch.no_grad()
def activation_stats(acts, cfg, gen) -> list:
    out = []
    for l, X in sorted(acts.items()):
        Xf = X.float()
        n_rows = Xf.shape[0]
        
        rows = Xf
        if n_rows > cfg["act_svd_rows"]:
            idx = torch.randint(0, n_rows, (cfg["act_svd_rows"],), generator=gen)
            rows = Xf[idx]
        gram = (rows.T @ rows) / rows.shape[0]          
        eig = torch.linalg.eigvalsh(gram).clamp_min(0)  
        sv = eig.flip(0).sqrt()                          
        er = effrank_from_sv(sv, cfg["energy_lo"], cfg["energy_hi"])
        out.append({
            "layer": l, "n_tokens": int(n_rows), "d": int(Xf.shape[1]),
            "a_kurtosis": excess_kurtosis(Xf, cfg["stat_cap"], gen),
            "a_outlier_frac": outlier_frac(Xf, cfg["outlier_k"], cfg["stat_cap"], gen),
            "a_massive_ratio": massive_ratio(Xf, cfg["stat_cap"], gen),
            "a_stable_rank": er["stable_rank"],
            "a_erank_entropy_norm": er["erank_entropy_norm"],
            "a_r_energy_hi": er["r_energy_hi"],
        })
    return out

def build_feature_vector(w_layers, a_layers, sqnr, K) -> dict:
    feats = {
        "w_kurtosis_mean": _mean([d["w_kurtosis"] for d in w_layers]),
        "w_outlier_frac_mean": _mean([d["w_outlier_frac"] for d in w_layers]),
        "w_nrmse_k%d_mean" % K: _mean([d["w_nrmse_k%d" % K] for d in w_layers]),
        "w_stable_rank_mean": _mean([d["w_stable_rank"] for d in w_layers]),
        "w_erank_entropy_norm_mean": _mean([d["w_erank_entropy_norm"] for d in w_layers]),
        "a_kurtosis_mean": _mean([d["a_kurtosis"] for d in a_layers]),
        "a_outlier_frac_mean": _mean([d["a_outlier_frac"] for d in a_layers]),
        "a_massive_ratio_mean": _mean([d["a_massive_ratio"] for d in a_layers]),
        "a_stable_rank_mean": _mean([d["a_stable_rank"] for d in a_layers]),
        "a_erank_entropy_norm_mean": _mean([d["a_erank_entropy_norm"] for d in a_layers]),
    }
    feats.update({("sqnr_late_k%d_db" % k): v for k, v in sqnr["sqnr_db"].items()})
    if sqnr.get("gamma") is not None:
        feats["sqnr_gamma_db"] = sqnr["gamma"]
    return feats

def run_battery(cfg) -> dict:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(cfg["seed"])
    gen = torch.Generator().manual_seed(cfg["seed"])
    dev = cfg["device"] or ("cuda" if torch.cuda.is_available() else "cpu")

    tok = AutoTokenizer.from_pretrained(cfg["model"], trust_remote_code=True)
    try:  
        model = AutoModelForCausalLM.from_pretrained(
            cfg["model"], dtype=torch.bfloat16, trust_remote_code=True)
    except TypeError:  
        model = AutoModelForCausalLM.from_pretrained(
            cfg["model"], torch_dtype=torch.bfloat16, trust_remote_code=True)
    model.to(dev).eval()

    layer_list = _find_decoder_layers(model)
    n_layers = len(layer_list)
    layers = _resolve_layers(cfg["layers"], n_layers)
    print("model=%s layers=%d selected=%s dev=%s" % (cfg["model"], n_layers, layers, dev),
          flush=True)

    examples = build_examples(tok, cfg)
    ex_dev = [e.to(dev) for e in examples]
    print("corpus=%s examples=%d" % (cfg["corpus"], len(examples)), flush=True)

    w_layers = weight_stats(model, layer_list, layers, cfg, gen)

    acts = collect_activations(model, layer_list, examples, layers, dev)
    a_layers = activation_stats(acts, cfg, gen)
    del acts
    if dev.startswith("cuda"):
        torch.cuda.empty_cache()

    Ks = (cfg["K"], cfg["K"] + 1) if cfg["sqnr_gamma"] else (cfg["K"],)
    sqnr_db, sqnr_meta = _measure_sqnr(model, dev, ex_dev, Ks=Ks,
                                       group=cfg["group"], late_frac=cfg["late_frac"])
    gamma = None
    if cfg["sqnr_gamma"] and (cfg["K"] + 1) in sqnr_db:
        gamma = sqnr_db[cfg["K"] + 1] - sqnr_db[cfg["K"]]  
    sqnr = {"sqnr_db": {int(k): float(v) for k, v in sqnr_db.items()},
            "gamma": gamma, "meta": sqnr_meta}

    features = build_feature_vector(w_layers, a_layers, sqnr, cfg["K"])
    return {
        "model": cfg["model"],
        "config": cfg,
        "n_layers": n_layers,
        "layers": layers,
        "weight_per_layer": w_layers,
        "activation_per_layer": a_layers,
        "sqnr_late": sqnr,
        "features": features,
    }

def _slug(model: str) -> str:
    return model.replace("/", "_").replace(":", "_")

def main():
    cfg = dict(CFG)
    
    for arg in sys.argv[1:]:
        if "=" in arg:
            k, v = arg.split("=", 1)
            if k in cfg:
                cur = cfg[k]
                cfg[k] = type(cur)(v) if isinstance(cur, (int, float)) and not isinstance(cur, bool) else v
    out = cfg["out"] or os.path.join(ROOT, "artifacts", "poc",
                                     "act_stats_%s_%s_K%d.json" % (
                                         _slug(cfg["model"]), cfg["corpus"], cfg["K"]))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    t0 = time.time()
    try:
        res = run_battery(cfg)
        res["wall_s"] = time.time() - t0
        json.dump(res, open(out, "w"), indent=2, default=str)
        f = res["features"]
        print("WROTE %s (%.0fs)" % (out, res["wall_s"]), flush=True)
        print("  w_nrmse_k%d=%.4f w_kurt=%.2f w_outl=%.4f | a_kurt=%.2f a_massive=%.1f "
              "a_erank=%.3f | sqnr_late_k%d=%.2fdB" % (
                  cfg["K"], f.get("w_nrmse_k%d_mean" % cfg["K"], float("nan")),
                  f["w_kurtosis_mean"], f["w_outlier_frac_mean"],
                  f["a_kurtosis_mean"], f["a_massive_ratio_mean"], f["a_erank_entropy_norm_mean"],
                  cfg["K"], f.get("sqnr_late_k%d_db" % cfg["K"], float("nan"))), flush=True)
    except Exception:
        tb = traceback.format_exc()
        print(tb, flush=True)
        raise

if __name__ == "__main__":
    main()
