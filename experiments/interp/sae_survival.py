
from __future__ import annotations

import json
import os
import statistics
import sys
import time
import traceback

ROOT = os.environ.get("POC_ROOT", "/content/PhD-propose")
sys.path[:0] = [ROOT, os.path.join(ROOT, "packages/bitnet_core/src")]

import torch  

from experiments.interp.sae import SAEConfig, BatchTopKSAE  

K = 32                        
DTYPE = torch.bfloat16
OUT = os.path.join(ROOT, "artifacts/poc")
DRIVE = "/content/drive/MyDrive/PhD_PoC"

_SKIP_QUANT = ("lm_head", "embed_tokens")
_ROUTER_LEAVES = ("gate", "router", "gating")
_EXPERT_LEAVES = ("gate_up_proj", "down_proj", "gate_proj", "up_proj")

class Cfg:
    def __init__(self):
        self.sae_pt = os.environ.get("SV_SAE_PT", "sae_ml_floatlm_L15.pt")
        self.layer = int(os.environ.get("SV_LAYER", "15"))
        self.fp_model = os.environ.get("SV_FP_MODEL", "SpectraSuite/FloatLM_2.4B")
        self.quant_models = os.environ.get(
            "SV_QUANT_MODELS", "trilm=hf:SpectraSuite/TriLM_2.4B_Unpacked")
        self.corpus = os.environ.get("SV_CORPUS", "wikitext-103")
        self.limit = int(os.environ.get("SV_LIMIT", "200"))
        self.seq_len = int(os.environ.get("SV_SEQ_LEN", "512"))
        self.batch = int(os.environ.get("SV_BATCH", "8"))
        self.group = int(os.environ.get("SV_QUANT_GROUP", "256"))  

    @property
    def tok_batch(self):
        return self.seq_len * self.batch          

    def _argparse_override(self, argv):
        import argparse
        p = argparse.ArgumentParser(description="SAE feature survival under quantization")
        p.add_argument("--sae-pt"); p.add_argument("--layer", type=int)
        p.add_argument("--fp-model"); p.add_argument("--quant-models")
        p.add_argument("--corpus"); p.add_argument("--limit", type=int)
        p.add_argument("--seq-len", type=int); p.add_argument("--smoke", action="store_true")
        a = p.parse_args(argv)
        for k, v in (("sae_pt", a.sae_pt), ("fp_model", a.fp_model),
                     ("quant_models", a.quant_models), ("corpus", a.corpus)):
            if v is not None:
                setattr(self, k, v)
        for k, v in (("layer", a.layer), ("limit", a.limit), ("seq_len", a.seq_len)):
            if v is not None:
                setattr(self, k, v)
        return a.smoke

def _load_frozen_inner(sae_pt, d_in, device):
    try:
        from experiments.interp.sae_absorption import _load_wrapped_sae
        wrapped = _load_wrapped_sae(sae_pt, 0, d_in, "frozen_fp_sae", device)
        return wrapped.inner.float().eval()
    except Exception as e:                     
        print("note: sae_absorption loader unavailable (%r); dependency-light load" % e,
              flush=True)
        pt = sae_pt if os.path.isabs(sae_pt) else os.path.join(OUT, sae_pt)
        sd = torch.load(pt, map_location=device)
        d_sae = sd["W_dec"].shape[1]
        inner = BatchTopKSAE(SAEConfig(d_in=d_in, d_sae=d_sae, k=K)).to(device).float()
        inner.load_state_dict(sd)
        return inner.eval()

def _fake_quant_k_(model, k_planes, group):
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

def _load_variant(spec, cfg, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if "=" not in spec:
        raise ValueError("bad variant spec %r (want label=KIND:ARG)" % spec)
    label, rhs = spec.split("=", 1)
    if ":" not in rhs:
        raise ValueError("bad variant spec %r (want label=KIND:ARG)" % spec)
    kind, arg = rhs.split(":", 1)
    if kind == "hf":
        tok = AutoTokenizer.from_pretrained(arg)
        model = AutoModelForCausalLM.from_pretrained(arg, torch_dtype=DTYPE).to(device).eval()
        return label, model, tok, {"kind": "hf", "source": arg}
    if kind == "fakequant":
        if "@" in arg:
            kstr, base = arg.split("@", 1)
        else:
            kstr, base = arg, cfg.fp_model
        k_planes = int(kstr)
        tok = AutoTokenizer.from_pretrained(base)
        model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=DTYPE).to(device).eval()
        stats = _fake_quant_k_(model, k_planes, cfg.group)
        return label, model, tok, {"kind": "fakequant", "base": base, "K": k_planes,
                                   "group": cfg.group, "quant_stats": stats}
    raise ValueError("unknown variant kind %r (want hf|fakequant)" % kind)

def _corpus_ids(tok, cfg):
    if cfg.corpus in ("wikitext-103", "wikitext", "wikitext-103-raw-v1"):
        from experiments.interp import sae_loss_recovered as slr
        slr.SEQ_LEN = cfg.seq_len
        slr.BATCH = cfg.batch
        return slr._eval_ids(tok, cfg.limit)
    if cfg.corpus.startswith("hf:"):
        from datasets import load_dataset
        _, ds_name, ds_cfg, split, field = cfg.corpus.split(":", 4)
        ds = load_dataset(ds_name, ds_cfg or None, split=split, streaming=True)
        eos = tok.eos_token_id
        buf, need = [], (cfg.limit + cfg.batch) * cfg.seq_len
        for ex in ds:
            t = (ex[field] or "").strip()
            if not t:
                continue
            buf.extend(tok(t, add_special_tokens=False).input_ids)
            buf.append(eos)
            if len(buf) >= need:
                break
        n = min(cfg.limit, len(buf) // cfg.seq_len)
        if n < 8:
            raise RuntimeError("only %d corpus tokens collected" % len(buf))
        return torch.tensor(buf[: n * cfg.seq_len], dtype=torch.long).view(n, cfg.seq_len)
    raise ValueError("unknown SV_CORPUS %r" % cfg.corpus)

def _collect(model, ids, layer, device, cfg):
    from experiments.interp import run_sae_multilayer as rml
    rml.BATCH = cfg.batch
    xs = rml.collect_multi(model, ids, [layer], ids.numel(), device)
    return xs[layer]

@torch.no_grad()
def survival_stats(inner, X_fp, X_q, device, tok_batch=4096, eps=1e-8):
    assert X_fp.shape == X_q.shape, (tuple(X_fp.shape), tuple(X_q.shape))
    N, d = X_fp.shape
    d_sae = inner.cfg.d_sae
    z = lambda: torch.zeros(d_sae, dtype=torch.float64, device=device)
    s_fp, s_q, ss_fp, ss_q, s_fpq = z(), z(), z(), z(), z()
    fire_fp, fire_q = z(), z()
    colsum_fp = torch.zeros(d, dtype=torch.float64, device=device)
    colsum_q = torch.zeros(d, dtype=torch.float64, device=device)
    sumsq_fp = sumsq_q = sse_fp = sse_q = l0_fp = l0_q = 0.0
    n = 0
    for b in range(0, N, tok_batch):
        xf = X_fp[b: b + tok_batch].to(device).float()
        xq = X_q[b: b + tok_batch].to(device).float()
        xhf, zf, _ = inner(xf)
        xhq, zq, _ = inner(xq)
        zf64, zq64 = zf.double(), zq.double()
        s_fp += zf64.sum(0); s_q += zq64.sum(0)
        ss_fp += (zf64 * zf64).sum(0); ss_q += (zq64 * zq64).sum(0)
        s_fpq += (zf64 * zq64).sum(0)
        fire_fp += (zf > 0).double().sum(0); fire_q += (zq > 0).double().sum(0)
        l0_fp += float((zf > 0).double().sum().item())
        l0_q += float((zq > 0).double().sum().item())
        colsum_fp += xf.double().sum(0); colsum_q += xq.double().sum(0)
        sumsq_fp += float((xf.double() ** 2).sum().item())
        sumsq_q += float((xq.double() ** 2).sum().item())
        sse_fp += float(((xf - xhf).double() ** 2).sum().item())
        sse_q += float(((xq - xhq).double() ** 2).sum().item())
        n += xf.shape[0]
    num = n * s_fpq - s_fp * s_q
    den = torch.sqrt((n * ss_fp - s_fp * s_fp).clamp_min(0.0)
                     * (n * ss_q - s_q * s_q).clamp_min(0.0))
    corr = torch.where(den > eps, num / den.clamp_min(eps),
                       torch.full_like(num, float("nan")))
    fire_fp_r, fire_q_r = fire_fp / n, fire_q / n
    totvar_fp = sumsq_fp - float((colsum_fp * colsum_fp).sum().item()) / n
    totvar_q = sumsq_q - float((colsum_q * colsum_q).sum().item()) / n
    fve_fp = 1.0 - sse_fp / (totvar_fp + 1e-9)
    fve_q = 1.0 - sse_q / (totvar_q + 1e-9)
    alive_fp = fire_fp_r > eps
    silent_q = fire_q_r <= eps
    n_alive = int(alive_fp.sum().item())
    fell_silent = float((alive_fp & silent_q).double().sum().item()) / max(n_alive, 1)
    corr_alive = corr[alive_fp]
    corr_alive = corr_alive[~torch.isnan(corr_alive)]
    ca = corr_alive.cpu().tolist()
    summary = {}
    if ca:
        summary = {"corr_mean": statistics.fmean(ca), "corr_median": statistics.median(ca),
                   "corr_p10": statistics.quantiles(ca, n=10)[0] if len(ca) > 1 else ca[0],
                   "frac_corr_gt_0p9": sum(c > 0.9 for c in ca) / len(ca),
                   "frac_corr_gt_0p5": sum(c > 0.5 for c in ca) / len(ca)}
    per_feature = []
    fpr, fqr, cc = fire_fp_r.cpu().tolist(), fire_q_r.cpu().tolist(), corr.cpu().tolist()
    for j in alive_fp.nonzero().flatten().cpu().tolist():
        cj = cc[j]
        per_feature.append({"feature": int(j),
                            "corr": None if cj != cj else round(cj, 4),   
                            "fire_fp": round(fpr[j], 5), "fire_q": round(fqr[j], 5)})
    return {"n_tokens": n, "n_features": d_sae, "n_alive_fp": n_alive,
            "fell_silent_frac": fell_silent,
            "n_new_firing": int((silent_q.logical_not() & alive_fp.logical_not()).sum().item()),
            "fve_fp": fve_fp, "fve": fve_q, "l0_fp": l0_fp / n, "l0": l0_q / n,
            "survival": summary, "per_feature": per_feature}

@torch.no_grad()
def loss_recovered_cross(model, ids, layer, frozen_inner, device):
    from experiments.interp import sae_loss_recovered as slr
    l_clean = slr._ce(model, ids, device)
    mean_vec = slr._layer_mean(model, ids, layer, device)
    l_recon = slr._ce(model, ids, device, hook_layer=layer,
                      hook_fn=slr._mk_recon_hook(frozen_inner))
    l_abl = slr._ce(model, ids, device, hook_layer=layer,
                    hook_fn=slr._mk_mean_hook(mean_vec))
    denom = l_abl - l_clean
    recovered = (l_abl - l_recon) / denom if denom > 1e-9 else float("nan")
    return {"L_clean": l_clean, "L_recon": l_recon, "L_ablate": l_abl,
            "loss_recovered": recovered, "delta_ce_recon": l_recon - l_clean}

def _assert_tokenizer_parity(fp_tok, var_tok, label):
    a, b = getattr(fp_tok, "vocab_size", None), getattr(var_tok, "vocab_size", None)
    if a is not None and b is not None and a != b:
        raise RuntimeError(
            "variant %r tokenizer vocab_size %s != FP %s; token positions would not align "
            "(survival correlation requires a shared tokenizer)" % (label, b, a))

def smoke():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    d_in, d_sae, n = 64, 256, 4096
    inner = BatchTopKSAE(SAEConfig(d_in=d_in, d_sae=d_sae, k=8)).to(dev).float().eval()
    X_fp = torch.randn(n, d_in)
    X_q = X_fp + 0.05 * torch.randn(n, d_in)                
    r_close = survival_stats(inner, X_fp, X_q, dev, tok_batch=2048)
    r_self = survival_stats(inner, X_fp, X_fp, dev, tok_batch=2048)
    assert 0.0 <= r_self["fell_silent_frac"] <= 1.0
    assert r_self["survival"] and r_self["survival"]["corr_median"] > 0.999, r_self["survival"]
    assert abs(r_self["fve"] - r_self["fve_fp"]) < 1e-6, (r_self["fve"], r_self["fve_fp"])
    assert isinstance(r_close["per_feature"], list)
    
    X_far = torch.randn(n, d_in)
    r_far = survival_stats(inner, X_fp, X_far, dev, tok_batch=2048)
    close_m = r_close["survival"].get("corr_median", 0.0) if r_close["survival"] else 0.0
    far_m = r_far["survival"].get("corr_median", 0.0) if r_far["survival"] else 0.0
    assert close_m > far_m, (close_m, far_m)
    print("[smoke] PASS  self_corr_med=%.4f close_med=%.3f far_med=%.3f fell_silent=%.3f "
          "fve_self=%.3f l0=%.1f n_alive=%d" % (
              r_self["survival"]["corr_median"], close_m, far_m,
              r_close["fell_silent_frac"], r_self["fve"], r_self["l0"], r_self["n_alive_fp"]),
          flush=True)

def main():
    cfg = Cfg()
    do_smoke = cfg._argparse_override(sys.argv[1:]) or os.environ.get("SV_SMOKE") == "1"
    if do_smoke:
        smoke()
        return

    os.makedirs(OUT, exist_ok=True)
    status = os.path.join(OUT, "SAESURV_STATUS.txt")
    open(status, "w").write("RUNNING\n")
    t0 = time.time()
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        dev = "cuda"
        fp_tok = AutoTokenizer.from_pretrained(cfg.fp_model)
        if fp_tok.pad_token is None:
            fp_tok.pad_token = fp_tok.eos_token
        ids = _corpus_ids(fp_tok, cfg)
        print("eval ids:", tuple(ids.shape), "corpus", cfg.corpus, flush=True)

        fp_model = AutoModelForCausalLM.from_pretrained(cfg.fp_model, torch_dtype=DTYPE).to(dev).eval()
        d_in = fp_model.config.hidden_size
        frozen = _load_frozen_inner(cfg.sae_pt, d_in, dev)
        print("frozen SAE: d_in=%d d_sae=%d layer=%d" % (d_in, frozen.cfg.d_sae, cfg.layer),
              flush=True)
        X_fp = _collect(fp_model, ids, cfg.layer, dev, cfg)
        fp_lr = loss_recovered_cross(fp_model, ids, cfg.layer, frozen, dev)
        del fp_model
        torch.cuda.empty_cache()
        print("[fp] collected X_fp %s  self loss_recovered=%.4f" % (
            tuple(X_fp.shape), fp_lr["loss_recovered"]), flush=True)

        results = {}
        
        fp_self = survival_stats(frozen, X_fp, X_fp, dev, tok_batch=cfg.tok_batch)
        fp_self.update(fp_lr)
        fp_self["meta"] = {"kind": "fp_baseline", "source": cfg.fp_model}
        results["fp"] = fp_self

        for spec in [s for s in cfg.quant_models.split(";") if s.strip()]:
            label, model, var_tok, meta = _load_variant(spec.strip(), cfg, dev)
            _assert_tokenizer_parity(fp_tok, var_tok, label)
            X_q = _collect(model, ids, cfg.layer, dev, cfg)
            if X_q.shape[1] != d_in:
                raise RuntimeError("variant %r hidden %d != FP %d" % (label, X_q.shape[1], d_in))
            var_lr = loss_recovered_cross(model, ids, cfg.layer, frozen, dev)
            del model
            torch.cuda.empty_cache()
            surv = survival_stats(frozen, X_fp, X_q, dev, tok_batch=cfg.tok_batch)
            surv.update(var_lr)
            surv["meta"] = meta
            results[label] = surv
            del X_q
            sm = surv.get("survival", {})
            print("[%s] fve=%.3f (fp %.3f) l0=%.1f fell_silent=%.3f corr_med=%.3f "
                  "loss_recovered=%.4f" % (
                      label, surv["fve"], surv["fve_fp"], surv["l0"],
                      surv["fell_silent_frac"], sm.get("corr_median", float("nan")),
                      surv["loss_recovered"]), flush=True)

        out = {"sae_pt": cfg.sae_pt, "layer": cfg.layer, "fp_model": cfg.fp_model,
               "corpus": cfg.corpus, "n_eval_seqs": int(ids.shape[0]), "seq_len": cfg.seq_len,
               "k": K, "d_in": d_in, "d_sae": frozen.cfg.d_sae,
               "variants": results, "wall_s": time.time() - t0}
        out_json = os.path.join(OUT, "sae_survival_L%d.json" % cfg.layer)
        json.dump(out, open(out_json, "w"), indent=2)
        if os.path.isdir(DRIVE):
            json.dump(out, open(os.path.join(DRIVE, "sae_survival_L%d.json" % cfg.layer), "w"),
                      indent=2)
        open(status, "w").write("DONE %.0fs\n" % (time.time() - t0) + json.dumps(
            {lbl: {"fve": round(r["fve"], 4), "fell_silent": round(r["fell_silent_frac"], 4),
                   "loss_recovered": round(r["loss_recovered"], 4)}
             for lbl, r in results.items()}))
        print("SAE SURVIVAL DONE %.0fs -> %s" % (time.time() - t0, out_json), flush=True)
    except Exception:
        tb = traceback.format_exc(); print(tb)
        open(status, "w").write("ERROR\n" + tb)
        raise

if __name__ == "__main__":
    main()
