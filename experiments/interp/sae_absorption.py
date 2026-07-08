
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

from sae_bench.evals.absorption.eval_config import AbsorptionEvalConfig  
from sae_bench.evals.absorption.k_sparse_probing import run_k_sparse_probing_experiment  
from sae_bench.evals.absorption.feature_absorption import run_feature_absortion_experiment  
from sae_bench.evals.absorption.main import _aggregate_results_df  

OUT = os.path.join(ROOT, "artifacts/poc")
DRIVE = "/content/drive/MyDrive/PhD_PoC"
DTYPE = torch.bfloat16

SPECS = {  
    "ternary":        ("microsoft/bitnet-b1.58-2B-4T-bf16", 15, "sae_ml_L15.pt"),
    "gemma_shim":     ("google/gemma-2-2b",                 13, "sae_ml_gemma_L13.pt"),
    "gemma_official": ("google/gemma-2-2b",                 13, "sae_ml_gemma_L13.pt"),

    
    "ternary_L6":     ("microsoft/bitnet-b1.58-2B-4T-bf16",  6, "sae_ml_L6.pt"),
    "ternary_L23":    ("microsoft/bitnet-b1.58-2B-4T-bf16", 23, "sae_ml_L23.pt"),
    "gemma_L5":       ("google/gemma-2-2b",                  5, "sae_ml_gemma_L5.pt"),
    "gemma_L20":      ("google/gemma-2-2b",                 20, "sae_ml_gemma_L20.pt"),

    

    "trilm_L6":       ("SpectraSuite/TriLM_2.4B_Unpacked",  6, "sae_ml_trilm_L6.pt"),
    "trilm_L15":      ("SpectraSuite/TriLM_2.4B_Unpacked", 15, "sae_ml_trilm_L15.pt"),
    "trilm_L23":      ("SpectraSuite/TriLM_2.4B_Unpacked", 23, "sae_ml_trilm_L23.pt"),
    "floatlm_L6":     ("SpectraSuite/FloatLM_2.4B",  6, "sae_ml_floatlm_L6.pt"),
    "floatlm_L15":    ("SpectraSuite/FloatLM_2.4B", 15, "sae_ml_floatlm_L15.pt"),
    "floatlm_L23":    ("SpectraSuite/FloatLM_2.4B", 23, "sae_ml_floatlm_L23.pt"),
}

class _Cfg:
    def __init__(self, model_name, d_model, device):
        self.model_name = model_name
        self.d_model = d_model
        self.device = device

class _StopForward(Exception):
    pass

class HookedTransformerShim:

    def __init__(self, hf_model, tokenizer, d_model, model_name, device):
        self.model = hf_model
        self.tokenizer = tokenizer
        self.cfg = _Cfg(model_name, d_model, device)
        self._device = device

    def to_tokens(self, text):
        enc = self.tokenizer(text, return_tensors="pt", add_special_tokens=True)
        return enc["input_ids"].to(self._device)

    @torch.no_grad()
    def run_with_cache(self, prompts, names_filter):
        hook_point = names_filter[0] if isinstance(names_filter, (list, tuple)) else names_filter
        layer = int(hook_point.split(".")[1])
        enc = self.tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=True)
        ids = enc["input_ids"].to(self._device)

        if enc["attention_mask"].min().item() == 0:
            raise RuntimeError("ragged batch in run_with_cache (prompts not equal length)")
        store = {}

        def hook(m, i, o):
            store["a"] = o[0] if isinstance(o, tuple) else o
            raise _StopForward

        h = self.model.model.layers[layer].register_forward_hook(hook)
        try:
            self.model(input_ids=ids, use_cache=False)
        except _StopForward:
            pass
        finally:
            h.remove()
        return None, {hook_point: store["a"]}

class _SAECfg:
    def __init__(self, d_in, d_sae, hook_layer, model_name):
        self.d_in = d_in
        self.d_sae = d_sae
        self.hook_layer = hook_layer
        self.hook_name = f"blocks.{hook_layer}.hook_resid_post"
        self.model_name = model_name

    def to_dict(self):
        return {"d_in": self.d_in, "d_sae": self.d_sae, "hook_layer": self.hook_layer,
                "hook_name": self.hook_name, "model_name": self.model_name,
                "architecture": "batch_topk", "inference": "per_token_topk"}

class WrappedSAE(torch.nn.Module):

    def __init__(self, inner: BatchTopKSAE, hook_layer, model_name):
        super().__init__()
        self.inner = inner
        self.k = inner.cfg.k
        self.cfg = _SAECfg(inner.cfg.d_in, inner.cfg.d_sae, hook_layer, model_name)

    @property
    def device(self):
        return self.inner.b_dec.device

    @property
    def dtype(self):
        return self.inner.b_dec.dtype

    @property
    def W_dec(self):                 
        return self.inner.W_dec.t()

    @property
    def W_enc(self):                 
        return self.inner.W_enc.t()

    def encode(self, x):
        pre = self.inner.pre_acts(x.to(self.dtype))
        acts = torch.relu(pre)
        topv, topi = acts.topk(self.k, dim=-1)
        z = torch.zeros_like(acts).scatter_(-1, topi, topv)
        return z

def _load_wrapped_sae(sae_pt, layer, d_in, model_name, device):
    pt = os.path.join(OUT, sae_pt)
    sd = torch.load(pt, map_location=device)
    d_sae = sd["W_dec"].shape[1]
    inner = BatchTopKSAE(SAEConfig(d_in=d_in, d_sae=d_sae, k=32)).to(device).to(DTYPE)
    inner.load_state_dict(sd)
    inner.eval()
    return WrappedSAE(inner, layer, model_name).to(device)

def _absorption_scores(model, sae, layer, cfg, sae_name, work_dir):
    from pathlib import Path
    ks_dir = Path(work_dir) / "ksparse"
    fa_dir = Path(work_dir) / "feat_abs"
    probes_dir = Path(work_dir) / "probes"
    ks = run_k_sparse_probing_experiment(
        model=model, sae=sae, layer=layer, sae_name=sae_name, force=False,
        max_k_value=cfg.max_k_value, f1_jump_threshold=cfg.f1_jump_threshold,
        prompt_template=cfg.prompt_template, prompt_token_pos=cfg.prompt_token_pos,
        device=str(sae.device), k_sparse_probe_l1_decay=cfg.k_sparse_probe_l1_decay,
        k_sparse_probe_batch_size=cfg.k_sparse_probe_batch_size,
        k_sparse_probe_num_epochs=cfg.k_sparse_probe_num_epochs,
        eval_batch_size=cfg.eval_k_sparse_probe_batch_size,
        precalc_k_sparse_probe_sae_acts=cfg.precalc_k_sparse_probe_sae_acts,
        experiment_dir=ks_dir, probes_dir=probes_dir)
    n_gt = int((ks["f1_probe"] > cfg.min_GT_probe_f1).sum())
    if n_gt < cfg.min_feats_for_eval:
        raise RuntimeError(f"insufficient first-letter features: {n_gt} < {cfg.min_feats_for_eval}")
    raw = run_feature_absortion_experiment(
        model=model, sae=sae, layer=layer, sae_name=sae_name, force=False,
        max_k_value=cfg.max_k_value, feature_split_f1_jump_threshold=cfg.f1_jump_threshold,
        prompt_template=cfg.prompt_template, prompt_token_pos=cfg.prompt_token_pos,
        batch_size=cfg.llm_batch_size, device=str(sae.device),
        experiment_dir=fa_dir, sparse_probing_experiment_dir=ks_dir, probes_dir=probes_dir)
    agg = _aggregate_results_df(raw)
    maf, far, nsf, f1s = [], [], [], []
    for _, row in agg.iterrows():
        f1 = ks[ks["letter"] == row["letter"]]["f1_probe"].item()
        if f1 > cfg.min_GT_probe_f1:
            maf.append(row["mean_absorption_fraction"])
            far.append(row["full_absorption_rate"])
            nsf.append(row["num_split_feats"])
            f1s.append(f1)               
    return {"n_letters_scored": len(maf), "n_gt_feats": n_gt,
            "mean_absorption_fraction": statistics.mean(maf),
            "mean_full_absorption_rate": statistics.mean(far),
            "mean_num_split_features": statistics.mean(nsf),

            "mean_gt_probe_f1": statistics.mean(f1s) if f1s else float("nan"),
            "min_gt_probe_f1_threshold": cfg.min_GT_probe_f1,
            "per_letter_absorption_fraction": maf}

def main():
    mode = sys.argv[1]
    if mode in SPECS:
        hf_id, layer, sae_pt = SPECS[mode]
    else:
        
        import re as _re
        _m = _re.match(r"(trilm|floatlm)(D|S\d)?_L(\d+)$", mode)
        _hf = {"trilm": "SpectraSuite/TriLM_2.4B_Unpacked", "floatlm": "SpectraSuite/FloatLM_2.4B"}
        _fam, _suf, _L = _m.group(1), _m.group(2) or "", int(_m.group(3))
        hf_id, layer, sae_pt = _hf[_fam], _L, f"sae_ml_{_fam}{_suf}_L{_L}.pt"
    os.makedirs(OUT, exist_ok=True)
    status = os.path.join(OUT, "ABS_STATUS_%s.txt" % mode)
    open(status, "w").write("RUNNING\n")
    t0 = time.time()
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        dev = "cuda"
        tok = AutoTokenizer.from_pretrained(hf_id)
        if tok.pad_token is None:          
            tok.pad_token = tok.eos_token
        cfg = AbsorptionEvalConfig(model_name=hf_id)
        cfg.llm_dtype = "bfloat16"

        cfg.llm_batch_size = 48
        work_dir = "/content/abs/%s" % mode   

        if mode == "gemma_official":
            from transformer_lens import HookedTransformer
            model = HookedTransformer.from_pretrained_no_processing(
                "gemma-2-2b", device=dev, dtype=DTYPE)
            d_in = model.cfg.d_model
        else:
            hf = AutoModelForCausalLM.from_pretrained(hf_id, dtype=DTYPE).to(dev).eval()
            d_in = hf.config.hidden_size
            model = HookedTransformerShim(hf, tok, d_in, hf_id, dev)

        sae = _load_wrapped_sae(sae_pt, layer, d_in, hf_id, dev)
        print(f"[{mode}] model+sae ready (layer {layer}, d_in {d_in}, d_sae {sae.cfg.d_sae})", flush=True)
        res = _absorption_scores(model, sae, layer, cfg, f"{mode}", work_dir)
        res.update({"mode": mode, "model": hf_id, "layer": layer, "dtype": "bfloat16",
                    "wall_s": time.time() - t0})
        print(f"[{mode}] mean_absorption_fraction={res['mean_absorption_fraction']:.4f} "
              f"full={res['mean_full_absorption_rate']:.4f} over {res['n_letters_scored']} letters", flush=True)
        out_json = os.path.join(OUT, "sae_absorption_%s.json" % mode)
        json.dump(res, open(out_json, "w"), indent=2)
        if os.path.isdir(DRIVE):
            json.dump(res, open(os.path.join(DRIVE, "sae_absorption_%s.json" % mode), "w"), indent=2)
        open(status, "w").write("DONE %.0fs\n" % (time.time() - t0) + json.dumps(
            {k: res[k] for k in ("mean_absorption_fraction", "mean_full_absorption_rate", "n_letters_scored")}))
    except Exception:
        tb = traceback.format_exc(); print(tb)
        open(status, "w").write("ERROR\n" + tb)
        raise

if __name__ == "__main__":
    main()
