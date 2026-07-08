
from __future__ import annotations

import json
import os
import sys
import time
import traceback

ROOT = os.environ.get("POC_ROOT", "/content/PhD-propose")
sys.path[:0] = [ROOT, os.path.join(ROOT, "packages/bitnet_core/src")]

import torch  

OUT = os.path.join(ROOT, "artifacts/poc")
DRIVE = "/content/drive/MyDrive/PhD_PoC"
DTYPE = torch.bfloat16

DEFAULT_MODELS = ("SpectraSuite/TriLM_99M_Unpacked",
                  "SpectraSuite/TriLM_560M_Unpacked",
                  "SpectraSuite/TriLM_1.1B_Unpacked")

class Cfg:
    def __init__(self):
        env_models = os.environ.get("ABS_MODELS")
        self.models = [m.strip() for m in env_models.split(",") if m.strip()]            if env_models else list(DEFAULT_MODELS)
        self.layer_frac = float(os.environ.get("ABS_LAYER_FRAC", "0.5"))
        self.sae_steps = int(os.environ.get("ABS_SAE_STEPS", "4000"))
        self.d_sae = int(os.environ["ABS_DSAE"]) if os.environ.get("ABS_DSAE") else None
        self.k = int(os.environ.get("ABS_K", "32"))
        self.n_vec = int(os.environ.get("ABS_NVEC", "400000"))
        self.batch = int(os.environ.get("ABS_BATCH", "8"))
        self.seed = int(os.environ.get("ABS_SEED", "0"))
        self.out_json = os.environ.get("ABS_OUT_JSON",
                                       os.path.join(OUT, "absorption_ladder.json"))

    def to_dict(self):
        return {"models": self.models, "layer_frac": self.layer_frac,
                "sae_steps": self.sae_steps,
                "d_sae": self.d_sae if self.d_sae is not None else "auto(6.4x expansion)",
                "k": self.k, "n_vec": self.n_vec, "batch": self.batch, "seed": self.seed,
                "out_json": self.out_json}

def _mid_layer(num_layers, frac):
    return max(0, min(num_layers - 1, int(round(frac * num_layers))))

def _safe_tag(hf_id):
    return hf_id.replace("/", "__").replace(".", "p")

def _train_sae_for_model(rml, model, tok, layer, cfg, dev):
    
    rml.SEQ_LEN = getattr(rml, "SEQ_LEN", 512)
    rml.BATCH = cfg.batch
    rml.K = cfg.k
    rml.D_SAE = cfg.d_sae          
    os.environ["SAE_SEED"] = str(cfg.seed)   
    ids = rml._load_corpus_ids(tok, cfg.n_vec)
    Xs = rml.collect_multi(model, ids, [layer], cfg.n_vec, dev)
    X = Xs[layer]
    if X.shape[0] < 4000 or X.shape[1] < 64:
        raise RuntimeError("bad activation collect for SAE: shape %s" % (tuple(X.shape),))
    sae, r = rml._run_layer(layer, X, ids, tok, dev, steps=cfg.sae_steps, do_dash=False)
    del Xs, X
    torch.cuda.empty_cache()
    return sae, r

def _wrap_sae_for_absorption(sae_abs, sae, layer, d_in, hf_id, cfg, dev):
    tag = _safe_tag(hf_id)
    pt_name = "sae_absladder_%s_L%d.pt" % (tag, layer)
    os.makedirs(OUT, exist_ok=True)
    torch.save(sae.state_dict(), os.path.join(OUT, pt_name))
    if os.path.isdir(DRIVE):
        torch.save(sae.state_dict(), os.path.join(DRIVE, pt_name))
    wrapped = sae_abs._load_wrapped_sae(pt_name, layer, d_in, hf_id, dev)
    if cfg.k != 32:
        wrapped.k = cfg.k
        wrapped.inner.cfg.k = cfg.k
    return wrapped, pt_name

def _run_one(hf_id, cfg, rml, sae_abs, dev):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sae_bench.evals.absorption.eval_config import AbsorptionEvalConfig
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(hf_id)
    if tok.pad_token is None:               
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(hf_id, torch_dtype=DTYPE).to(dev).eval()
    d_in = model.config.hidden_size
    num_layers = model.config.num_hidden_layers
    n_params = int(sum(p.numel() for p in model.parameters()))
    layer = _mid_layer(num_layers, cfg.layer_frac)
    print("[%s] loaded: d_in=%d layers=%d -> mid L%d, params=%.3gM" % (
        hf_id, d_in, num_layers, layer, n_params / 1e6), flush=True)

    sae, tr = _train_sae_for_model(rml, model, tok, layer, cfg, dev)
    metrics = tr["metrics"]
    print("[%s] SAE trained L%d: fve=%.3f l0=%.1f dead=%d d_sae=%d" % (
        hf_id, layer, metrics["fve"], metrics["l0"], metrics["dead"], metrics["d_sae"]),
        flush=True)

    wrapped, pt_name = _wrap_sae_for_absorption(sae_abs, sae, layer, d_in, hf_id, cfg, dev)
    del sae
    torch.cuda.empty_cache()
    acfg = AbsorptionEvalConfig(model_name=hf_id)
    acfg.llm_dtype = "bfloat16"
    acfg.llm_batch_size = 48
    shim = sae_abs.HookedTransformerShim(model, tok, d_in, hf_id, dev)
    work_dir = "/content/abs_ladder/%s" % _safe_tag(hf_id)
    abs_res = sae_abs._absorption_scores(shim, wrapped, layer, acfg,
                                         "absladder_%s" % _safe_tag(hf_id), work_dir)
    del model
    torch.cuda.empty_cache()

    row = {"model": hf_id, "n_params": n_params, "num_layers": num_layers,
           "d_in": d_in, "layer": layer, "layer_frac": cfg.layer_frac,
           
           "absorption": abs_res["mean_absorption_fraction"],
           "gt_probe_f1": abs_res.get("mean_gt_probe_f1"),
           "fve": metrics["fve"], "n_dead": metrics["dead"],
           
           "full_absorption_rate": abs_res["mean_full_absorption_rate"],
           "mean_num_split_features": abs_res["mean_num_split_features"],
           "n_letters_scored": abs_res["n_letters_scored"],
           "n_gt_feats": abs_res["n_gt_feats"],
           "min_gt_probe_f1_threshold": abs_res.get("min_gt_probe_f1_threshold"),
           "per_letter_absorption_fraction": abs_res["per_letter_absorption_fraction"],
           "l0": metrics["l0"], "d_sae": metrics["d_sae"], "k": cfg.k,
           "sae_pt": pt_name, "wall_s": round(time.time() - t0, 1), "ok": True}
    print("[%s] absorption=%.4f (over %d letters, gt_f1=%.3f) fve=%.3f dead=%d" % (
        hf_id, row["absorption"], row["n_letters_scored"],
        row["gt_probe_f1"] if row["gt_probe_f1"] is not None else float("nan"),
        row["fve"], row["n_dead"]), flush=True)
    return row

def main():
    cfg = Cfg()
    os.makedirs(OUT, exist_ok=True)
    status = os.path.join(OUT, "ABSLADDER_STATUS.txt")
    open(status, "w").write("RUNNING\n" + json.dumps(cfg.to_dict()) + "\n")
    t0 = time.time()

    import experiments.interp.run_sae_multilayer as rml
    import experiments.interp.sae_absorption as sae_abs
    dev = "cuda"

    rows = []
    for hf_id in cfg.models:
        try:
            rows.append(_run_one(hf_id, cfg, rml, sae_abs, dev))
        except Exception:
            tb = traceback.format_exc()
            print("[%s] FAILED\n%s" % (hf_id, tb), flush=True)
            
            rows.append({"model": hf_id, "ok": False, "error": tb.splitlines()[-1],
                         "traceback": tb})
            torch.cuda.empty_cache()

    n_ok = sum(1 for r in rows if r.get("ok"))
    out = {
        "_meta": {
            "goal": "SAE feature-absorption across the Spectra ternary scale ladder "
                    "(interp-scaling curve extending Paper G / #53).",
            "reused_harness": {
                "sae_training": "experiments.interp.run_sae_multilayer "
                                "(_load_corpus_ids / collect_multi / _run_layer -> "
                                "experiments.interp.sae.train_sae)",
                "absorption_metric": "experiments.interp.sae_absorption._absorption_scores "
                                     "(official SAEBench, via validated HookedTransformerShim; "
                                     "WrappedSAE / _load_wrapped_sae / HookedTransformerShim)",
            },
            "caveat": "SINGLE SEED, ONE LAYER per model, absorption on SMALL/UNTUNED SAEs "
                      "(fixed ~6.4x expansion, K=%d, %d steps — not SAEBench-tuned). Absolute "
                      "values are NOT comparable to tuned-SAE literature numbers; the signal is "
                      "the TREND ACROSS SCALE within this fixed protocol, not the absolute "
                      "per-model value. A tiny size may fail to yield enough first-letter GT "
                      "features -> recorded as an error row, never a fabricated 0."
                      % (cfg.k, cfg.sae_steps),
            "corpus": "wikitext-103-raw-v1 (train), same as Paper G SAE training",
            "dtype": "bfloat16",
            "config": cfg.to_dict(),
            "n_models": len(cfg.models), "n_ok": n_ok,
            "wall_s": round(time.time() - t0, 1),
        },
        "rows": rows,
    }
    json.dump(out, open(cfg.out_json, "w"), indent=2)
    if os.path.isdir(DRIVE):
        json.dump(out, open(os.path.join(DRIVE, os.path.basename(cfg.out_json)), "w"), indent=2)
    open(status, "w").write("DONE %.0fs  %d/%d ok\n" % (time.time() - t0, n_ok, len(cfg.models)))
    print("ABSORPTION LADDER DONE %.0fs  %d/%d ok -> %s" % (
        time.time() - t0, n_ok, len(cfg.models), cfg.out_json), flush=True)
    if n_ok == 0:
        raise SystemExit("no model produced an absorption score (all rows failed)")

if __name__ == "__main__":
    main()
