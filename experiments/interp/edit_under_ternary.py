from __future__ import annotations

import json
import os
import sys
import time
import traceback

ROOT = os.environ.get("POC_ROOT", "/content/PhD-propose")

sys.path[:0] = [ROOT, os.path.join(ROOT, "sasori/src"), os.path.join(ROOT, "sasori/scripts")]

import torch  

MODEL = os.environ.get("EU_MODEL", "gpt2-xl")
NEDITS = int(os.environ.get("EU_NEDITS", "20"))
ALG = os.environ.get("EU_ALG", "ROME").upper()
DATASET = os.environ.get("EU_DATASET", "counterfact")
EASYEDIT_ROOT = os.environ.get("EU_EASYEDIT_ROOT", "./EasyEdit")
NPROBE = int(os.environ.get("EU_NPROBE", "40"))
GROUP = int(os.environ.get("EU_GROUP", "256"))
REF_THRESH = float(os.environ.get("EU_REF_THRESH", "0.9"))
TAG = os.environ.get("EU_TAG", "edit_under_ternary")
_DEV_ENV = os.environ.get("EU_DEVICE", "")
DEVICE = _DEV_ENV or ("cuda" if torch.cuda.is_available() else "cpu")
OUT_JSON = os.environ.get("EU_OUT_JSON") or os.path.join(ROOT, "sasori/bench", f"{TAG}.json")

def _default_hparams_path() -> str:
    short = MODEL.split("/")[-1].lower()
    return os.path.join(EASYEDIT_ROOT, "hparams", ALG, f"{short}.yaml")

HPARAMS = os.environ.get("EU_HPARAMS") or _default_hparams_path()

def ensure_easyeditor() -> tuple[bool, str | None]:
    try:
        import easyeditor  
        return True, None
    except Exception as e0:  
        print("[edit] easyeditor NOT importable; trying "
              "`pip install --break-system-packages easyeditor` ...", file=sys.stderr, flush=True)
        import subprocess
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "--break-system-packages", "easyeditor"],
                check=True,
            )
        except Exception as e1:  
            print(f"[edit] pip install easyeditor FAILED: {e1!r}", file=sys.stderr, flush=True)
        try:
            import easyeditor  
            return True, None
        except Exception as e2:  
            print(
                "[edit] EasyEdit is NOT importable. The PyPI 'easyeditor' wheel may be missing/"
                "incomplete — install FROM SOURCE on the pod:\n"
                "  git clone https://github.com/zjunlp/EasyEdit && cd EasyEdit\n"
                "  pip install -r requirements.txt\n"
                "  export PYTHONPATH=$PWD:$PYTHONPATH\n"
                "  export EU_HPARAMS=$PWD/hparams/ROME/gpt2-xl.yaml\n"
                "and download the edit dataset (counterfact/zsre) -> EU_DATASET=<path>.json",
                file=sys.stderr, flush=True,
            )
            return False, f"{e0!r} / {e2!r}"

def _first(d: dict, *keys, default=None):
    for k in keys:
        if k in d and d[k] not in (None, "", []):
            return d[k]
    return default

def _norm_target(v):
    if isinstance(v, dict):
        return v.get("str") or v.get("value") or ""
    if isinstance(v, (list, tuple)):
        return v[0] if v else ""
    return v if v is not None else ""

def _parse_record(rec: dict) -> dict | None:
    
    if "requested_rewrite" in rec:
        rr = rec["requested_rewrite"]
        subject = rr.get("subject", "")
        prompt_tmpl = rr.get("prompt", "")
        prompt = prompt_tmpl.format(subject) if "{}" in prompt_tmpl else prompt_tmpl
        para = rec.get("paraphrase_prompts") or []
        neigh = rec.get("neighborhood_prompts") or []
        return {
            "prompt": prompt,
            "target_new": _norm_target(rr.get("target_new")),
            "ground_truth": _norm_target(rr.get("target_true")),
            "subject": subject,
            "rephrase_prompt": para[0] if para else None,
            "locality_prompt": neigh[0] if neigh else None,
            "locality_answer": _norm_target(rr.get("target_true")),  
        }
    
    if "src" in rec and ("alt" in rec or "answers" in rec):
        return {
            "prompt": rec.get("src", ""),
            "target_new": _norm_target(_first(rec, "alt", "target_new")),
            "ground_truth": _norm_target(_first(rec, "answers", "answer", "ground_truth")),
            "subject": rec.get("subject", ""),
            "rephrase_prompt": _first(rec, "rephrase", "rephrase_prompt"),
            "locality_prompt": _first(rec, "loc", "locality_prompt"),
            "locality_answer": _norm_target(_first(rec, "loc_ans", "locality_ground_truth")),
        }
    
    if "prompt" in rec and ("target_new" in rec or "alt" in rec):
        loc = rec.get("locality") or {}
        loc_prompt = _first(rec, "locality_prompt") or (
            _first(loc, "neighborhood", "nq") if isinstance(loc, dict) else None)
        return {
            "prompt": rec["prompt"],
            "target_new": _norm_target(_first(rec, "target_new", "alt")),
            "ground_truth": _norm_target(_first(rec, "ground_truth", "target_true")),
            "subject": rec.get("subject", ""),
            "rephrase_prompt": _first(rec, "rephrase_prompt", "rephrase"),
            "locality_prompt": loc_prompt if isinstance(loc_prompt, str) else None,
            "locality_answer": _norm_target(_first(rec, "locality_ground_truth", "loc_ans")),
        }
    return None

def load_dataset_records(spec: str) -> list[dict]:
    path = spec
    if not spec.endswith(".json"):
        for cand in (
            os.path.join(EASYEDIT_ROOT, "data", f"{spec}.json"),
            os.path.join(EASYEDIT_ROOT, "data", spec, f"{spec}.json"),
            os.path.join(EASYEDIT_ROOT, "data", f"{spec}-edit.json"),
        ):
            if os.path.isfile(cand):
                path = cand
                break
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"EU_DATASET={spec!r} did not resolve to a JSON file (tried {path!r}). "
            "Download the counterfact/zsre json on the pod and point EU_DATASET at it.")
    with open(path) as f:
        raw = json.load(f)
    if isinstance(raw, dict):  
        raw = raw.get("data") or next((v for v in raw.values() if isinstance(v, list)), [])
    return list(raw)

def build_requests_and_probes(records: list[dict], n_edit: int, n_probe: int):
    parsed = [p for p in (_parse_record(r) for r in records) if p and p["prompt"] and p["subject"]]
    if len(parsed) < n_edit + 1:
        raise ValueError(f"only {len(parsed)} usable records parsed; need >= {n_edit + 1}")
    requests = parsed[:n_edit]

    probes = [
        {"prompt": p["prompt"], "answer": p["ground_truth"]}
        for p in parsed[n_edit:n_edit + n_probe]
        if p["ground_truth"]
    ]
    return requests, probes

def _import_fakequant():
    
    from exp_moe_distill import fake_quant_k_, _SKIP_QUANT, _ROUTER_LEAVES  
    from sasori.reconstruct import quantize_matrix_k  
    return fake_quant_k_, _SKIP_QUANT, _ROUTER_LEAVES, quantize_matrix_k

@torch.no_grad()
def apply_ternary(model, K: int, group: int) -> dict:
    import torch.nn as nn
    from transformers.pytorch_utils import Conv1D

    fake_quant_k_, _SKIP_QUANT, _ROUTER_LEAVES, quantize_matrix_k = _import_fakequant()
    stats = fake_quant_k_(model, K, group=group)  
    stats["conv1d"] = 0
    for _name, mod in model.named_modules():
        for leaf, child in list(mod.named_children()):
            if not isinstance(child, Conv1D):
                continue
            if leaf in _SKIP_QUANT or leaf in _ROUTER_LEAVES:
                continue
            w = child.weight.data  
            if min(w.shape) < 64:
                continue
            kp = quantize_matrix_k(w.t().contiguous(), K, group=group, row_chunk=65536)
            child.weight.data.copy_(kp.dequantize(w.dtype).t().contiguous())
            stats["conv1d"] += 1
    return stats

def edited_module_names(hparams) -> list[str]:
    tmpl = getattr(hparams, "rewrite_module_tmp", None)
    layers = getattr(hparams, "layers", None)
    if not tmpl or not layers:
        return []
    out = []
    for l in layers:
        try:
            out.append(tmpl.format(l))
        except Exception:  
            out.append(tmpl.replace("{}", str(l)))
    return out

def quant_reaches_edits(model, names: list[str]) -> dict:
    import torch.nn as nn
    from transformers.pytorch_utils import Conv1D

    _, _SKIP_QUANT, _ROUTER_LEAVES, _ = _import_fakequant()
    by_name = dict(model.named_modules())
    covered, detail = 0, {}
    for n in names:
        mod = by_name.get(n)
        if mod is None:
            detail[n] = "MISSING"
            continue
        leaf = n.split(".")[-1]
        is_lin = isinstance(mod, nn.Linear)
        is_conv = isinstance(mod, Conv1D)
        if not (is_lin or is_conv):
            detail[n] = f"UNHANDLED:{type(mod).__name__}"
            continue
        if leaf in _SKIP_QUANT or leaf in _ROUTER_LEAVES or min(tuple(mod.weight.shape)) < 64:
            detail[n] = "SKIPPED_BY_POLICY"
            continue
        detail[n] = "Linear" if is_lin else "Conv1D"
        covered += 1
    return {"n_edited_modules": len(names), "n_covered": covered, "detail": detail}

def _join(prompt: str, target: str) -> str:
    if not target:
        return prompt
    if prompt.endswith((" ", "\n")) or target.startswith((" ", "\n")):
        return prompt + target
    return prompt + " " + target

@torch.no_grad()
def _tf_positions(tok, prompt: str, target: str):
    full = _join(prompt, target)
    p_ids = tok(prompt, return_tensors="pt")["input_ids"]
    f_ids = tok(full, return_tensors="pt")["input_ids"]
    start = p_ids.shape[1]
    end = f_ids.shape[1]
    return f_ids, start, end

@torch.no_grad()
def token_acc(model, tok, prompt: str, target: str, device) -> float | None:
    f_ids, start, end = _tf_positions(tok, prompt, target)
    if end - start <= 0:
        return None
    logits = model(f_ids.to(device)).logits[0]           
    pred = logits[start - 1:end - 1].argmax(-1)           
    labels = f_ids[0, start:end].to(pred.device)
    return (pred == labels).float().mean().item()

@torch.no_grad()
def base_argmax_at(model, tok, prompt: str, target: str, device):
    f_ids, start, end = _tf_positions(tok, prompt, target)
    if end - start <= 0:
        return None
    logits = model(f_ids.to(device)).logits[0]
    pred = logits[start - 1:end - 1].argmax(-1).cpu()
    return {"full_ids": f_ids, "start": start, "end": end, "pred": pred}

@torch.no_grad()
def agreement_at(model, ref, device) -> float:
    logits = model(ref["full_ids"].to(device)).logits[0]
    pred = logits[ref["start"] - 1:ref["end"] - 1].argmax(-1).cpu()
    return (pred == ref["pred"]).float().mean().item()

def score_edit_axes(model, tok, requests, device) -> dict:
    rw, pp = [], []
    for r in requests:
        a = token_acc(model, tok, r["prompt"], r["target_new"], device)
        if a is not None:
            rw.append(a)
        if r.get("rephrase_prompt"):
            b = token_acc(model, tok, r["rephrase_prompt"], r["target_new"], device)
            if b is not None:
                pp.append(b)
    return {
        "rewrite_acc": round(sum(rw) / len(rw), 4) if rw else None,
        "paraphrase_acc": round(sum(pp) / len(pp), 4) if pp else None,
    }

def score_locality(model, loc_refs, device) -> float | None:
    if not loc_refs:
        return None
    xs = [agreement_at(model, ref, device) for ref in loc_refs]
    return round(sum(xs) / len(xs), 4)

def score_intrinsic(model, tok, probe_refs, device) -> float | None:
    if not probe_refs:
        return None
    xs = []
    for ref in probe_refs:
        logits = model(ref["full_ids"].to(device)).logits[0]
        pred = logits[ref["start"] - 1:ref["end"] - 1].argmax(-1).cpu()
        xs.append(float((pred == ref["labels"]).all().item()))
    return round(sum(xs) / len(xs), 4)

def full_variant_scores(model, tok, requests, loc_refs, probe_refs, device) -> dict:
    axes = score_edit_axes(model, tok, requests, device)
    axes["locality"] = score_locality(model, loc_refs, device)
    axes["intrinsic_retention"] = score_intrinsic(model, tok, probe_refs, device)
    return axes

def write_out(results: dict):
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"[edit] wrote {OUT_JSON}", flush=True)
    print(json.dumps(results, indent=2, default=str), flush=True)

def main():
    t0 = time.time()
    results = {
        "_meta": {
            "experiment": "#86 do ROME/MEMIT edits survive ternarization?",
            "model": MODEL, "alg": ALG, "n_edits": NEDITS, "dataset": DATASET,
            "hparams": HPARAMS, "group": GROUP, "ref_thresh": REF_THRESH,
            "device": DEVICE, "n_probe": NPROBE,
        },
        "reference_ok": False,
        "fp_edit": None, "k2_edit": "NEEDS_VALIDATION", "k3_edit": "NEEDS_VALIDATION",
    }

    ok, err = ensure_easyeditor()
    if not ok:
        results["_meta"]["status"] = "NEEDS_VALIDATION: easyeditor unavailable"
        results["_meta"]["easyeditor_error"] = err
        write_out(results)
        return

    if not os.path.isfile(HPARAMS):
        results["_meta"]["status"] = f"NEEDS_VALIDATION: hparams not found ({HPARAMS})"
        write_out(results)
        return

    records = load_dataset_records(DATASET)
    requests, probes = build_requests_and_probes(records, NEDITS, NPROBE)
    results["_meta"]["n_requests_built"] = len(requests)
    results["_meta"]["n_probes_built"] = len(probes)

    from easyeditor import BaseEditor
    if ALG == "MEMIT":
        from easyeditor import MEMITHyperParams as HP
    else:
        from easyeditor import ROMEHyperParams as HP
    hparams = HP.from_hparams(HPARAMS)
    editor = BaseEditor.from_hparams(hparams)
    model, tok = editor.model, editor.tok
    device = next(model.parameters()).device
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    results["_meta"]["hparams_model_name"] = getattr(hparams, "model_name", None)
    results["_meta"]["edited_module_names"] = edited_module_names(hparams)

    model.eval()
    loc_refs = []
    for r in requests:
        if r.get("locality_prompt") and r.get("locality_answer"):
            ref = base_argmax_at(model, tok, r["locality_prompt"], r["locality_answer"], device)
            if ref is not None:
                loc_refs.append(ref)
    probe_refs = []
    for p in probes:
        f_ids, start, end = _tf_positions(tok, p["prompt"], p["answer"])
        if end - start <= 0:
            continue
        logits = model(f_ids.to(device)).logits[0]
        pred = logits[start - 1:end - 1].argmax(-1).cpu()
        labels = f_ids[0, start:end].cpu()
        if bool((pred == labels).all().item()):        
            probe_refs.append({"full_ids": f_ids, "start": start, "end": end, "labels": labels})
    results["_meta"]["n_loc_refs"] = len(loc_refs)
    results["_meta"]["n_base_known_probes"] = len(probe_refs)
    print(f"[edit] base refs: locality={len(loc_refs)} known-probes={len(probe_refs)}", flush=True)

    edit_kwargs = dict(
        prompts=[r["prompt"] for r in requests],
        target_new=[r["target_new"] for r in requests],
        subject=[r["subject"] for r in requests],
        ground_truth=[r["ground_truth"] or "<|endoftext|>" for r in requests],
    )
    try:
        ee_metrics, edited_model, _ = editor.edit(
            **edit_kwargs, sequential_edit=True, keep_original_weight=False, verbose=True)
    except TypeError:  
        try:
            ee_metrics, edited_model, _ = editor.edit(**edit_kwargs, keep_original_weight=False)
        except TypeError:
            ee_metrics, edited_model, _ = editor.edit(**edit_kwargs)
    model = edited_model
    try:  
        results["_easyedit_metrics"] = ee_metrics if isinstance(ee_metrics, list) else str(ee_metrics)
    except Exception:  
        pass

    model.eval()
    results["fp_edit"] = full_variant_scores(model, tok, requests, loc_refs, probe_refs, device)
    print(f"[edit] FP-edited: {results['fp_edit']}", flush=True)

    cov = quant_reaches_edits(model, results["_meta"]["edited_module_names"])
    results["_meta"]["quant_coverage_of_edits"] = cov
    fp_rw = results["fp_edit"].get("rewrite_acc")
    fp_ok = fp_rw is not None and fp_rw >= REF_THRESH
    cov_ok = cov["n_covered"] > 0
    results["reference_ok"] = bool(fp_ok and cov_ok)
    if not results["reference_ok"]:
        reasons = []
        if not fp_ok:
            reasons.append(f"FP rewrite_acc {fp_rw} < {REF_THRESH}")
        if not cov_ok:
            reasons.append("ternarization does NOT reach the edited modules "
                           "(gpt2 Conv1D? wrong hparams? -> would be a FALSE non-collapse)")
        results["_meta"]["gate_failed"] = "; ".join(reasons)
        print(f"[edit] REFERENCE GATE FAILED: {results['_meta']['gate_failed']} "
              "-> ternary arms marked NEEDS_VALIDATION (not computed)", flush=True)
        results["_meta"]["minutes"] = round((time.time() - t0) / 60, 1)
        write_out(results)
        return

    edited_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    for K, key in ((2, "k2_edit"), (3, "k3_edit")):
        try:
            model.load_state_dict(edited_sd)      
            model.to(device)
            qstats = apply_ternary(model, K, GROUP)
            model.eval()
            scores = full_variant_scores(model, tok, requests, loc_refs, probe_refs, device)
            scores["quant"] = qstats
            results[key] = scores
            print(f"[edit] K{K}-edited: {scores}", flush=True)
        except Exception as e:  
            results[key] = {"error": repr(e)}
            print(f"[edit] K{K} FAILED: {e}", flush=True)
            traceback.print_exc()

    results["_meta"]["minutes"] = round((time.time() - t0) / 60, 1)
    write_out(results)

if __name__ == "__main__":
    main()
