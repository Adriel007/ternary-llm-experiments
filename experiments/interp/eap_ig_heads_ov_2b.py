
from __future__ import annotations

import json
import os
import sys
import time
import traceback

ROOT = "/content/PhD-propose"
sys.path[:0] = [ROOT, os.path.join(ROOT, "packages/bitnet_core/src")]

import numpy as np  
import torch  

from experiments.interp.eap_ig_2b import (  
    MID, OUT, DRIVE, _node_modules, _cache_run, _build_pairs,
)
from experiments.interp.eap_ig_circuits_2b import _build_ioi  
from experiments.interp.eap_ig_heads_2b import _head_dims, _oproj  
from experiments.interp.eap_ig_heads_path_2b import _sources  

MIN_PER_FAMILY = 5

CAUSAL_HEADS = {"ioi": [(22, 17), (22, 19), (27, 13)], "antonym": [(19, 12), (19, 6), (27, 11)]}

def _final_norm(model):
    norm = model.model.norm
    eps = float(getattr(norm, "variance_epsilon", getattr(norm, "eps", 1e-5)))
    return norm.weight, eps

def _targets(pair):
    if pair["family"] == "ioi":
        return {"io": int(pair["ans_clean"]), "s": int(pair["ans_corrupt"])}
    io, _ = _sources(pair)
    cue = int(pair["clean_ids"][0, io[-1]].item()) if io else int(pair["ans_corrupt"])
    return {"cue": cue, "ans": int(pair["ans_clean"])}

@torch.no_grad()
def _cache_clean(model, ids):
    n = len(model.model.layers)
    store, fr = {}, {}

    def mk(i):
        def pre(m, args):
            store[i] = args[0].detach()
        return pre

    def frpre(m, args):
        fr["x"] = args[0].detach()

    handles = [_oproj(model, i).register_forward_pre_hook(mk(i)) for i in range(n)]
    handles.append(model.model.norm.register_forward_pre_hook(frpre))
    out = model(input_ids=ids)
    for h in handles:
        h.remove()
    return [store[i] for i in range(n)], fr["x"], out.logits

@torch.no_grad()
def _head_write(model, layer, head, hd, x_full_end):
    op = _oproj(model, layer)
    full = op(x_full_end[None, None, :])[0, 0]
    z = x_full_end.clone()
    z[head * hd:(head + 1) * hd] = 0
    minus = op(z[None, None, :])[0, 0]
    return (full - minus).float()

@torch.no_grad()
def _coupling_residual(model, layer, hd, H, x_full_end):
    op = _oproj(model, layer)
    full = op(x_full_end[None, None, :])[0, 0].float()
    acc = torch.zeros_like(full)
    for h in range(H):
        acc += _head_write(model, layer, h, hd, x_full_end)
    return float((acc - full).norm() / (full.norm() + 1e-9))

def _dla(write_h, fr_end, gamma, eps, WU, tid):
    rms = torch.sqrt(fr_end.float().pow(2).mean() + eps)
    normed = gamma.float() * (write_h / rms)
    return float(normed @ WU[tid].float())

@torch.no_grad()
def _ablate_head_logits(model, ids, layer, head, hd, mean_slice, end, tids):
    sl = slice(head * hd, (head + 1) * hd)

    def pre(m, args):
        x = args[0].clone()
        x[:, end, sl] = mean_slice.to(x.dtype)
        return (x,) + tuple(args[1:])

    h = _oproj(model, layer).register_forward_pre_hook(pre)
    out = model(input_ids=ids)
    h.remove()
    lg = out.logits[0, end].float()
    return {k: float(lg[t]) for k, t in tids.items()}

def _solved(model, nodes, p):
    _, Lc = _cache_run(model, nodes, p["clean_ids"], p["ans_clean"], p["ans_corrupt"])
    _, Lk = _cache_run(model, nodes, p["corrupt_ids"], p["ans_clean"], p["ans_corrupt"])
    p["L_clean"], p["L_corrupt"] = Lc, Lk
    return Lc > 0 and Lk < 0

def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    status = os.path.join(OUT, "EAPIG6_STATUS.txt")
    open(status, "w").write("LOADING\n")
    t0 = time.time()
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        tok = AutoTokenizer.from_pretrained(MID)
        model = AutoModelForCausalLM.from_pretrained(MID, dtype=torch.bfloat16).to(dev).eval()
        model.requires_grad_(False)
        n_layers = len(model.model.layers)
        nodes = _node_modules(model)
        H, hd = _head_dims(model)
        gamma, eps = _final_norm(model)
        WU = model.lm_head.weight  
        print("loaded", MID, "| layers", n_layers, "| query-heads", H, "| head_dim", hd,
              "| final-eps", eps, "| dev", dev)

        raw = _build_ioi(tok) + [p for p in _build_pairs(tok) if p["family"] == "antonym"]
        for p in raw:
            p["clean_ids"] = p["clean_ids"].to(dev)
            p["corrupt_ids"] = p["corrupt_ids"].to(dev)
        fam_pairs = {"ioi": [], "antonym": []}
        for p in raw:
            if _solved(model, nodes, p):
                fam_pairs[p["family"]].append(p)
        for fam, ps in fam_pairs.items():
            print("family %-8s: %d solved" % (fam, len(ps)))
            assert len(ps) >= MIN_PER_FAMILY, f"family {fam}: only {len(ps)} pairs (<{MIN_PER_FAMILY})"

        open(status, "w").write("SELFTEST\n")
        p0 = fam_pairs["ioi"][0]
        co0, _, _ = _cache_clean(model, p0["clean_ids"])
        end0 = p0["clean_ids"].shape[1] - 1
        res = _coupling_residual(model, 22, hd, H, co0[22][0, end0])
        assert res < 0.05, f"o_proj per-head decomposition unfaithful: coupling residual {res:.4f} (>5%)"
        print("SELFTEST OK | layer 22 o_proj coupling residual = %.4f (additive head decomposition valid)" % res)
        open(status, "w").write("RUNNING\n")

        out = {"config": {"model": MID, "n_layers": n_layers, "n_query_heads": H, "head_dim": hd,
                          "coupling_residual_L22": res, "position": "end",
                          "families": {f: len(ps) for f, ps in fam_pairs.items()},
                          "causal_heads": CAUSAL_HEADS},
               "families": {}}

        for fam, ps in fam_pairs.items():
            heads = CAUSAL_HEADS[fam]
            
            cache = []
            for p in ps:
                co, fr, logits = _cache_clean(model, p["clean_ids"])
                end = p["clean_ids"].shape[1] - 1
                tids = _targets(p)
                clean_lg = {k: float(logits[0, end, t]) for k, t in tids.items()}
                cache.append({"co": co, "fr_end": fr[0, end].float(), "end": end,
                              "tids": tids, "clean_lg": clean_lg, "ids": p["clean_ids"]})
            
            mean_slice = {}
            for (L, h) in heads:
                sl = slice(h * hd, (h + 1) * hd)
                mean_slice[(L, h)] = torch.stack([c["co"][L][0, c["end"], sl] for c in cache]).mean(0)

            res_heads = {}
            for (L, h) in heads:
                dla = {k: [] for k in cache[0]["tids"]}
                dabl = {k: [] for k in cache[0]["tids"]}
                for c in cache:
                    w = _head_write(model, L, h, hd, c["co"][L][0, c["end"]])
                    for k, t in c["tids"].items():
                        dla[k].append(_dla(w, c["fr_end"], gamma, eps, WU, t))
                    abl = _ablate_head_logits(model, c["ids"], L, h, hd, mean_slice[(L, h)], c["end"], c["tids"])
                    for k in c["tids"]:
                        dabl[k].append(abl[k] - c["clean_lg"][k])
                key = "L%dH%d" % (L, h)
                res_heads[key] = {
                    "dla": {k: float(np.mean(v)) for k, v in dla.items()},
                    "dla_sem": {k: float(np.std(v) / np.sqrt(len(v))) for k, v in dla.items()},
                    "d_logit_meanabl": {k: float(np.mean(v)) for k, v in dabl.items()},
                    "d_logit_meanabl_sem": {k: float(np.std(v) / np.sqrt(len(v))) for k, v in dabl.items()},
                }
                print("  >> %s %s | DLA %s | meanabl Δlogit %s" % (
                    fam, key,
                    {k: round(res_heads[key]["dla"][k], 3) for k in res_heads[key]["dla"]},
                    {k: round(res_heads[key]["d_logit_meanabl"][k], 3) for k in res_heads[key]["d_logit_meanabl"]}))
            out["families"][fam] = {"n_pairs": len(ps), "heads": res_heads}

        out["wall_s"] = time.time() - t0
        path = os.path.join(OUT, "eap_ig_heads_ov_2b.json")
        json.dump(out, open(path, "w"), indent=2)
        if os.path.isdir(DRIVE):
            try:
                json.dump(out, open(os.path.join(DRIVE, "eap_ig_heads_ov_2b.json"), "w"), indent=2)
            except Exception as e:
                print("drive save failed:", e)
        open(status, "w").write("DONE %.0fs\n" % (time.time() - t0))
        print("DONE %.1fs" % (time.time() - t0))
    except Exception:
        tb = traceback.format_exc()
        print(tb)
        open(status, "w").write("ERROR\n" + tb)
        raise

if __name__ == "__main__":
    main()
