
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
import torch.nn.functional as F  

MID = "microsoft/bitnet-b1.58-2B-4T-bf16"
OUT = os.path.join(ROOT, "artifacts/poc")
DRIVE = "/content/drive/MyDrive/PhD_PoC"

PROMPTS = [
    "The capital of France is",
    "The capital of Japan is",
    "The opposite of hot is",
    "The opposite of up is",
    "Water is made of hydrogen and",
    "The sky on a clear day is the color",
    "Two plus two equals",
    "The first president of the United States was George",
    "A baby dog is called a",
    "The sun rises in the",
    "Roses are red, violets are",
    "The chemical symbol for gold is",
    "Monday, Tuesday, Wednesday,",
    "She picked up the keys and opened the",
    "The Earth orbits the",
    "In winter the weather is usually very",
    "He was hungry so he decided to eat some",
    "The largest planet in our solar system is",
    "To unlock the door she used a",
    "The cat sat on the",
]

def _find_norm_and_head(model):
    base = model.model
    norm = getattr(base, "norm", None) or getattr(base, "final_layernorm", None)        or getattr(base, "ln_f", None)
    head = model.get_output_embeddings()
    if norm is None or head is None:
        raise RuntimeError("could not locate final norm / lm_head on the model")
    return norm, head

@torch.no_grad()
def _lens_logits(norm, head, h_last):
    x = norm(h_last.unsqueeze(0))          
    return head(x).squeeze(0).float()      

@torch.no_grad()
def _resid_by_depth(model, ids):
    store = {}

    def mk(k):
        def hook(m, i, o):
            store[k] = (o[0] if isinstance(o, tuple) else o)[0, -1].detach()
        return hook

    handles = [model.model.embed_tokens.register_forward_hook(mk("emb"))]
    handles += [layer.register_forward_hook(mk(idx)) for idx, layer in enumerate(model.model.layers)]
    out = model(**ids)
    for h in handles:
        h.remove()
    depths = [store["emb"]] + [store[i] for i in range(len(model.model.layers))]
    return depths, out.logits[0, -1].float()

@torch.no_grad()
def _profile_prompt(model, tok, norm, head, prompt, dev):
    ids = tok(prompt, return_tensors="pt").to(dev)
    depths, final_logits = _resid_by_depth(model, ids)
    target = int(final_logits.argmax())
    probs, ranks, logits = [], [], []
    for h in depths:
        ll = _lens_logits(norm, head, h)
        p = F.softmax(ll, dim=-1)
        probs.append(float(p[target]))
        ranks.append(int((ll > ll[target]).sum().item()))   
        logits.append(float(ll[target]))
    return {"target": target, "target_str": tok.decode([target]),
            "prob": probs, "rank": ranks, "logit": logits}

def _aggregate(rows, n_depths):
    prob = np.array([r["prob"] for r in rows])             
    rank = np.array([r["rank"] for r in rows])
    logit = np.array([r["logit"] for r in rows])
    incr = np.diff(logit, axis=1)                          
    
    crys = []
    for r in rows:
        hit = next((d for d, rk in enumerate(r["rank"]) if rk == 0), len(r["rank"]) - 1)
        crys.append(hit)
    return {
        "n_depths": n_depths,
        "prob_mean": prob.mean(0).tolist(), "prob_std": prob.std(0).tolist(),
        "rank_median": np.median(rank, axis=0).tolist(),
        "rank_mean": rank.mean(0).tolist(),
        "logit_increment_mean": incr.mean(0).tolist(),    
        "crystallization_depth_mean": float(np.mean(crys)),
        "crystallization_depth_median": float(np.median(crys)),
        "crystallization_depth_per_prompt": crys,
        "n_depths_total": n_depths,
    }

@torch.no_grad()
def _selftest(model, tok, norm, head, dev) -> None:
    worst = 0.0
    for p in PROMPTS[:6]:
        ids = tok(p, return_tensors="pt").to(dev)
        depths, real = _resid_by_depth(model, ids)
        lens = _lens_logits(norm, head, depths[-1])   
        worst = max(worst, (lens - real).abs().max().item())
    assert worst < 1.0, f"final-layer lens != real logits (max|Δ|={worst:.3f}) -- norm/head wiring wrong"
    print("SELFTEST OK | final-lens reproduces real logits, max|Δ|=%.4f (bf16 noise)" % worst)

def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    status = os.path.join(OUT, "LENS_STATUS.txt")
    open(status, "w").write("LOADING\n")
    t0 = time.time()
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        tok = AutoTokenizer.from_pretrained(MID)
        model = AutoModelForCausalLM.from_pretrained(MID, dtype=torch.bfloat16).to(dev).eval()
        norm, head = _find_norm_and_head(model)
        n_layers = len(model.model.layers)
        print("loaded", MID, "| layers", n_layers, "| dev", dev)
        open(status, "w").write("SELFTEST\n")
        _selftest(model, tok, norm, head, dev)
        open(status, "w").write("RUNNING\n")
        rows = [_profile_prompt(model, tok, norm, head, p, dev) for p in PROMPTS]
        n_depths = len(rows[0]["prob"])
        summary = _aggregate(rows, n_depths)
        out = {"config": {"model": MID, "n_layers": n_layers, "n_depths": n_depths,
                          "n_prompts": len(PROMPTS)},
               "prompts": PROMPTS,
               "rows": [{k: r[k] for k in ("target", "target_str", "rank", "prob")} for r in rows],
               "summary": summary, "wall_s": time.time() - t0}
        p = os.path.join(OUT, "logit_lens_2b.json")
        json.dump(out, open(p, "w"), indent=2)
        if os.path.isdir(DRIVE):
            try:
                json.dump(out, open(os.path.join(DRIVE, "logit_lens_2b.json"), "w"), indent=2)
            except Exception as e:
                print("drive save failed:", e)
        cd = summary["crystallization_depth_median"]
        open(status, "w").write("DONE %.0fs crys_depth_median=%.1f/%d\n" % (time.time() - t0, cd, n_layers))
        print("DONE %.1fs | crystallization depth median=%.1f/%d mean=%.1f"
              % (time.time() - t0, cd, n_layers, summary["crystallization_depth_mean"]))
        print("rank_median by depth:", [int(x) for x in summary["rank_median"]])
        print("prob_mean by depth:", [round(x, 3) for x in summary["prob_mean"]])
    except Exception:
        tb = traceback.format_exc()
        print(tb)
        open(status, "w").write("ERROR\n" + tb)
        raise

if __name__ == "__main__":
    main()
