
from __future__ import annotations

import json
import os
import sys
import time
import traceback

ROOT = os.environ.get("POC_ROOT", "/content/PhD-propose")
sys.path[:0] = [ROOT, os.path.join(ROOT, "packages/bitnet_core/src")]

import torch  

from experiments.interp.sae import SAEConfig, BatchTopKSAE, train_sae  

MID = "microsoft/bitnet-b1.58-2B-4T-bf16"
LAYERS = [6, 15, 23]          
N_VECTORS = 400_000
SEQ_LEN = 512
BATCH = 8
D_SAE = 16384                 
K = 32
SAE_STEPS = int(os.environ.get("SAE_STEPS", "4000"))   
TOP_FEATS = 24                
TOP_EX = 6                    
CTX = (6, 2)                  
OUT = os.path.join(ROOT, "artifacts/poc")
DRIVE = "/content/drive/MyDrive/PhD_PoC"
TAG = ""                      

def _apply_overrides():
    global MID, LAYERS, D_SAE, TAG
    mid = os.environ.get("SAE_MID")
    if not mid:
        return
    MID = mid
    if os.environ.get("SAE_LAYERS"):
        LAYERS = [int(x) for x in os.environ["SAE_LAYERS"].split(",")]
    TAG = os.environ.get("SAE_TAG", "_mirror")
    if os.environ.get("SAE_DSAE"):
        D_SAE = int(os.environ["SAE_DSAE"])
    else:
        D_SAE = None             

EXPANSION = 16384 / 2560        

def _load_corpus_ids(tok, n_target):
    from datasets import load_dataset
    stream = os.environ.get("POC_STREAM", "1") == "1"   
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train", streaming=stream)
    eos = tok.eos_token_id
    buf, need = [], n_target + SEQ_LEN * BATCH
    for ex in ds:
        t = ex["text"].strip()
        if not t:
            continue
        buf.extend(tok(t, add_special_tokens=False).input_ids)
        buf.append(eos)
        if len(buf) >= need:
            break
    n_seq = len(buf) // SEQ_LEN
    return torch.tensor(buf[: n_seq * SEQ_LEN], dtype=torch.long).view(n_seq, SEQ_LEN)

@torch.no_grad()
def collect_multi(model, ids, layers, n_target, device):
    stores = {l: [] for l in layers}

    def mk(l):
        def hook(m, i, o):
            h = o[0] if isinstance(o, tuple) else o

            stores[l].append(h.reshape(-1, h.shape[-1]).bfloat16().cpu())
        return hook

    handles = [model.model.layers[l].register_forward_hook(mk(l)) for l in layers]
    seen = 0
    for b in range(0, ids.shape[0], BATCH):
        model(input_ids=ids[b: b + BATCH].to(device), use_cache=False)
        seen += stores[layers[0]][-1].shape[0]
        if seen >= n_target:
            break
    for h in handles:
        h.remove()
    Xs = {l: torch.cat(s, 0)[:n_target].contiguous() for l, s in stores.items()}
    torch.cuda.empty_cache()
    return Xs

@torch.no_grad()
def feature_dashboard(sae, X, ids, tok, device, top_feats=TOP_FEATS, top_ex=TOP_EX):
    N = X.shape[0]
    bs = 8192
    d_sae = sae.cfg.d_sae
    freq = torch.zeros(d_sae)
    for b in range(0, N, bs):
        x = X[b: b + bs].to(device).float()
        acts = torch.relu(sae.pre_acts(x))
        freq += (acts > 0).float().sum(0).cpu()
    
    rate = freq / N
    cand = ((rate > 0.01) & (rate < 0.40)).nonzero().flatten()
    if cand.numel() == 0:
        cand = freq.argsort(descending=True)[:top_feats]
    sel = cand[freq[cand].argsort(descending=True)[:top_feats]].tolist()
    
    best_val = {j: torch.full((top_ex,), -1.0) for j in sel}
    best_row = {j: torch.full((top_ex,), -1, dtype=torch.long) for j in sel}
    sel_t = torch.tensor(sel)
    for b in range(0, N, bs):
        x = X[b: b + bs].to(device).float()
        a = torch.relu(sae.pre_acts(x))[:, sel_t].cpu()      
        rows = torch.arange(b, b + x.shape[0])
        for k, j in enumerate(sel):
            v = torch.cat([best_val[j], a[:, k]]); r = torch.cat([best_row[j], rows])
            tv, ti = v.topk(min(top_ex, v.numel()))
            best_val[j], best_row[j] = tv, r[ti]
    out = []
    L = SEQ_LEN
    for j in sel:
        exs = []
        for val, row in zip(best_val[j].tolist(), best_row[j].tolist()):
            if row < 0 or val <= 0:
                continue
            seq, pos = row // L, row % L
            lo, hi = max(0, pos - CTX[0]), min(L, pos + CTX[1] + 1)
            ctx = tok.decode(ids[seq, lo:hi].tolist())
            focus = tok.decode(ids[seq, pos:pos + 1].tolist())
            exs.append({"act": round(val, 3), "focus": focus, "context": ctx})
        out.append({"feature": int(j), "fire_rate": round(float((freq[j] / N).item()), 4),
                    "examples": exs})
    return out

def _resolve_dsae(d_in):
    if D_SAE is not None:
        return D_SAE
    return int(round(EXPANSION * d_in / 64)) * 64

def _run_layer(layer, X, ids, tok, device, steps=SAE_STEPS, do_dash=True):
    d_sae = _resolve_dsae(X.shape[1])
    cfg = SAEConfig(d_in=X.shape[1], d_sae=d_sae, k=K, steps=steps, batch_size=4096,
                    seed=int(os.environ.get("SAE_SEED", "0")))
    Xg = X.to(device).float()
    sae, hist, final = train_sae(Xg, cfg, device=device, log_every=1000)
    dash = []
    if do_dash:
        try:
            dash = feature_dashboard(sae, X, ids, tok, device)
        except Exception as e:
            print("dashboard failed (kept SAE):", repr(e), flush=True)
    del Xg; torch.cuda.empty_cache()
    return sae, {"layer": layer, "metrics": final, "history": hist, "dashboard": dash,
                 "n_vectors": int(X.shape[0]), "d_in": int(X.shape[1]),
                 "sae": {"d_sae": d_sae, "k": K, "steps": steps}}

def smoke(model, tok, device):
    ids = _load_corpus_ids(tok, 6000)
    Xs = collect_multi(model, ids, [LAYERS[0]], 5000, device)
    X = Xs[LAYERS[0]]
    assert X.shape[0] >= 4000 and X.shape[1] > 1000, ("bad collect", tuple(X.shape))
    _, r = _run_layer(LAYERS[0], X, ids, tok, device, steps=200)
    assert 0.0 < r["metrics"]["fve"] <= 1.0, r["metrics"]
    assert isinstance(r["dashboard"], list), "dashboard must be a list"
    print("[smoke] PASS fve=%.3f l0=%.1f dash=%d feats" % (
        r["metrics"]["fve"], r["metrics"]["l0"], len(r["dashboard"])), flush=True)

def main():
    _apply_overrides()
    os.makedirs(OUT, exist_ok=True)
    status = os.path.join(OUT, "SAEML_STATUS%s.txt" % TAG)
    open(status, "w").write("RUNNING\n")
    t0 = time.time()
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        dev = "cuda"
        tok = AutoTokenizer.from_pretrained(MID)
        model = AutoModelForCausalLM.from_pretrained(MID, torch_dtype=torch.bfloat16).to(dev).eval()
        print("model loaded", flush=True)

        smoke(model, tok, dev)                              

        ids = _load_corpus_ids(tok, N_VECTORS)
        Xs = collect_multi(model, ids, LAYERS, N_VECTORS, dev)
        print("collected layers:", {l: tuple(x.shape) for l, x in Xs.items()}, flush=True)
        del model; torch.cuda.empty_cache()

        layers_out = []
        for l in LAYERS:
            sae, r = _run_layer(l, Xs[l], ids, tok, dev)
            torch.save(sae.state_dict(), os.path.join(OUT, "sae_ml%s_L%d.pt" % (TAG, l)))
            if os.path.isdir(DRIVE):
                torch.save(sae.state_dict(), os.path.join(DRIVE, "sae_ml%s_L%d.pt" % (TAG, l)))
            layers_out.append(r)
            del sae, Xs[l]; torch.cuda.empty_cache()
            print("[layer %d] FVE=%.3f L0=%.0f dead=%d dash=%d" % (
                l, r["metrics"]["fve"], r["metrics"]["l0"], r["metrics"]["dead"], len(r["dashboard"])), flush=True)
            
            res = {"model": MID, "layers": LAYERS, "corpus": "wikitext-103-raw-v1",
                   "tag": TAG, "results": layers_out, "wall_s": time.time() - t0}
            json.dump(res, open(os.path.join(OUT, "sae_multilayer%s.json" % TAG), "w"), indent=2)
            if os.path.isdir(DRIVE):
                json.dump(res, open(os.path.join(DRIVE, "sae_multilayer%s.json" % TAG), "w"), indent=2)

        open(status, "w").write("DONE %.0fs\n" % (time.time() - t0) +
                                json.dumps({l["layer"]: l["metrics"] for l in layers_out}))
        print("SAE MULTILAYER DONE %.0fs" % (time.time() - t0), flush=True)
    except Exception:
        tb = traceback.format_exc(); print(tb)
        open(status, "w").write("ERROR\n" + tb)
        raise

if __name__ == "__main__":
    main()
