
from __future__ import annotations

import json
import os
import sys
import time
import traceback

ROOT = "/content/PhD-propose"
sys.path[:0] = [ROOT, os.path.join(ROOT, "packages/bitnet_core/src")]

import torch  
import torch.nn.functional as F  

from experiments.interp.sae import SAEConfig, BatchTopKSAE  

MID = "microsoft/bitnet-b1.58-2B-4T-bf16"
LAYERS = [6, 15, 23]
SEQ_LEN = 512
BATCH = 8                     
N_EVAL_SEQS = 200             
K = 32
TAG = ""                      
OUT = os.path.join(ROOT, "artifacts/poc")
DRIVE = "/content/drive/MyDrive/PhD_PoC"

def _apply_overrides():
    global MID, LAYERS, TAG
    if os.environ.get("SAE_MID"):
        MID = os.environ["SAE_MID"]
        TAG = os.environ.get("SAE_TAG", "_mirror")
        if os.environ.get("SAE_LAYERS"):
            LAYERS = [int(x) for x in os.environ["SAE_LAYERS"].split(",")]

def _eval_ids(tok, n_seq):
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="test", streaming=True)
    eos = tok.eos_token_id
    buf, need = [], (n_seq + BATCH) * SEQ_LEN
    for ex in ds:
        t = ex["text"].strip()
        if not t:
            continue
        buf.extend(tok(t, add_special_tokens=False).input_ids)
        buf.append(eos)
        if len(buf) >= need:
            break
    n = min(n_seq, len(buf) // SEQ_LEN)
    if n < 8:
        raise RuntimeError(f"only {len(buf)} eval tokens collected")
    return torch.tensor(buf[: n * SEQ_LEN], dtype=torch.long).view(n, SEQ_LEN)

@torch.no_grad()
def _ce(model, ids, device, hook_layer=None, hook_fn=None):
    handle = None
    if hook_layer is not None:
        handle = model.model.layers[hook_layer].register_forward_hook(hook_fn)
    tot_loss, tot_tok = 0.0, 0
    for b in range(0, ids.shape[0], BATCH):
        x = ids[b: b + BATCH].to(device)
        out = model(input_ids=x, use_cache=False)
        logits = out.logits[:, :-1, :]
        tgt = x[:, 1:]
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(),
                               tgt.reshape(-1), reduction="sum")
        tot_loss += float(loss.item()); tot_tok += tgt.numel()
    if handle is not None:
        handle.remove()
    return tot_loss / tot_tok

@torch.no_grad()
def _layer_mean(model, ids, layer, device):
    acc = {"sum": None, "n": 0}

    def hook(m, i, o):
        h = o[0] if isinstance(o, tuple) else o
        flat = h.reshape(-1, h.shape[-1]).float()
        s = flat.sum(0)
        acc["sum"] = s if acc["sum"] is None else acc["sum"] + s
        acc["n"] += flat.shape[0]

    handle = model.model.layers[layer].register_forward_hook(hook)
    for b in range(0, ids.shape[0], BATCH):
        model(input_ids=ids[b: b + BATCH].to(device), use_cache=False)
    handle.remove()
    return (acc["sum"] / acc["n"]).to(device)

def _mk_recon_hook(sae):
    def hook(m, i, o):
        h = o[0] if isinstance(o, tuple) else o
        shp = h.shape
        x = h.reshape(-1, shp[-1]).float()
        xhat, _, _ = sae(x)
        xhat = xhat.to(h.dtype).reshape(shp)
        return (xhat,) + tuple(o[1:]) if isinstance(o, tuple) else xhat
    return hook

def _mk_mean_hook(mean_vec):
    def hook(m, i, o):
        h = o[0] if isinstance(o, tuple) else o
        hh = mean_vec.to(h.dtype).expand_as(h)
        return (hh,) + tuple(o[1:]) if isinstance(o, tuple) else hh
    return hook

def _load_sae(layer, d_in, device):
    pt = os.path.join(OUT, "sae_ml%s_L%d.pt" % (TAG, layer))
    if not os.path.exists(pt):
        raise FileNotFoundError(pt + " (run run_sae_multilayer.py first)")
    sd = torch.load(pt, map_location=device)
    d_sae = sd["W_dec"].shape[1]          
    cfg = SAEConfig(d_in=d_in, d_sae=d_sae, k=K)
    sae = BatchTopKSAE(cfg).to(device)
    sae.load_state_dict(sd)
    return sae.eval()

def main():
    _apply_overrides()
    os.makedirs(OUT, exist_ok=True)
    status = os.path.join(OUT, "SAELR_STATUS%s.txt" % TAG)
    open(status, "w").write("RUNNING\n")
    t0 = time.time()
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        dev = "cuda"
        tok = AutoTokenizer.from_pretrained(MID)
        model = AutoModelForCausalLM.from_pretrained(MID, dtype=torch.bfloat16).to(dev).eval()
        ids = _eval_ids(tok, N_EVAL_SEQS)
        d_in = model.config.hidden_size
        print("eval ids:", tuple(ids.shape), "d_in", d_in, flush=True)

        l_clean = _ce(model, ids, dev)
        print("L_clean = %.4f" % l_clean, flush=True)

        results = []
        for l in LAYERS:
            sae = _load_sae(l, d_in, dev)
            mean_vec = _layer_mean(model, ids, l, dev)
            l_recon = _ce(model, ids, dev, hook_layer=l, hook_fn=_mk_recon_hook(sae))
            l_abl = _ce(model, ids, dev, hook_layer=l, hook_fn=_mk_mean_hook(mean_vec))
            recovered = (l_abl - l_recon) / (l_abl - l_clean) if (l_abl - l_clean) > 1e-9 else float("nan")
            row = {"layer": l, "L_clean": l_clean, "L_recon": l_recon, "L_ablate": l_abl,
                   "loss_recovered": recovered, "delta_ce_recon": l_recon - l_clean}
            results.append(row)
            print("[L%d] L_recon=%.4f L_abl=%.4f recovered=%.4f (dCE=%.4f)" % (
                l, l_recon, l_abl, recovered, l_recon - l_clean), flush=True)
            del sae; torch.cuda.empty_cache()

        res = {"model": MID, "tag": TAG, "eval": "wikitext-103-raw-v1/test",
               "n_eval_seqs": int(ids.shape[0]), "seq_len": SEQ_LEN, "k": K, "L_clean": l_clean,
               "results": results, "wall_s": time.time() - t0}
        json.dump(res, open(os.path.join(OUT, "sae_loss_recovered%s.json" % TAG), "w"), indent=2)
        if os.path.isdir(DRIVE):
            json.dump(res, open(os.path.join(DRIVE, "sae_loss_recovered%s.json" % TAG), "w"), indent=2)
        open(status, "w").write("DONE %.0fs\n" % (time.time() - t0) +
                                json.dumps({r["layer"]: round(r["loss_recovered"], 4) for r in results}))
        print("SAE LOSS-RECOVERED DONE %.0fs" % (time.time() - t0), flush=True)
    except Exception:
        tb = traceback.format_exc(); print(tb)
        open(status, "w").write("ERROR\n" + tb)
        raise

if __name__ == "__main__":
    main()
