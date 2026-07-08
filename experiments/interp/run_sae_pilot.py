
from __future__ import annotations

import json
import os
import sys
import time
import traceback

ROOT = "/content/PhD-propose"
sys.path[:0] = [ROOT, os.path.join(ROOT, "packages/bitnet_core/src")]

import torch  

from experiments.interp.sae import SAEConfig, train_sae  

MID = "microsoft/bitnet-b1.58-2B-4T-bf16"
LAYER = 15                 
N_VECTORS = 400_000        
SEQ_LEN = 512
BATCH = 8
D_SAE = 16384             
K = 32
SAE_STEPS = 3000
OUT = os.path.join(ROOT, "artifacts/poc")
DRIVE = "/content/drive/MyDrive/PhD_PoC"

def collect_activations(model, tok, layer_idx, n_target, device):
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train", streaming=True)
    eos = tok.eos_token_id
    buf = []
    need_ids = n_target + SEQ_LEN * BATCH
    for ex in ds:
        t = ex["text"].strip()
        if not t:
            continue
        buf.extend(tok(t, add_special_tokens=False).input_ids)
        buf.append(eos)
        if len(buf) >= need_ids:
            break
    n_seq = len(buf) // SEQ_LEN
    ids = torch.tensor(buf[: n_seq * SEQ_LEN], dtype=torch.long).view(n_seq, SEQ_LEN)

    store = []
    total = 0
    def hook(m, i, o):
        h = o[0] if isinstance(o, tuple) else o
        store.append(h.reshape(-1, h.shape[-1]).float())   
    layer = model.model.layers[layer_idx]
    handle = layer.register_forward_hook(hook)
    with torch.no_grad():
        for b in range(0, n_seq, BATCH):
            batch = ids[b : b + BATCH].to(device)
            model(input_ids=batch, use_cache=False)
            total += store[-1].shape[0]
            if total >= n_target:
                break
    handle.remove()
    X = torch.cat(store, 0)[:n_target].contiguous()
    del store
    torch.cuda.empty_cache()
    return X

def main():
    os.makedirs(OUT, exist_ok=True)
    status = os.path.join(OUT, "SAE_STATUS.txt")
    open(status, "w").write("RUNNING\n")
    t0 = time.time()
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        dev = "cuda"
        tok = AutoTokenizer.from_pretrained(MID)
        model = AutoModelForCausalLM.from_pretrained(MID, dtype=torch.bfloat16).to(dev).eval()
        print("model loaded", flush=True)

        X = collect_activations(model, tok, LAYER, N_VECTORS, dev)
        print(f"cached activations: {tuple(X.shape)} | mean|x|={X.abs().mean():.3f}", flush=True)
        del model
        torch.cuda.empty_cache()

        cfg = SAEConfig(d_in=X.shape[1], d_sae=D_SAE, k=K, steps=SAE_STEPS, batch_size=4096)
        sae, hist, final = train_sae(X, cfg, device=dev, log_every=500)
        print("SAE trained:", final, flush=True)

        result = {
            "model": MID, "layer": LAYER, "corpus": "wikitext-103-raw-v1",
            "n_vectors": int(X.shape[0]), "d_in": int(X.shape[1]),
            "sae": {"d_sae": D_SAE, "k": K, "steps": SAE_STEPS, "expansion": D_SAE / X.shape[1]},
            "final_metrics": final, "history": hist, "wall_s": time.time() - t0,
        }
        json.dump(result, open(os.path.join(OUT, "sae_pilot.json"), "w"), indent=2)
        torch.save(sae.state_dict(), os.path.join(OUT, "sae_layer%d.pt" % LAYER))
        if os.path.isdir(DRIVE):
            json.dump(result, open(os.path.join(DRIVE, "sae_pilot.json"), "w"), indent=2)
            torch.save(sae.state_dict(), os.path.join(DRIVE, "sae_layer%d.pt" % LAYER))
        open(status, "w").write(f"DONE {time.time()-t0:.0f}s\n" + json.dumps(final))
        print("SAE PILOT DONE", round(time.time() - t0, 1), "s |", final, flush=True)
    except Exception:
        tb = traceback.format_exc(); print(tb)
        open(status, "w").write("ERROR\n" + tb)
        raise

if __name__ == "__main__":
    main()
