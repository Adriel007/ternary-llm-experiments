
from __future__ import annotations

import json
import math
import os
import sys
import time
import traceback

ROOT = os.environ.get("PHD_ROOT", os.environ.get("POC_ROOT", "/content/PhD-propose"))
sys.path[:0] = [ROOT, os.path.join(ROOT, "packages/bitnet_core/src")]

import numpy as np  
import torch  
import torch.nn as nn  
import torch.nn.functional as F  
from torch import autocast  

BITNET = "microsoft/bitnet-b1.58-2B-4T-bf16"
FP_MODEL = "unsloth/Llama-3.2-1B"   
BITNET_LAYER = 15          
FP_LAYER = 8               
SEED = 0
SEQ_LEN = 256
N_ACT = 200_000            
ACT_BATCH = 8
N_SEQS = 1200              
D_SAE = 16384
K = 32
CC_STEPS = 4000
CC_BATCH = 4096
CC_LR = 3e-4
AUX_COEF = 1.0 / 32
DEAD_STEPS = 200

OUT = os.path.join(ROOT, "artifacts/poc")
OUT_JSON = os.path.join(OUT, "crosscoder_scale.json")
STATUS = os.path.join(OUT, "CC_SCALE_STATUS.txt")
MODELS = ("bitnet", "llama_fp")

class CrossCoderHetero(nn.Module):

    def __init__(self, dims, d_sae, k, seed=0):
        super().__init__()
        self.dims = list(dims)           
        self.d_sae, self.k, self.n_models = d_sae, k, len(dims)
        g = torch.Generator().manual_seed(seed)
        self.W_dec = nn.ParameterList([
            nn.Parameter(torch.randn(d_sae, d, generator=g) / (d ** 0.5)) for d in dims])
        D = sum(dims)
        self.W_enc = nn.Parameter(torch.randn(D, d_sae, generator=g) / (D ** 0.5))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.b_dec = nn.ParameterList([nn.Parameter(torch.zeros(d)) for d in dims])
        self.register_buffer("last_active", torch.zeros(d_sae, dtype=torch.long))
        self.normalize_decoder_()

    def pre_acts(self, acts):                       
        xc = torch.cat([acts[m] - self.b_dec[m] for m in range(self.n_models)], dim=-1)  
        return xc @ self.W_enc + self.b_enc

    def batch_topk(self, acts):
        b = acts.shape[0]
        n = self.k * b
        flat = acts.flatten()
        if 0 < n < flat.numel():
            thresh = torch.topk(flat, n).values.min()
            acts = acts * (acts >= thresh)
        return acts

    def decode(self, z):                            
        return [z @ self.W_dec[m] + self.b_dec[m] for m in range(self.n_models)]

    def forward(self, acts):
        pre = self.pre_acts(acts)
        a_full = F.relu(pre)
        z = self.batch_topk(a_full)
        a_hat = self.decode(z)
        return a_hat, z, a_full

    @torch.no_grad()
    def normalize_decoder_(self):
        
        sq = sum(self.W_dec[m].pow(2).sum(dim=1) for m in range(self.n_models))  
        norm = sq.sqrt().clamp_min(1e-8)[:, None]
        for m in range(self.n_models):
            self.W_dec[m].div_(norm)

def _aux_loss(cc, a_full, residual, dead):
    if cc.k <= 0 or dead.sum() == 0:
        return residual[0].new_zeros(())
    kk = min(256, int(dead.sum().item()))
    ad = a_full.clone()
    ad[:, ~dead] = 0.0
    topv, topi = ad.topk(kk, dim=-1)
    z_aux = torch.zeros_like(ad).scatter_(-1, topi, topv)
    rec = [z_aux @ cc.W_dec[m] for m in range(cc.n_models)]
    return sum(F.mse_loss(rec[m], residual[m]) for m in range(cc.n_models))

@torch.no_grad()
def classify_features(cc, alive_thresh=1e-3):
    n0 = cc.W_dec[0].detach().norm(dim=1)          
    n1 = cc.W_dec[1].detach().norm(dim=1)
    total = n0 + n1
    alive = total > alive_thresh
    r = n0 / total.clamp_min(1e-9)                  
    r_alive = r[alive]
    shared = alive & (r > 0.35) & (r < 0.65)
    bit_excl = alive & (r >= 0.8)
    fp_excl = alive & (r <= 0.2)
    n_alive = int(alive.sum().item())
    edges = np.linspace(0, 1, 21)
    return {
        "d_sae": int(cc.d_sae), "k": int(cc.k), "n_alive": n_alive,
        "n_dead": int((~alive).sum().item()),
        "n_shared": int(shared.sum().item()), "n_bitnet_excl": int(bit_excl.sum().item()),
        "n_fp_excl": int(fp_excl.sum().item()),
        "frac_shared": shared.sum().item() / max(n_alive, 1),
        "frac_bitnet_excl": bit_excl.sum().item() / max(n_alive, 1),
        "frac_fp_excl": fp_excl.sum().item() / max(n_alive, 1),
        "r_mean": float(r_alive.mean().item()), "r_median": float(r_alive.median().item()),
        "r_hist_edges": edges.tolist(),
        "r_hist_counts": np.histogram(r_alive.cpu().numpy(), bins=edges)[0].tolist(),
    }

def _token_windows(tok, n_seqs, seq_len):
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train", streaming=True)
    eos = tok.eos_token_id or 0
    need = n_seqs * seq_len
    buf = []
    for ex in ds:
        t = ex["text"].strip()
        if not t:
            continue
        buf.extend(tok(t, add_special_tokens=False).input_ids)
        buf.append(eos)
        if len(buf) >= need:
            break
    return torch.tensor(buf[:need], dtype=torch.long).view(n_seqs, seq_len)

@torch.no_grad()
def collect_paired(models, layers, ids, n_target, device):
    stores = [[], []]

    def mk_hook(buf):
        def hook(m, i, o):
            h = o[0] if isinstance(o, tuple) else o
            buf.append(h.reshape(-1, h.shape[-1]).float().cpu())
        return hook

    handles = [models[k].model.layers[layers[k]].register_forward_hook(mk_hook(stores[k]))
               for k in range(2)]
    seen = 0
    for b in range(0, ids.shape[0], ACT_BATCH):
        batch = ids[b: b + ACT_BATCH].to(device)
        with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            for mdl in models:
                mdl(input_ids=batch)
        seen += stores[0][-1].shape[0]
        if seen >= n_target:
            break
    for h in handles:
        h.remove()
    return [torch.cat(s, 0)[:n_target].contiguous() for s in stores]

def standardize_(A):
    scales = []
    for m in range(len(A)):
        d = A[m].shape[1]
        A[m].sub_(A[m].mean(0, keepdim=True))
        s = (A[m].pow(2).sum(1).mean() / d).sqrt().clamp_min(1e-8)
        A[m].div_(s)
        scales.append(float(s.item()))
    return scales

def train_crosscoder(A, dims, device="cuda", log_every=500):
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(SEED)
    N = A[0].shape[0]
    cc = CrossCoderHetero(dims, D_SAE, K, seed=SEED).to(device)
    with torch.no_grad():
        for m in range(2):
            cc.b_dec[m].data = A[m][:65536].mean(0).to(device)
    opt = torch.optim.Adam(cc.parameters(), lr=CC_LR)
    gen = torch.Generator().manual_seed(SEED + 1)
    hist = {"step": [], "fve_bitnet": [], "fve_fp": [], "l0": [], "dead": []}
    for step in range(CC_STEPS):
        idx = torch.randint(N, (CC_BATCH,), generator=gen)
        a = [A[m][idx].to(device) for m in range(2)]
        a_hat, z, a_full = cc(a)
        recon = sum(F.mse_loss(a_hat[m], a[m]) for m in range(2))
        active = (z > 0).any(0)
        cc.last_active[active] = 0
        cc.last_active[~active] += 1
        dead = cc.last_active > DEAD_STEPS
        aux = _aux_loss(cc, a_full, [(a[m] - a_hat[m]).detach() for m in range(2)], dead)
        loss = recon + AUX_COEF * aux
        opt.zero_grad(set_to_none=True)
        loss.backward()
        cc.normalize_decoder_()
        opt.step()
        if step % log_every == 0 or step == CC_STEPS - 1:
            with torch.no_grad():
                fve = []
                for m in range(2):
                    num = (a[m] - a_hat[m]).pow(2).sum().item()
                    den = (a[m] - a[m].mean(0)).pow(2).sum().item() + 1e-9
                    fve.append(1.0 - num / den)
                l0 = (z > 0).float().sum(-1).mean().item()
            hist["step"].append(step); hist["fve_bitnet"].append(fve[0]); hist["fve_fp"].append(fve[1])
            hist["l0"].append(l0); hist["dead"].append(int(dead.sum().item()))
            print("  cc step %d: fve_bit=%.3f fve_fp=%.3f l0=%.1f dead=%d"
                  % (step, fve[0], fve[1], l0, int(dead.sum().item())), flush=True)
    metrics = classify_features(cc)
    metrics["final_fve_bitnet"] = hist["fve_bitnet"][-1]
    metrics["final_fve_fp"] = hist["fve_fp"][-1]
    metrics["final_l0"] = hist["l0"][-1]
    return cc, hist, metrics

def _status(s):
    os.makedirs(OUT, exist_ok=True)
    open(STATUS, "w").write(s + "\n")

def main():
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    _status("LOADING")
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        tok = AutoTokenizer.from_pretrained(BITNET)        
        print("loading models...", flush=True)
        m_bit = AutoModelForCausalLM.from_pretrained(BITNET, dtype=torch.bfloat16).to(dev).eval()
        m_fp = AutoModelForCausalLM.from_pretrained(FP_MODEL, dtype=torch.bfloat16,
                                                    token=os.environ.get("HF_TOKEN")).to(dev).eval()
        m_bit.requires_grad_(False); m_fp.requires_grad_(False)
        d0 = m_bit.config.hidden_size
        d1 = m_fp.config.hidden_size
        
        assert m_bit.config.vocab_size == m_fp.config.vocab_size,            "tokenizer/vocab mismatch -> residuals not positionally comparable"
        print("bitnet d=%d (L%d) | %s d=%d (L%d) | vocab %d"
              % (d0, BITNET_LAYER, FP_MODEL, d1, FP_LAYER, m_bit.config.vocab_size), flush=True)

        _status("COLLECT")
        ids = _token_windows(tok, N_SEQS, SEQ_LEN)
        A = collect_paired([m_bit, m_fp], [BITNET_LAYER, FP_LAYER], ids, N_ACT, dev)
        scales = standardize_(A)
        print("collected paired acts: bitnet %s | fp %s | scales %s"
              % (tuple(A[0].shape), tuple(A[1].shape), [round(s, 3) for s in scales]), flush=True)
        del m_bit, m_fp
        torch.cuda.empty_cache()

        _status("TRAIN CC")
        cc, hist, metrics = train_crosscoder(A, [d0, d1], dev)

        result = {
            "models": {"bitnet": BITNET, "fp": FP_MODEL}, "layers": [BITNET_LAYER, FP_LAYER],
            "dims": [d0, d1], "d_sae": D_SAE, "k": K, "n_act": N_ACT, "seq_len": SEQ_LEN,
            "cc_steps": CC_STEPS, "scales": scales, "metrics": metrics, "history": hist,
            "wall_s": time.time() - t0,
        }
        os.makedirs(OUT, exist_ok=True)
        json.dump(result, open(OUT_JSON + ".tmp", "w"), indent=2)
        os.replace(OUT_JSON + ".tmp", OUT_JSON)
        torch.save(cc.state_dict(), os.path.join(OUT, "crosscoder_scale.pt"))

        print("\n==== SCALED DIFF (BitNet-2B vs %s) ====" % FP_MODEL, flush=True)
        print("alive %d/%d | shared %.3f | bitnet-excl %.3f | fp-excl %.3f | r_med %.3f"
              % (metrics["n_alive"], D_SAE, metrics["frac_shared"], metrics["frac_bitnet_excl"],
                 metrics["frac_fp_excl"], metrics["r_median"]))
        print("fve bitnet %.3f / fp %.3f | l0 %.1f"
              % (metrics["final_fve_bitnet"], metrics["final_fve_fp"], metrics["final_l0"]))
        _status("DONE %.0fs" % (time.time() - t0))
        print("\nDONE %.1fs" % (time.time() - t0), flush=True)
    except Exception:
        tb = traceback.format_exc()
        print(tb, flush=True)
        _status("ERROR\n" + tb)
        raise

if __name__ == "__main__":
    main()
