
from __future__ import annotations

import json
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

from experiments.interp.crosscoder_scale import (  
    BITNET, FP_MODEL, BITNET_LAYER, FP_LAYER, SEQ_LEN,
    collect_paired, _token_windows, train_crosscoder,
)

def standardize_fit_apply(A_tr, A_ho):
    scales = []
    for m in range(len(A_tr)):
        d = A_tr[m].shape[1]
        mu = A_tr[m].mean(0, keepdim=True)
        A_tr[m].sub_(mu)
        s = (A_tr[m].pow(2).sum(1).mean() / d).sqrt().clamp_min(1e-8)
        A_tr[m].div_(s)
        A_ho[m].sub_(mu).div_(s)
        scales.append(float(s.item()))
    return scales

SEED = 0
N_ACT = 220_000            
HELDOUT = 20_000
N_SEQS = 1300
D_SAE = 16384
EXCL_FRAC = 0.05           
K = 32
CC_STEPS = 4000
CC_BATCH = 4096
CC_LR = 3e-4
AUX_COEF = 1.0 / 32
DEAD_STEPS = 200
TOPN_MAXACT = 20           

OUT = os.path.join(ROOT, "artifacts/poc")
OUT_JSON = os.path.join(OUT, "dfc_scale.json")
STATUS = os.path.join(OUT, "DFC_SCALE_STATUS.txt")

class DFC(nn.Module):

    def __init__(self, dims, n_excl, n_shared, k, seed=0):
        super().__init__()
        d0, d1 = dims
        self.dims = [d0, d1]
        self.nA = self.nB = int(n_excl)
        self.nS = int(n_shared)
        self.d_sae = self.nA + self.nB + self.nS
        self.k, self.n_models = k, 2
        g = torch.Generator().manual_seed(seed)
        D = d0 + d1
        self.W_enc = nn.Parameter(torch.randn(D, self.d_sae, generator=g) / (D ** 0.5))
        self.b_enc = nn.Parameter(torch.zeros(self.d_sae))
        self.W_dec = nn.ParameterList([
            nn.Parameter(torch.randn(self.d_sae, d, generator=g) / (d ** 0.5)) for d in dims])
        self.b_dec = nn.ParameterList([nn.Parameter(torch.zeros(d)) for d in dims])
        
        emask = torch.ones(D, self.d_sae)
        emask[d0:, 0:self.nA] = 0.0                       
        emask[0:d0, self.nA:self.nA + self.nB] = 0.0      
        self.register_buffer("emask", emask)
        
        dm0 = torch.ones(self.d_sae); dm0[self.nA:self.nA + self.nB] = 0.0   
        dm1 = torch.ones(self.d_sae); dm1[0:self.nA] = 0.0                   
        self.register_buffer("dmask0", dm0[:, None])
        self.register_buffer("dmask1", dm1[:, None])
        self.register_buffer("last_active", torch.zeros(self.d_sae, dtype=torch.long))
        self.normalize_decoder_()

    def slot_partition(self):
        idx = torch.arange(self.d_sae)
        return {"I_A": idx < self.nA,
                "I_B": (idx >= self.nA) & (idx < self.nA + self.nB),
                "I_S": idx >= self.nA + self.nB}

    def pre_acts(self, acts):
        xc = torch.cat([acts[m] - self.b_dec[m] for m in range(2)], dim=-1)
        return xc @ (self.W_enc * self.emask) + self.b_enc

    def batch_topk(self, a):
        b = a.shape[0]; n = self.k * b; flat = a.flatten()
        if 0 < n < flat.numel():
            thresh = torch.topk(flat, n).values.min()
            a = a * (a >= thresh)
        return a

    def decode(self, z, exclude=None):
        dm0, dm1 = self.dmask0, self.dmask1
        if exclude == "exclusive":
            keep = self.slot_partition()["I_S"].to(z.device)[:, None].float()
            dm0 = dm0 * keep; dm1 = dm1 * keep
        return [z @ (self.W_dec[0] * dm0) + self.b_dec[0],
                z @ (self.W_dec[1] * dm1) + self.b_dec[1]]

    def forward(self, acts):
        a_full = F.relu(self.pre_acts(acts))
        z = self.batch_topk(a_full)
        return self.decode(z), z, a_full

    @torch.no_grad()
    def normalize_decoder_(self):
        sq = ((self.W_dec[0] * self.dmask0).pow(2).sum(1)
              + (self.W_dec[1] * self.dmask1).pow(2).sum(1))
        norm = sq.sqrt().clamp_min(1e-8)[:, None]
        self.W_dec[0].div_(norm); self.W_dec[1].div_(norm)
        self.W_dec[0].mul_(self.dmask0); self.W_dec[1].mul_(self.dmask1)   
        self.W_enc.mul_(self.emask)

def _aux_loss(cc, a_full, residual, dead):
    if cc.k <= 0 or dead.sum() == 0:
        return residual[0].new_zeros(())
    kk = min(256, int(dead.sum().item()))
    ad = a_full.clone(); ad[:, ~dead] = 0.0
    topv, topi = ad.topk(kk, dim=-1)
    z_aux = torch.zeros_like(ad).scatter_(-1, topi, topv)
    rec = cc.decode(z_aux)
    return sum(F.mse_loss(rec[m] - cc.b_dec[m], residual[m]) for m in range(2))

def train_dfc(A_tr, dims, device="cuda", log_every=500):
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(SEED)
    N = A_tr[0].shape[0]
    n_excl = int(round(EXCL_FRAC * D_SAE))
    cc = DFC(dims, n_excl, D_SAE - 2 * n_excl, K, seed=SEED).to(device)
    with torch.no_grad():
        for m in range(2):
            cc.b_dec[m].data = A_tr[m][:65536].mean(0).to(device)
    opt = torch.optim.Adam(cc.parameters(), lr=CC_LR)
    gen = torch.Generator().manual_seed(SEED + 1)
    hist = {"step": [], "fve_bitnet": [], "fve_fp": [], "l0": [], "dead": []}
    for step in range(CC_STEPS):
        idx = torch.randint(N, (CC_BATCH,), generator=gen)
        a = [A_tr[m][idx].to(device) for m in range(2)]
        a_hat, z, a_full = cc(a)
        recon = sum(F.mse_loss(a_hat[m], a[m]) for m in range(2))
        active = (z > 0).any(0)
        cc.last_active[active] = 0; cc.last_active[~active] += 1
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
            print("  dfc step %d: fve_bit=%.3f fve_fp=%.3f l0=%.1f dead=%d"
                  % (step, fve[0], fve[1], l0, int(dead.sum().item())), flush=True)
    return cc, hist

@torch.no_grad()
def _fve(x, x_hat):
    num = (x - x_hat).pow(2).sum().item()
    den = (x - x.mean(0)).pow(2).sum().item() + 1e-9
    return 1.0 - num / den

@torch.no_grad()
def analyze_dfc(cc, A_ho, ids_ho, tok, device, batch=8192):
    N = A_ho[0].shape[0]
    part = cc.slot_partition()
    fve_full = [0.0, 0.0]; fve_shared = [0.0, 0.0]
    
    num_full = [0.0, 0.0]; num_shared = [0.0, 0.0]; den = [0.0, 0.0]
    mean = [A_ho[m].mean(0).to(device) for m in range(2)]
    act_count = torch.zeros(cc.d_sae, device=device)      
    act_sum = torch.zeros(cc.d_sae, device=device)        
    
    for b in range(0, N, batch):
        a = [A_ho[m][b:b + batch].to(device) for m in range(2)]
        a_full = F.relu(cc.pre_acts(a))
        z = cc.batch_topk(a_full)
        ah_full = cc.decode(z)
        ah_shared = cc.decode(z, exclude="exclusive")
        for m in range(2):
            num_full[m] += (a[m] - ah_full[m]).pow(2).sum().item()
            num_shared[m] += (a[m] - ah_shared[m]).pow(2).sum().item()
            den[m] += (a[m] - mean[m]).pow(2).sum().item()
        act_count += (z > 0).float().sum(0)
        act_sum += z.sum(0)
    for m in range(2):
        fve_full[m] = 1.0 - num_full[m] / (den[m] + 1e-9)
        fve_shared[m] = 1.0 - num_shared[m] / (den[m] + 1e-9)
    alive = act_count > 0
    def pcount(mask):
        mm = mask.to(device)
        return {"n": int(mm.sum().item()), "n_alive": int((mm & alive).sum().item()),
                "alive_frac": float((mm & alive).float().sum().item() / max(int(mm.sum().item()), 1))}

    def top_features(part_mask, n):
        mm = (part_mask.to(device) & alive)
        score = act_sum * mm.float()
        order = torch.argsort(score, descending=True)[:n]
        return [int(i) for i in order if score[int(i)] > 0]

    def max_act_tokens(feat_ids, topk=8, ctx=4):
        out = {}
        ids_flat = ids_ho  
        for fid in feat_ids:
            acts_f = torch.empty(N)
            for b in range(0, N, batch):
                a = [A_ho[m][b:b + batch].to(device) for m in range(2)]
                z = cc.batch_topk(F.relu(cc.pre_acts(a)))
                acts_f[b:b + batch] = z[:, fid].cpu()
            top = torch.argsort(acts_f, descending=True)[:topk]
            ex = []
            for j in top:
                j = int(j)
                if acts_f[j] <= 0:
                    continue
                lo = max(0, j - ctx)
                ctx_ids = ids_flat[lo:j + 1].tolist()
                txt = tok.decode(ctx_ids).replace("\n", " ")
                ex.append({"act": round(float(acts_f[j]), 3),
                           "tok": tok.decode([int(ids_flat[j])]), "ctx": txt[-60:]})
            out[str(fid)] = ex
        return out

    ia_top = top_features(part["I_A"], TOPN_MAXACT)
    is_top = top_features(part["I_S"], TOPN_MAXACT)
    return {
        "fve_full_bitnet": fve_full[0], "fve_full_fp": fve_full[1],
        "fve_shared_only_bitnet": fve_shared[0], "fve_shared_only_fp": fve_shared[1],
        "excl_gain_bitnet": fve_full[0] - fve_shared[0],
        "excl_gain_fp": fve_full[1] - fve_shared[1],
        "partition": {"I_A_bitnet_excl": pcount(part["I_A"]),
                      "I_B_fp_excl": pcount(part["I_B"]),
                      "I_S_shared": pcount(part["I_S"])},
        "maxact_bitnet_excl": max_act_tokens(ia_top),
        "maxact_shared": max_act_tokens(is_top),
    }

def _status(s):
    os.makedirs(OUT, exist_ok=True); open(STATUS, "w").write(s + "\n")

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
        d0, d1 = m_bit.config.hidden_size, m_fp.config.hidden_size
        assert m_bit.config.vocab_size == m_fp.config.vocab_size, "vocab mismatch"
        print("bitnet d=%d L%d | %s d=%d L%d | vocab %d"
              % (d0, BITNET_LAYER, FP_MODEL, d1, FP_LAYER, m_bit.config.vocab_size), flush=True)

        _status("COLLECT")
        ids = _token_windows(tok, N_SEQS, SEQ_LEN)
        A = collect_paired([m_bit, m_fp], [BITNET_LAYER, FP_LAYER], ids, N_ACT, dev)
        ids_flat = ids.reshape(-1)[:A[0].shape[0]].contiguous()   
        del m_bit, m_fp; torch.cuda.empty_cache()
        
        n = A[0].shape[0]; nho = min(HELDOUT, n // 5)
        A_tr = [A[m][:n - nho].contiguous() for m in range(2)]
        A_ho = [A[m][n - nho:].contiguous() for m in range(2)]
        ids_ho = ids_flat[n - nho:].contiguous()
        scales = standardize_fit_apply(A_tr, A_ho)              
        del A
        print("collected: train %d ho %d | scales %s"
              % (n - nho, nho, [round(s, 3) for s in scales]), flush=True)

        _status("TRAIN DFC")
        cc, hist = train_dfc(A_tr, [d0, d1], dev)
        torch.save(cc.state_dict(), os.path.join(OUT, "dfc_scale.pt"))

        _status("TRAIN STD-CC (baseline)")
        std_cc, std_hist, std_metrics = train_crosscoder(A_tr, [d0, d1], dev)

        _status("ANALYZE")
        ana = analyze_dfc(cc, A_ho, ids_ho, tok, dev)

        result = {
            "method": "DFC", "ref": "arXiv:2602.11729",
            "models": {"bitnet": BITNET, "fp": FP_MODEL}, "layers": [BITNET_LAYER, FP_LAYER],
            "dims": [d0, d1], "d_sae": D_SAE, "excl_frac": EXCL_FRAC, "k": K,
            "n_act": N_ACT, "n_heldout": nho, "seq_len": SEQ_LEN, "cc_steps": CC_STEPS,
            "scales": scales, "dfc": ana, "dfc_history": hist,
            "std_crosscoder": {"metrics": std_metrics, "history": std_hist},
            "wall_s": time.time() - t0,
        }
        json.dump(result, open(OUT_JSON + ".tmp", "w"), indent=2)
        os.replace(OUT_JSON + ".tmp", OUT_JSON)

        print("\n==== DFC SCALED DIFF (BitNet-2B vs %s) ====" % FP_MODEL, flush=True)
        print("DFC held-out FVE: bitnet full %.3f (shared-only %.3f, excl gain %.3f) | "
              "fp full %.3f (shared-only %.3f, excl gain %.3f)"
              % (ana["fve_full_bitnet"], ana["fve_shared_only_bitnet"], ana["excl_gain_bitnet"],
                 ana["fve_full_fp"], ana["fve_shared_only_fp"], ana["excl_gain_fp"]))
        for kk, v in ana["partition"].items():
            print("  %-16s alive %d/%d (%.2f)" % (kk, v["n_alive"], v["n"], v["alive_frac"]))
        print("STD crosscoder (confounded baseline): bitnet-excl %.3f fp-excl %.3f r_med %.3f"
              % (std_metrics["frac_bitnet_excl"], std_metrics["frac_fp_excl"], std_metrics["r_median"]))
        _status("DONE %.0fs" % (time.time() - t0))
        print("\nDONE %.1fs" % (time.time() - t0), flush=True)
    except Exception:
        tb = traceback.format_exc()
        print(tb, flush=True)
        _status("ERROR\n" + tb)
        raise

if __name__ == "__main__":
    main()
