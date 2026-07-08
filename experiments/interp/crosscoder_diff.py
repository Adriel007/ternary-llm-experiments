
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
import torch.nn as nn  
import torch.nn.functional as F  
from torch import autocast  

from experiments.dynamics.data import build_tinystories_tokens  
from experiments.dynamics.experiment import PoCConfig, train_one  

LAYER = 4                 
DEPTH_LAYERS = [1, 3, 5, 7]  
SEED = 0
TRAIN_STEPS = 3000        
SEQ_LEN = 256
N_ACT = 250_000           
ACT_BATCH = 16
D_SAE = 4096              
K = 32                    
CC_STEPS = 4000
CC_BATCH = 4096
CC_LR = 3e-4
AUX_COEF = 1.0 / 32
DEAD_STEPS = 200
OUT = os.path.join(ROOT, "artifacts/poc")
DRIVE = "/content/drive/MyDrive/PhD_PoC"
MODELS = ("ternary", "fp")   

class CrossCoder(nn.Module):

    def __init__(self, d_in: int, d_sae: int, k: int, n_models: int = 2, seed: int = 0):
        super().__init__()
        self.d_in, self.d_sae, self.k, self.n_models = d_in, d_sae, k, n_models
        g = torch.Generator().manual_seed(seed)
        
        Wd = torch.randn(n_models, d_sae, d_in, generator=g) / (d_in ** 0.5)
        self.W_dec = nn.Parameter(Wd)

        self.W_enc = nn.Parameter(
            torch.randn(n_models * d_in, d_sae, generator=g) / (n_models * d_in) ** 0.5)
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.b_dec = nn.Parameter(torch.zeros(n_models, d_in))
        self.register_buffer("last_active", torch.zeros(d_sae, dtype=torch.long))
        self.normalize_decoder_()

    def pre_acts(self, a):                       
        xc = (a - self.b_dec[:, None, :]).permute(1, 0, 2).reshape(a.shape[1], -1)   
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
        return torch.einsum("bs,msd->mbd", z, self.W_dec) + self.b_dec[:, None, :]

    def forward(self, a):
        pre = self.pre_acts(a)
        acts = F.relu(pre)
        z = self.batch_topk(acts)
        a_hat = self.decode(z)
        return a_hat, z, acts

    @torch.no_grad()
    def normalize_decoder_(self):
        
        norm = self.W_dec.pow(2).sum(dim=(0, 2), keepdim=True).sqrt().clamp_min(1e-8)
        self.W_dec.div_(norm)

def _aux_loss(cc, acts, residual, dead):
    if cc.k <= 0 or dead.sum() == 0:
        return residual.new_zeros(())
    kk = min(256, int(dead.sum().item()))
    acts_dead = acts.clone()
    acts_dead[:, ~dead] = 0.0
    topv, topi = acts_dead.topk(kk, dim=-1)
    z_aux = torch.zeros_like(acts_dead).scatter_(-1, topi, topv)
    res_hat = torch.einsum("bs,msd->mbd", z_aux, cc.W_dec)
    return F.mse_loss(res_hat, residual)

def train_crosscoder(A, device="cuda", log_every=500):
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(SEED)
    M, N, d = A.shape
    cc = CrossCoder(d, D_SAE, K, n_models=M, seed=SEED).to(device)
    cc.b_dec.data = A[:, :65536].mean(1).to(device)         
    opt = torch.optim.Adam(cc.parameters(), lr=CC_LR)
    gen = torch.Generator().manual_seed(SEED + 1)           
    hist = {"step": [], "fve_tern": [], "fve_fp": [], "l0": [], "dead": []}
    for step in range(CC_STEPS):
        idx = torch.randint(N, (CC_BATCH,), generator=gen)
        a = A[:, idx].to(device)                            
        a_hat, z, acts_full = cc(a)
        recon = F.mse_loss(a_hat, a)
        active = (z > 0).any(0)
        cc.last_active[active] = 0
        cc.last_active[~active] += 1
        dead = cc.last_active > DEAD_STEPS
        aux = _aux_loss(cc, acts_full, (a - a_hat).detach(), dead)
        loss = recon + AUX_COEF * aux
        opt.zero_grad(set_to_none=True)
        loss.backward()
        cc.normalize_decoder_()
        opt.step()
        if step % log_every == 0 or step == CC_STEPS - 1:
            with torch.no_grad():
                fve = []
                for m in range(M):
                    num = (a[m] - a_hat[m]).pow(2).sum().item()
                    den = (a[m] - a[m].mean(0)).pow(2).sum().item() + 1e-9
                    fve.append(1.0 - num / den)
                l0 = (z > 0).float().sum(-1).mean().item()
            hist["step"].append(step); hist["fve_tern"].append(fve[0]); hist["fve_fp"].append(fve[1])
            hist["l0"].append(l0); hist["dead"].append(int(dead.sum().item()))
            print(f"  cc step {step}: fve_t={fve[0]:.3f} fve_fp={fve[1]:.3f} l0={l0:.1f} "
                  f"dead={int(dead.sum().item())}", flush=True)
    metrics = classify_features(cc)
    metrics["final_fve_tern"] = hist["fve_tern"][-1]
    metrics["final_fve_fp"] = hist["fve_fp"][-1]
    metrics["final_l0"] = hist["l0"][-1]
    return cc, hist, metrics

@torch.no_grad()
def classify_features(cc, freq=None, alive_thresh=1e-3):
    Wd = cc.W_dec.detach()                              
    n_tern = Wd[0].norm(dim=-1)                         
    n_fp = Wd[1].norm(dim=-1)
    total = (n_tern + n_fp)
    alive = total > alive_thresh
    r = (n_tern / total.clamp_min(1e-9))               
    r_alive = r[alive]
    cos = F.cosine_similarity(Wd[0], Wd[1], dim=-1)     
    
    shared = alive & (r > 0.35) & (r < 0.65)
    tern_excl = alive & (r >= 0.8)
    fp_excl = alive & (r <= 0.2)
    n_alive = int(alive.sum().item())
    hist_edges = np.linspace(0, 1, 21)
    hist_counts = np.histogram(r_alive.cpu().numpy(), bins=hist_edges)[0].tolist()
    return {
        "d_sae": int(cc.d_sae), "k": int(cc.k),
        "n_alive": n_alive, "n_dead": int((~alive).sum().item()),
        "n_shared": int(shared.sum().item()),
        "n_tern_excl": int(tern_excl.sum().item()),
        "n_fp_excl": int(fp_excl.sum().item()),
        "frac_shared": float(shared.sum().item() / max(n_alive, 1)),
        "frac_tern_excl": float(tern_excl.sum().item() / max(n_alive, 1)),
        "frac_fp_excl": float(fp_excl.sum().item() / max(n_alive, 1)),
        "r_mean": float(r_alive.mean().item()), "r_median": float(r_alive.median().item()),
        "cos_shared_mean": float(cos[shared].mean().item()) if shared.any() else None,
        "cos_shared_median": float(cos[shared].median().item()) if shared.any() else None,
        "r_hist_edges": hist_edges.tolist(), "r_hist_counts": hist_counts,
    }

@torch.no_grad()
def collect_paired(models, ids, layer_idx, n_target, device, amp_dtype):
    stores = [[] for _ in models]

    def mk_hook(buf):
        def hook(m, i, o):
            h = o[0] if isinstance(o, tuple) else o
            buf.append(h.reshape(-1, h.shape[-1]).float().cpu())
        return hook

    handles = [mdl.model.layers[layer_idx].register_forward_hook(mk_hook(stores[k]))
               for k, mdl in enumerate(models)]
    n_seq = ids.shape[0]
    seen = 0
    for b in range(0, n_seq, ACT_BATCH):
        batch = ids[b: b + ACT_BATCH].to(device)
        with autocast(device_type="cuda", dtype=amp_dtype, enabled=device == "cuda"):
            for mdl in models:
                mdl(input_ids=batch)
        seen += stores[0][-1].shape[0]
        if seen >= n_target:
            break
    for h in handles:
        h.remove()
    A = torch.stack([torch.cat(s, 0)[:n_target].contiguous() for s in stores], 0)  
    return A

def standardize_(A):
    M, N, d = A.shape
    mean = A.mean(dim=1, keepdim=True)
    A.sub_(mean)
    scale = (A.pow(2).sum(dim=2).mean(dim=1, keepdim=True) / d).sqrt().clamp_min(1e-8)  
    A.div_(scale[:, :, None])
    return {"scale_tern": float(scale[0].item()), "scale_fp": float(scale[1].item())}

def main():
    os.makedirs(OUT, exist_ok=True)

    
    mode = sys.argv[1] if (len(sys.argv) > 1 and sys.argv[1] in ("control", "depth", "poscontrol")) else "diff"
    if mode == "control":
        specs = [(False, 0, "fp_s0"), (False, 1, "fp_s1")]
        out_json, pt_name, status = "crosscoder_control.json", "crosscoder_control_L%d.pt" % LAYER, os.path.join(OUT, "CC_CTRL_STATUS.txt")
    elif mode == "poscontrol":      

        
        specs = [(False, SEED, "fp_tinystories"), (False, SEED, "fp_wikitext")]
        out_json, pt_name, status = "crosscoder_poscontrol.json", "crosscoder_poscontrol_L%d.pt" % LAYER, os.path.join(OUT, "CC_POS_STATUS.txt")
    elif mode == "depth":           
        specs = [(True, SEED, "ternary"), (False, SEED, "fp")]
        out_json, pt_name, status = "crosscoder_depth.json", None, os.path.join(OUT, "CC_DEPTH_STATUS.txt")
    else:
        specs = [(True, SEED, "ternary"), (False, SEED, "fp")]
        out_json, pt_name, status = "crosscoder_diff.json", "crosscoder_L%d.pt" % LAYER, os.path.join(OUT, "CC_STATUS.txt")
    open(status, "w").write("RUNNING\n")
    t0 = time.time()
    try:
        dev = "cuda"
        print("MODE:", mode, "| models:", [s[2] for s in specs], flush=True)
        cfg = PoCConfig(seeds=(SEED,), steps=TRAIN_STEPS, seq_len=SEQ_LEN,
                        n_train_tokens=8_000_000, n_val_tokens=1_000_000,
                        out_dir=OUT)
        from experiments.dynamics.data import TokenLoader, build_wikitext_tokens
        tr, va = build_tinystories_tokens(cfg.data_dir, cfg.n_train_tokens, cfg.n_val_tokens)
        loaders = (TokenLoader(tr, cfg.seq_len, dev), TokenLoader(va, cfg.seq_len, dev))

        
        
        if mode == "poscontrol":

            
            WIKI_VAL_TOKENS = 200_000
            tr2, va2 = build_wikitext_tokens(cfg.data_dir, cfg.n_train_tokens, WIKI_VAL_TOKENS)
            loaders_wiki = (TokenLoader(tr2, cfg.seq_len, dev), TokenLoader(va2, cfg.seq_len, dev))
            spec_loaders = [loaders, loaders_wiki]   
        else:
            spec_loaders = [loaders] * len(specs)

        
        
        models, twin_val = [], {}
        for (quant, seed, key), ld in zip(specs, spec_loaders):
            r = train_one(quant, seed, cfg, ld)
            models.append(r["model"].eval())
            twin_val[key] = float(r["final_val_loss"])
            print(f"trained {key}: final_val_loss={r['final_val_loss']:.4f}", flush=True)

        if mode == "poscontrol":

            
            n_ts = (len(va) // SEQ_LEN); n_wk = (len(va2) // SEQ_LEN); n_each = min(n_ts, n_wk)
            ids_ts = torch.tensor(va[: n_each * SEQ_LEN], dtype=torch.long).view(-1, SEQ_LEN)
            ids_wk = torch.tensor(va2[: n_each * SEQ_LEN], dtype=torch.long).view(-1, SEQ_LEN)
            val_ids = torch.stack([ids_ts, ids_wk], dim=1).reshape(-1, SEQ_LEN)  
        else:
            val_ids = torch.tensor(va[: (len(va) // SEQ_LEN) * SEQ_LEN], dtype=torch.long).view(-1, SEQ_LEN)

        if mode == "depth":
            layers_out = []
            for L in DEPTH_LAYERS:
                A = collect_paired(models, val_ids, L, N_ACT, dev, cfg.amp_dtype)
                scales = standardize_(A)
                cc, hist, metrics = train_crosscoder(A, device=dev)
                layers_out.append({"layer": L, "scales": scales, "metrics": metrics})
                print("[layer %d] cos_med=%.3f shared=%d tern_excl=%d fve_t=%.3f" % (
                    L, metrics["cos_shared_median"], metrics["n_shared"],
                    metrics["n_tern_excl"], metrics["final_fve_tern"]), flush=True)
                res = {"setup": {"mode": "depth", "seed": SEED, "twin_steps": TRAIN_STEPS,
                                 "layers": DEPTH_LAYERS, "d_in": int(A.shape[2]),
                                 "models": [s[2] for s in specs],
                                 "crosscoder": {"d_sae": D_SAE, "k": K, "steps": CC_STEPS}},
                       "twin_val_loss": twin_val, "results": layers_out, "wall_s": time.time() - t0}
                json.dump(res, open(os.path.join(OUT, out_json), "w"), indent=2)
                if os.path.isdir(DRIVE):
                    json.dump(res, open(os.path.join(DRIVE, out_json), "w"), indent=2)
                del A, cc; torch.cuda.empty_cache()
            open(status, "w").write("DONE %.0fs\n" % (time.time() - t0))
            print("CROSSCODER DEPTH DONE", round(time.time() - t0, 1), "s", flush=True)
            return

        A = collect_paired(models, val_ids, LAYER, N_ACT, dev, cfg.amp_dtype)
        print(f"paired acts: {tuple(A.shape)}", flush=True)
        scales = standardize_(A)

        cc, hist, metrics = train_crosscoder(A, device=dev)
        print("DIFF metrics:", json.dumps({k: metrics[k] for k in
              ("n_alive", "n_shared", "n_tern_excl", "n_fp_excl", "r_median",
               "cos_shared_median", "final_fve_tern", "final_fve_fp")}), flush=True)

        result = {
            "setup": {"mode": mode, "layer": LAYER, "seed": SEED, "twin_steps": TRAIN_STEPS,
                      "d_in": int(A.shape[2]), "n_act": int(A.shape[1]),
                      "models": [s[2] for s in specs], "scales": scales,
                      "crosscoder": {"d_sae": D_SAE, "k": K, "steps": CC_STEPS,
                                     "expansion": D_SAE / A.shape[2]}},
            "twin_val_loss": twin_val,
            "metrics": metrics, "history": hist, "wall_s": time.time() - t0,
        }
        json.dump(result, open(os.path.join(OUT, out_json), "w"), indent=2)
        torch.save(cc.state_dict(), os.path.join(OUT, pt_name))
        if os.path.isdir(DRIVE):
            json.dump(result, open(os.path.join(DRIVE, out_json), "w"), indent=2)
            torch.save(cc.state_dict(), os.path.join(DRIVE, pt_name))
        open(status, "w").write(f"DONE {time.time()-t0:.0f}s\n" + json.dumps(metrics))
        print(f"CROSSCODER {mode.upper()} DONE", round(time.time() - t0, 1), "s",
              "| cos_median=%.3f" % metrics["cos_shared_median"] if metrics["cos_shared_median"] else "", flush=True)
    except Exception:
        tb = traceback.format_exc(); print(tb)
        open(status, "w").write("ERROR\n" + tb)
        raise

if __name__ == "__main__":
    main()
