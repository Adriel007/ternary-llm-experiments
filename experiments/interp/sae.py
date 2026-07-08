
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class SAEConfig:
    d_in: int
    d_sae: int                 
    k: int                     
    lr: float = 3e-4
    steps: int = 4000
    batch_size: int = 4096
    aux_k: int = 256           
    aux_coef: float = 1.0 / 32
    dead_steps: int = 200      
    seed: int = 0

class BatchTopKSAE(nn.Module):
    def __init__(self, cfg: SAEConfig):
        super().__init__()
        self.cfg = cfg
        g = torch.Generator().manual_seed(cfg.seed)
        W = torch.randn(cfg.d_in, cfg.d_sae, generator=g) / (cfg.d_in ** 0.5)
        W = W / W.norm(dim=0, keepdim=True).clamp_min(1e-8)   
        self.W_dec = nn.Parameter(W.clone())
        self.W_enc = nn.Parameter(W.t().clone())             
        self.b_enc = nn.Parameter(torch.zeros(cfg.d_sae))
        self.b_dec = nn.Parameter(torch.zeros(cfg.d_in))
        self.register_buffer("last_active", torch.zeros(cfg.d_sae, dtype=torch.long))

    def pre_acts(self, x):
        return (x - self.b_dec) @ self.W_enc.t() + self.b_enc

    def batch_topk(self, acts):
        b = acts.shape[0]
        n = self.cfg.k * b
        flat = acts.flatten()
        if 0 < n < flat.numel():
            thresh = torch.topk(flat, n).values.min()
            acts = acts * (acts >= thresh)
        return acts

    def decode(self, z):
        return z @ self.W_dec.t() + self.b_dec

    def forward(self, x):
        pre = self.pre_acts(x)
        acts = F.relu(pre)
        z = self.batch_topk(acts)
        x_hat = self.decode(z)
        return x_hat, z, acts

    @torch.no_grad()
    def normalize_decoder_(self):
        self.W_dec.div_(self.W_dec.norm(dim=0, keepdim=True).clamp_min(1e-8))

def _aux_loss(sae, acts, residual, step_dead):
    cfg = sae.cfg
    dead = step_dead
    if cfg.aux_k <= 0 or dead.sum() == 0:
        return residual.new_zeros(())
    acts_dead = acts.clone()
    acts_dead[:, ~dead] = 0.0
    kk = min(cfg.aux_k, int(dead.sum().item()))
    topv, topi = acts_dead.topk(kk, dim=-1)
    z_aux = torch.zeros_like(acts_dead).scatter_(-1, topi, topv)
    res_hat = z_aux @ sae.W_dec.t()
    return F.mse_loss(res_hat, residual)

def train_sae(acts: torch.Tensor, cfg: SAEConfig, device: str = "cuda", log_every: int = 500):
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(cfg.seed)
    sae = BatchTopKSAE(cfg).to(device)
    sae.b_dec.data = acts[:65536].mean(0).to(device)       
    opt = torch.optim.Adam(sae.parameters(), lr=cfg.lr)
    N = acts.shape[0]
    gen = torch.Generator(device=device).manual_seed(cfg.seed + 1)   
    hist = {"step": [], "fve": [], "l0": [], "dead": []}
    for step in range(cfg.steps):
        idx = torch.randint(N, (cfg.batch_size,), generator=gen, device=device)
        x = acts[idx].to(device)
        x_hat, z, acts_full = sae(x)
        recon = F.mse_loss(x_hat, x)
        active = (z > 0).any(0)
        sae.last_active[active] = 0
        sae.last_active[~active] += 1
        dead = sae.last_active > cfg.dead_steps
        aux = _aux_loss(sae, acts_full, (x - x_hat).detach(), dead)
        loss = recon + cfg.aux_coef * aux
        opt.zero_grad(set_to_none=True)
        loss.backward()
        sae.normalize_decoder_()
        opt.step()
        if step % log_every == 0 or step == cfg.steps - 1:
            with torch.no_grad():
                fve = 1.0 - (x - x_hat).pow(2).sum().item() / ((x - x.mean(0)).pow(2).sum().item() + 1e-9)
                l0 = (z > 0).float().sum(-1).mean().item()
            hist["step"].append(step); hist["fve"].append(fve); hist["l0"].append(l0)
            hist["dead"].append(int(dead.sum().item()))
            print(f"  sae step {step}: fve={fve:.3f} l0={l0:.1f} dead={int(dead.sum().item())}", flush=True)
    try:
        final = evaluate_sae(sae, acts, device)
    except Exception as e:   
        print("eval failed, falling back to last training metrics:", repr(e), flush=True)
        final = {"fve": hist["fve"][-1], "l0": hist["l0"][-1], "dead": hist["dead"][-1],
                 "d_sae": cfg.d_sae, "k": cfg.k}
    return sae, hist, final

@torch.no_grad()
def evaluate_sae(sae: BatchTopKSAE, acts: torch.Tensor, device: str = "cuda", n_eval: int = 65536, bs: int = 4096):
    sae.eval()
    n = min(n_eval, acts.shape[0])
    x_all = acts[-n:]
    mean = x_all.to(device).float().mean(0)
    sse, tot, l0_sum, seen = 0.0, 0.0, 0.0, 0
    for b in range(0, n, bs):
        x = x_all[b : b + bs].to(device)
        x_hat, z, _ = sae(x)
        sse += (x - x_hat).pow(2).sum().item()
        tot += (x - mean).pow(2).sum().item()
        l0_sum += (z > 0).float().sum(-1).sum().item()
        seen += x.shape[0]
    fve = 1.0 - sse / (tot + 1e-9)
    dead = int((sae.last_active > sae.cfg.dead_steps).sum().item())
    return {"fve": float(fve), "l0": float(l0_sum / seen), "dead": dead, "d_sae": sae.cfg.d_sae, "k": sae.cfg.k}
