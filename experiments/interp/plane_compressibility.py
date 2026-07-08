from __future__ import annotations

import json
import os
import sys

ROOT = os.environ.get("POC_ROOT", "/workspace/PhD")
sys.path[:0] = [ROOT, os.path.join(ROOT, "sasori/src")]

import torch  
from sasori.reconstruct import quantize_matrix_k  

MODEL = os.environ.get("PC_MODEL", "Qwen/Qwen2.5-0.5B")
GROUP = int(os.environ.get("PC_GROUP", "256"))
KVALS = [int(k) for k in os.environ.get("PC_KVALS", "2,3").split(",")]
NMAT = int(os.environ.get("PC_NMAT", "24"))          
OUT = os.environ.get("PC_OUT", os.path.join(ROOT, "sasori/bench", "plane_compressibility.json"))
LOG2_3 = 1.5849625007211562                            

def _ent_bits(counts: torch.Tensor) -> float:
    p = counts.double()
    p = p / p.sum()
    nz = p[p > 0]
    return float(-(nz * nz.log2()).sum())

def plane_stats(codes: list[torch.Tensor]) -> dict:
    K = len(codes)
    
    sym = [(c.reshape(-1).to(torch.int64) + 1) for c in codes]   
    n = sym[0].numel()
    Hk, zerof = [], []
    for s in sym:
        cnt = torch.bincount(s, minlength=3).float()
        Hk.append(_ent_bits(cnt))
        zerof.append(float((s == 1).float().mean()))             
    
    mi_pairs = {}
    for i in range(K):
        for j in range(i + 1, K):
            joint = torch.bincount(3 * sym[i] + sym[j], minlength=9).float().reshape(3, 3)
            pj = joint / joint.sum()
            pi = pj.sum(1, keepdim=True)
            pjm = pj.sum(0, keepdim=True)
            mask = pj > 0
            mi = float((pj[mask] * (pj[mask] / (pi.expand_as(pj)[mask] * pjm.expand_as(pj)[mask])).log2()).sum())
            mi_pairs[f"MI_{i}{j}"] = mi
    
    idx = torch.zeros(n, dtype=torch.int64)
    for s in sym:
        idx = idx * 3 + s
    Hjoint = _ent_bits(torch.bincount(idx, minlength=3 ** K).float())
    sumHk = float(sum(Hk))
    return {
        "H_per_plane": [round(h, 4) for h in Hk],
        "zero_frac": [round(z, 4) for z in zerof],
        "within_headroom_bits": round(sum(LOG2_3 - h for h in Hk), 4),   
        "sumH": round(sumHk, 4),
        "H_joint": round(Hjoint, 4),
        "cross_redundancy_bits": round(sumHk - Hjoint, 4),               
        **{k: round(v, 5) for k, v in mi_pairs.items()},
    }

def main():
    from transformers import AutoModelForCausalLM
    print(f"[pc] loading {MODEL}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32)
    model.eval()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    mats = [(n, p) for n, p in model.named_parameters()
            if p.dim() == 2 and min(p.shape) >= 64 and "embed" not in n and "lm_head" not in n]
    step = max(1, len(mats) // NMAT)
    picked = mats[::step][:NMAT]
    print(f"[pc] {len(mats)} eligible matrices; measuring {len(picked)}", flush=True)

    results = {"_meta": {"model": MODEL, "group": GROUP, "n_matrices": len(picked),
                         "log2_3": LOG2_3, "note": "A=within-plane (sparsity, trivial); "
                         "B=cross-plane redundancy (the novel #95 lane)"}}
    for K in KVALS:
        per_mat = []
        for n, p in picked:
            kp = quantize_matrix_k(p.data.to(dev), K, group=GROUP, row_chunk=65536)
            st = plane_stats([c.cpu() for c in kp.codes])
            st["name"] = n
            per_mat.append(st)
        
        agg = {
            "H_per_plane_mean": [round(sum(m["H_per_plane"][k] for m in per_mat) / len(per_mat), 4)
                                 for k in range(K)],
            "zero_frac_mean": [round(sum(m["zero_frac"][k] for m in per_mat) / len(per_mat), 4)
                               for k in range(K)],
            "within_headroom_bits_mean": round(sum(m["within_headroom_bits"] for m in per_mat) / len(per_mat), 4),
            "cross_redundancy_bits_mean": round(sum(m["cross_redundancy_bits"] for m in per_mat) / len(per_mat), 4),
            "cross_redundancy_frac_of_alloc": round(
                sum(m["cross_redundancy_bits"] for m in per_mat) / len(per_mat) / (K * LOG2_3), 5),
        }
        
        mi_keys = [k for k in per_mat[0] if k.startswith("MI_")]
        agg["MI_mean"] = {k: round(sum(m[k] for m in per_mat) / len(per_mat), 5) for k in mi_keys}
        results[f"K{K}"] = {"aggregate": agg, "per_matrix": per_mat}
        print(f"[pc] K{K}: H/plane={agg['H_per_plane_mean']} within={agg['within_headroom_bits_mean']}b "
              f"cross_redundancy={agg['cross_redundancy_bits_mean']}b "
              f"({agg['cross_redundancy_frac_of_alloc']*100:.2f}% of alloc) MI={agg['MI_mean']}", flush=True)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[pc] wrote {OUT}", flush=True)

if __name__ == "__main__":
    main()
