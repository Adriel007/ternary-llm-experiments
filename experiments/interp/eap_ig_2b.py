
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

MID = "microsoft/bitnet-b1.58-2B-4T-bf16"
OUT = os.path.join(ROOT, "artifacts/poc")
DRIVE = "/content/drive/MyDrive/PhD_PoC"
M_IG = 7          
MIN_PAIRS = 6     

ANTONYMS = [
    ("hot", "cold"), ("up", "down"), ("big", "small"), ("fast", "slow"),
    ("day", "night"), ("left", "right"), ("true", "false"), ("black", "white"),
    ("high", "low"), ("good", "bad"), ("rich", "poor"), ("hard", "soft"),
    ("wet", "dry"), ("open", "closed"), ("light", "dark"), ("old", "young"),
]
CAPITALS = [
    ("France", "Paris"), ("Japan", "Tokyo"), ("Italy", "Rome"), ("Spain", "Madrid"),
    ("China", "Beijing"), ("Russia", "Moscow"), ("Germany", "Berlin"), ("Egypt", "Cairo"),
]

def _build_pairs(tok):
    def ids(s):
        return tok(s, return_tensors="pt").input_ids
    def ans_id(w):
        t = tok(" " + w, add_special_tokens=False).input_ids
        return t[0] if len(t) == 1 else None

    pairs = []
    
    for a, b in ANTONYMS:
        ca, cb = ans_id(b), ans_id(a)               
        if ca is None or cb is None:
            continue
        ci, ki = ids(f"The opposite of {a} is"), ids(f"The opposite of {b} is")
        if ci.shape[1] != ki.shape[1]:
            continue
        pairs.append({"family": "antonym", "clean_text": f"The opposite of {a} is",
                      "corrupt_text": f"The opposite of {b} is",
                      "clean_ids": ci, "corrupt_ids": ki, "ans_clean": ca, "ans_corrupt": cb})
    
    for i in range(0, len(CAPITALS) - 1, 2):
        (A, ansA), (B, ansB) = CAPITALS[i], CAPITALS[i + 1]
        ca, cb = ans_id(ansA), ans_id(ansB)
        if ca is None or cb is None:
            continue
        ci, ki = ids(f"The capital of {A} is"), ids(f"The capital of {B} is")
        if ci.shape[1] != ki.shape[1]:
            continue
        pairs.append({"family": "capital", "clean_text": f"The capital of {A} is",
                      "corrupt_text": f"The capital of {B} is",
                      "clean_ids": ci, "corrupt_ids": ki, "ans_clean": ca, "ans_corrupt": cb})
    return pairs

def _node_modules(model):
    nodes = [("emb", model.model.embed_tokens)]
    for i, layer in enumerate(model.model.layers):
        nodes.append((f"a{i}", layer.self_attn))
        nodes.append((f"m{i}", layer.mlp))
    return nodes

def _out0(o):
    return o[0] if isinstance(o, tuple) else o

def _metric(logits_row, ans_clean, ans_corrupt):
    return (logits_row[ans_clean] - logits_row[ans_corrupt]).float()

@torch.no_grad()
def _cache_run(model, nodes, ids, ans_c, ans_k):
    cache = {}

    def mk(k):
        def hook(m, i, o):
            cache[k] = _out0(o).detach()          
        return hook

    handles = [mod.register_forward_hook(mk(k)) for k, mod in nodes]
    out = model(input_ids=ids)
    for h in handles:
        h.remove()
    L = _metric(out.logits[0, -1], ans_c, ans_k).item()
    return cache, L

@torch.no_grad()
def _patched_metric(model, nodes, base_ids, patch_acts, which, ans_c, ans_k):
    sel = set(which)

    def mk(k):
        def hook(m, i, o):
            if k not in sel:
                return None
            rep = patch_acts[k]
            if isinstance(o, tuple):
                return (rep,) + tuple(o[1:])
            return rep
        return hook

    handles = [mod.register_forward_hook(mk(k)) for k, mod in nodes]
    out = model(input_ids=base_ids)
    for h in handles:
        h.remove()
    return _metric(out.logits[0, -1], ans_c, ans_k).item()

def _exact_effects(model, nodes, pair):
    clean_cache, L_clean = _cache_run(model, nodes, pair["clean_ids"], pair["ans_clean"], pair["ans_corrupt"])
    _, L_corrupt = _cache_run(model, nodes, pair["corrupt_ids"], pair["ans_clean"], pair["ans_corrupt"])
    eff = {}
    for k, _ in nodes:
        Lk = _patched_metric(model, nodes, pair["corrupt_ids"], clean_cache, [k],
                             pair["ans_clean"], pair["ans_corrupt"])
        eff[k] = Lk - L_corrupt
    return eff, L_clean, L_corrupt, clean_cache

def _eap_ig_scores(model, nodes, pair, clean_cache, corrupt_cache, m=M_IG):
    emb = model.model.embed_tokens
    with torch.no_grad():
        e_clean = emb(pair["clean_ids"])           
        e_corrupt = emb(pair["corrupt_ids"])
    diff = (e_clean - e_corrupt)                    

    grad_acc = {k: None for k, _ in nodes if k != "emb"}
    emb_grad_acc = None

    def mk(k):
        def fhook(m, i, o):
            t = _out0(o)
            def save(g, k=k):
                grad_acc[k] = g.detach() if grad_acc[k] is None else grad_acc[k] + g.detach()
            t.register_hook(save)
        return fhook

    handles = [mod.register_forward_hook(mk(k)) for k, mod in nodes if k != "emb"]
    for step in range(1, m + 1):
        alpha = step / m
        inp = (e_corrupt + alpha * diff).detach().requires_grad_(True)
        out = model(inputs_embeds=inp)
        L = _metric(out.logits[0, -1], pair["ans_clean"], pair["ans_corrupt"])
        model.zero_grad(set_to_none=True)
        L.backward()
        emb_grad_acc = inp.grad.detach() if emb_grad_acc is None else emb_grad_acc + inp.grad.detach()
    for h in handles:
        h.remove()

    scores = {}
    for k, _ in nodes:
        if k == "emb":
            g = emb_grad_acc / m
            scores[k] = float(((e_clean - e_corrupt) * g).sum().item())
        else:
            g = grad_acc[k] / m
            scores[k] = float(((clean_cache[k] - corrupt_cache[k]) * g).sum().item())
    return scores

def _selftest(model, nodes, pairs) -> None:
    p = pairs[0]
    clean_cache, L_clean = _cache_run(model, nodes, p["clean_ids"], p["ans_clean"], p["ans_corrupt"])
    _, L_corrupt = _cache_run(model, nodes, p["corrupt_ids"], p["ans_clean"], p["ans_corrupt"])
    L_full = _patched_metric(model, nodes, p["corrupt_ids"], clean_cache, [k for k, _ in nodes],
                             p["ans_clean"], p["ans_corrupt"])
    dfull = abs(L_full - L_clean)
    assert dfull < 2.0, f"full-patch did not reconstruct clean (|L_full-L_clean|={dfull:.3f}) -- patch wiring wrong"
    print("SELFTEST full-patch OK | L_clean=%.3f L_corrupt=%.3f L_full=%.3f |Δ|=%.4f"
          % (L_clean, L_corrupt, L_full, dfull))

def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    status = os.path.join(OUT, "EAPIG_STATUS.txt")
    open(status, "w").write("LOADING\n")
    t0 = time.time()
    try:
        import inspect
        from transformers import AutoModelForCausalLM, AutoTokenizer
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        tok = AutoTokenizer.from_pretrained(MID)
        model = AutoModelForCausalLM.from_pretrained(MID, dtype=torch.bfloat16).to(dev).eval()
        model.requires_grad_(False)               
        n_layers = len(model.model.layers)
        nodes = _node_modules(model)
        print("loaded", MID, "| layers", n_layers, "| nodes", len(nodes), "| dev", dev)

        
        try:
            bl = None
            for mod in model.modules():
                if mod.__class__.__name__.lower().startswith("bitlinear") or "BitLinear" in mod.__class__.__name__:
                    bl = mod
                    break
            if bl is not None:
                print("=== BitLinear class:", bl.__class__.__name__, "===")
                print(inspect.getsource(bl.__class__.forward))
        except Exception as e:
            print("could not introspect BitLinear:", e)

        raw = _build_pairs(tok)
        for p in raw:
            p["clean_ids"] = p["clean_ids"].to(dev)
            p["corrupt_ids"] = p["corrupt_ids"].to(dev)
        pairs = []
        for p in raw:
            _, Lc = _cache_run(model, nodes, p["clean_ids"], p["ans_clean"], p["ans_corrupt"])
            _, Lk = _cache_run(model, nodes, p["corrupt_ids"], p["ans_clean"], p["ans_corrupt"])
            p["L_clean"], p["L_corrupt"] = Lc, Lk
            if Lc > 0 and Lk < 0:                  
                pairs.append(p)
        print("pairs: %d built, %d solved (kept)" % (len(raw), len(pairs)))
        for p in pairs:
            print("  [%s] %-28s L_clean=%+.2f  /  %-28s L_corrupt=%+.2f"
                  % (p["family"], p["clean_text"], p["L_clean"], p["corrupt_text"], p["L_corrupt"]))
        assert len(pairs) >= MIN_PAIRS, f"only {len(pairs)} pairs survived filtering (<{MIN_PAIRS}); aborting"

        open(status, "w").write("SELFTEST\n")
        _selftest(model, nodes, pairs)
        open(status, "w").write("RUNNING\n")

        node_keys = [k for k, _ in nodes]
        comp_keys = [k for k in node_keys if k != "emb"]
        K_list = [1, 3, 5, 10, 20, 30]
        exact = {k: [] for k in node_keys}         
        eapig = {k: [] for k in node_keys}         
        ig_completeness = []
        recovery = {K: [] for K in K_list}         
        for j, p in enumerate(pairs):
            denom = p["L_clean"] - p["L_corrupt"]
            eff, Lc, Lk, clean_cache = _exact_effects(model, nodes, p)
            corrupt_cache, _ = _cache_run(model, nodes, p["corrupt_ids"], p["ans_clean"], p["ans_corrupt"])
            with torch.enable_grad():
                sc = _eap_ig_scores(model, nodes, p, clean_cache, corrupt_cache)
            for k in node_keys:
                exact[k].append(eff[k] / denom)
                eapig[k].append(sc[k] / denom)

            
            ig_completeness.append(sc["emb"] / denom)

            
            rank = sorted(comp_keys, key=lambda k: abs(eff[k]), reverse=True)
            for K in K_list:
                Lj = _patched_metric(model, nodes, p["corrupt_ids"], clean_cache, rank[:K],
                                     p["ans_clean"], p["ans_corrupt"])
                recovery[K].append((Lj - Lk) / denom)
            print("  pair %d/%d done (%s)" % (j + 1, len(pairs), p["family"]))

        exact_mean = {k: float(np.mean(v)) for k, v in exact.items()}
        eapig_mean = {k: float(np.mean(v)) for k, v in eapig.items()}

        ex = np.array([exact_mean[k] for k in comp_keys])
        ei = np.array([eapig_mean[k] for k in comp_keys])
        from scipy.stats import spearmanr, pearsonr
        rho_s = float(spearmanr(ex, ei).correlation)
        rho_p = float(pearsonr(ex, ei)[0])

        order = sorted(comp_keys, key=lambda k: abs(exact_mean[k]), reverse=True)
        topk_recovery = {str(K): float(np.mean(recovery[K])) for K in K_list}

        
        layer_imp = [abs(exact_mean[f"a{i}"]) + abs(exact_mean[f"m{i}"]) for i in range(n_layers)]
        tot = sum(layer_imp) or 1.0
        cdf = np.cumsum(layer_imp) / tot
        median_imp_depth = int(np.searchsorted(cdf, 0.5))

        out = {
            "config": {"model": MID, "n_layers": n_layers, "n_nodes": len(nodes),
                       "m_ig": M_IG, "n_pairs": len(pairs),
                       "families": sorted({p["family"] for p in pairs})},
            "pairs": [{"family": p["family"], "clean": p["clean_text"], "corrupt": p["corrupt_text"],
                       "L_clean": p["L_clean"], "L_corrupt": p["L_corrupt"]} for p in pairs],
            "exact_effect_mean": exact_mean,         
            "eapig_score_mean": eapig_mean,          
            "faithfulness": {"spearman_eapig_vs_exact": rho_s, "pearson_eapig_vs_exact": rho_p,
                             "ig_completeness_mean": float(np.mean(ig_completeness)),
                             "ig_completeness_std": float(np.std(ig_completeness))},
            "topk_recovery_exact": topk_recovery,
            "depth": {"layer_importance_exact": layer_imp, "cdf": cdf.tolist(),
                      "median_importance_depth": median_imp_depth,
                      "emb_effect": exact_mean["emb"]},
            "top_nodes_exact": [{"node": k, "exact": exact_mean[k], "eapig": eapig_mean[k]} for k in order[:12]],
            "wall_s": time.time() - t0,
        }
        path = os.path.join(OUT, "eap_ig_2b.json")
        json.dump(out, open(path, "w"), indent=2)
        if os.path.isdir(DRIVE):
            try:
                json.dump(out, open(os.path.join(DRIVE, "eap_ig_2b.json"), "w"), indent=2)
            except Exception as e:
                print("drive save failed:", e)

        open(status, "w").write("DONE %.0fs rho_s=%.3f med_depth=%d/%d\n"
                                % (time.time() - t0, rho_s, median_imp_depth, n_layers))
        print("DONE %.1fs" % (time.time() - t0))
        print("faithfulness EAP-IG vs exact: spearman=%.3f pearson=%.3f | IG completeness=%.2f"
              % (rho_s, rho_p, np.mean(ig_completeness)))
        print("top-k exact recovery:", topk_recovery)
        print("median importance depth = %d/%d (logit-lens crystallization was 22/30)" % (median_imp_depth, n_layers))
        print("top nodes:", [(d["node"], round(d["exact"], 3)) for d in out["top_nodes_exact"]])
    except Exception:
        tb = traceback.format_exc()
        print(tb)
        open(status, "w").write("ERROR\n" + tb)
        raise

if __name__ == "__main__":
    main()
