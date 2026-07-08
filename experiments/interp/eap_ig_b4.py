
from __future__ import annotations

import json
import os
import sys
import time
import traceback

ROOT = os.environ.get("PHD_ROOT", "/content/PhD-propose")
sys.path[:0] = [ROOT, os.path.join(ROOT, "packages/bitnet_core/src")]

import numpy as np  
import torch  

from experiments.interp.eap_ig_2b import (  
    MID, OUT, DRIVE, _node_modules,
)
from experiments.interp.eap_ig_circuits_2b import _effects_both, _solved  
from experiments.interp.eap_ig_heads_b6 import _build_ioi_multi  

MIN_PAIRS = 50
REF_JSON = os.environ.get("REF_JSON", os.path.join(ROOT, "reports/data/eap_ig_2b.json"))

def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    status = os.path.join(OUT, "EAPIG_B4_STATUS.txt")
    open(status, "w").write("LOADING\n")
    t0 = time.time()
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from scipy.stats import spearmanr
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        tok = AutoTokenizer.from_pretrained(MID)
        model = AutoModelForCausalLM.from_pretrained(MID, dtype=torch.bfloat16).to(dev).eval()
        model.requires_grad_(False)
        n_layers = len(model.model.layers)
        nodes = _node_modules(model)
        node_keys = [k for k, _ in nodes]
        comp = [k for k in node_keys if k != "emb"]
        print("loaded", MID, "| layers", n_layers, "| nodes", len(nodes), "| dev", dev)

        ref = json.load(open(REF_JSON))
        ref_node = ref["exact_effect_mean"]
        ref_layer = ref["depth"]["layer_importance_exact"]   
        
        ref_comp = [k for k in comp if k in ref_node]
        print("reference: %s (nodes=%d, layers=%d)" % (os.path.basename(REF_JSON),
              len(ref_comp), len(ref_layer)))

        raw = _build_ioi_multi(tok)
        for p in raw:
            p["clean_ids"] = p["clean_ids"].to(dev)
            p["corrupt_ids"] = p["corrupt_ids"].to(dev)
        ps = [p for p in raw if _solved(model, nodes, p)]
        print("solved IOI pairs: %d" % len(ps))
        assert len(ps) >= MIN_PAIRS, f"only {len(ps)} solved (<{MIN_PAIRS})"
        open(status, "w").write("RUNNING\n")

        rows_den = []
        for j, p in enumerate(ps):
            den, _noi, Lc, Lk, _cc, _kc = _effects_both(model, nodes, p)
            rows_den.append(den)
            if (j + 1) % 10 == 0 or j + 1 == len(ps):
                print("  exact effects %d/%d (%.0fs)" % (j + 1, len(ps), time.time() - t0), flush=True)
        den_mean = {k: float(np.mean([r[k] for r in rows_den])) for k in node_keys}

        layer_imp_ioi = [abs(den_mean["a%d" % i]) + abs(den_mean["m%d" % i]) for i in range(n_layers)]
        n_ref_layers = len(ref_layer)

        ioi_node_vec = np.array([den_mean[k] for k in ref_comp])
        ref_node_vec = np.array([ref_node[k] for k in ref_comp])
        node_rho = float(spearmanr(ioi_node_vec, ref_node_vec).correlation)

        ioi_layer_vec = np.array(layer_imp_ioi[:n_ref_layers])
        ref_layer_vec = np.array(ref_layer[:n_ref_layers])
        layer_rho = float(spearmanr(ioi_layer_vec, ref_layer_vec).correlation)

        ioi_order = list(np.argsort(layer_imp_ioi)[::-1])
        ref_order = list(np.argsort(ref_layer)[::-1])

        out = {
            "config": {"model": MID, "n_layers": n_layers, "n_pairs_ioi": len(ps),
                       "reference": os.path.basename(REF_JSON), "ref_n_layers": n_ref_layers,
                       "ref_n_pairs": ref.get("config", {}).get("n_pairs")},
            "node_spearman_ioi_vs_ref": node_rho,
            "layer_spearman_ioi_vs_ref": layer_rho,
            "n_nodes_compared": len(ref_comp),
            "ioi_exact_effect_mean": den_mean,
            "ioi_layer_importance": layer_imp_ioi,
            "ioi_top_layers": [int(x) for x in ioi_order[:8]],
            "ref_top_layers": [int(x) for x in ref_order[:8]],
            "wall_s": time.time() - t0,
        }
        path = os.path.join(OUT, "eap_ig_b4.json")
        json.dump(out, open(path, "w"), indent=2)
        if os.path.isdir(DRIVE):
            try:
                json.dump(out, open(os.path.join(DRIVE, "eap_ig_b4.json"), "w"), indent=2)
            except Exception as e:
                print("drive save failed:", e)
        open(status, "w").write("DONE %.0fs\n" % (time.time() - t0))
        print("\nDONE %.1fs | %d IOI pairs" % (time.time() - t0, len(ps)))
        print("  node-level  Spearman(IOI, §3.16 ref) = %.3f  (n=%d nodes)" % (node_rho, len(ref_comp)))
        print("  layer-level Spearman(IOI, §3.16 ref) = %.3f  (n=%d layers)" % (layer_rho, n_ref_layers))
        print("  IOI top layers:", out["ioi_top_layers"], "| ref top layers:", out["ref_top_layers"])
    except Exception:
        tb = traceback.format_exc()
        print(tb)
        open(status, "w").write("ERROR\n" + tb)
        raise

if __name__ == "__main__":
    main()
