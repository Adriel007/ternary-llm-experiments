
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
d = json.load(open(os.path.join(ROOT, "reports", "data", "sae_multilayer.json")))
res = sorted(d["results"], key=lambda r: r["layer"])
layers = [r["layer"] for r in res]
fve = [r["metrics"]["fve"] for r in res]
dead = [int(r["metrics"]["dead"]) for r in res]
l0 = [r["metrics"]["l0"] for r in res]
assert all(abs(x - 32) < 1e-6 for x in l0), l0

fig, ax = plt.subplots(figsize=(6.4, 3.6))
xs = np.arange(len(layers))
bars = ax.bar(xs, fve, width=0.55, color="#c0392b")
ax.axhspan(min(fve), max(fve), color="#c0392b", alpha=0.08, zorder=0)
for x, f, dd in zip(xs, fve, dead):
    ax.text(x, f + 0.012, f"FVE {f:.3f}\ndead {dd}", ha="center", va="bottom", fontsize=9)
ax.set_xticks(xs); ax.set_xticklabels([f"layer {l}" for l in layers])
ax.set_ylim(0, 1.0)
ax.set_ylabel("fraction of variance explained")
ax.set_title("First SAEs on a deployed ternary LM (BitNet-2B):\n"
             "FVE 0.77-0.83 across depth   ($L_0=k=32$ exactly, 0 dead; 2 at layer 23)")
ax.axhline(0.0, color="k", lw=0.8)
plt.tight_layout()
for out in [os.path.join(ROOT, "papers", "arxivorg", "paper-a-circuits", "figures", "fig09_sae.png"),
            os.path.join(ROOT, "papers", "paper-a-circuits", "figures", "fig09_sae.png")]:
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", os.path.relpath(out, ROOT))
print("layers", layers, "fve", [round(f, 3) for f in fve], "dead", dead)
