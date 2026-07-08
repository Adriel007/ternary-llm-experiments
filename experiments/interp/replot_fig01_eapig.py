
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "reports", "data")
b = json.load(open(os.path.join(DATA, "eap_ig_2b.json")))
c = json.load(open(os.path.join(DATA, "eap_ig_circuits_2b.json")))

NL = b["config"]["n_layers"]
ex = b["exact_effect_mean"]
a_write = np.array([ex.get(f"a{l}", 0.0) for l in range(NL)])
m_write = np.array([ex.get(f"m{l}", 0.0) for l in range(NL)])
cdf = np.array(b["depth"]["cdf"])
median_depth = int(np.searchsorted(cdf, 0.5))          
CRYST = 22                                              
fa = b["faithfulness"]

fig, axes = plt.subplots(3, 1, figsize=(7.2, 9.4))

ax = axes[0]
ax.bar(np.arange(NL), m_write, color="#c0392b", label="MLP write", width=0.8)
ax.bar(np.arange(NL), a_write, color="#2e86de", label="attention write", width=0.5)
ax.axhline(0, color="k", lw=0.8)
ax.axvline(median_depth, color="green", ls="--", lw=1.5, label=f"median |importance| depth {median_depth}")
ax.axvline(CRYST, color="purple", ls=":", lw=1.5, label=f"logit-lens crystallization {CRYST}")
ax.set_xlabel("decoder layer (0..%d)" % (NL - 1)); ax.set_ylabel("causal denoising effect")
ax.set_title("(a) Causal map by depth: early-injection spike + late read-out cluster")
ax.legend(fontsize=8, loc="upper right")

ax = axes[1]
colors = {"antonym": "#c0392b", "ioi": "#2e86de"}
lims = [0, 0]
for fam in ("antonym", "ioi"):
    F = c["families"][fam]
    dn, es = F["denoise_mean"], F["eapig_score_mean"]
    keys = [k for k in dn if k in es]
    x = np.array([dn[k] for k in keys]); y = np.array([es[k] for k in keys])
    rho = F["eapig_spearman_vs_exact"]
    ax.scatter(x, y, s=22, alpha=0.75, color=colors[fam],
               label=f"{fam}  (Spearman {rho:.2f})")
    lims[0] = min(lims[0], x.min(), y.min()); lims[1] = max(lims[1], x.max(), y.max())
pad = 0.05 * (lims[1] - lims[0])
ln = [lims[0] - pad, lims[1] + pad]
ax.plot(ln, ln, "--", color="grey", lw=1, label="identity")
ax.set_xlim(ln); ax.set_ylim(ln)
ax.set_xlabel("exact denoising effect (gold standard)")
ax.set_ylabel("EAP-IG score (gradient approx.)")
ax.set_title("(b) EAP-IG is faithful, task-dependent: Spearman 0.86 (antonym) to 0.70 (IOI)")
ax.text(0.03, 0.97, f"IG-completeness {fa['ig_completeness_mean']:.2f}$\\pm${fa['ig_completeness_std']:.2f}",
        transform=ax.transAxes, va="top", ha="left", fontsize=9,
        bbox=dict(boxstyle="round", fc="white", ec="0.7"))
ax.legend(fontsize=8, loc="lower right")

ax = axes[2]
tk = b["topk_recovery_exact"]
ks = sorted(int(k) for k in tk); ys = [tk[str(k)] for k in ks]
ax.plot(ks, ys, "o-", color="#c0392b")
ax.axhline(1.0, ls="--", color="grey", label="full recovery")
ax.set_xscale("log"); ax.set_ylim(0, 1.05)
ax.set_xlabel("top-K nodes jointly patched (by |exact effect|)")
ax.set_ylabel("recovered fraction of logit-diff")
ax.set_title("(c) The causal circuit is concentrated (K=1 already ~100%)")
ax.legend(fontsize=8, loc="lower right")

plt.tight_layout()
for out in [os.path.join(ROOT, "papers", "arxivorg", "paper-a-circuits", "figures", "fig01_eap_ig.png"),
            os.path.join(ROOT, "papers", "paper-a-circuits", "figures", "fig01_eap_ig.png")]:
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", os.path.relpath(out, ROOT))
print(f"median_depth={median_depth}  antonym rho={c['families']['antonym']['eapig_spearman_vs_exact']:.4f}  "
      f"ioi rho={c['families']['ioi']['eapig_spearman_vs_exact']:.4f}")
