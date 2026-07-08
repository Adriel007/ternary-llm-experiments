
import json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 12, "axes.titlesize": 11.5, "axes.labelsize": 12,
                     "xtick.labelsize": 10.5, "ytick.labelsize": 10.5, "legend.fontsize": 9.5,
                     "axes.grid": True, "grid.alpha": 0.3, "savefig.dpi": 300})
TERN_C, FP_C, SHARED_C = "#c0392b", "#2c3e50", "#7f8c8d"
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
dj = json.load(open(os.path.join(ROOT, "reports", "data", "dfc_scale.json")))
OUT = os.path.join(ROOT, "papers", "paper-a-circuits", "figures", "fig12_dfc.png")

d = dj["dfc"]; part = d["partition"]; sc = dj["std_crosscoder"]["metrics"]
fig, ax = plt.subplots(3, 1, figsize=(7.6, 8.8))  

labels = ["BitNet-2B\n(ternary)", "Llama-3.2-1B\n(FP twin-of-method)"]
base = [d["fve_shared_only_bitnet"], d["fve_shared_only_fp"]]
gain = [d["excl_gain_bitnet"], d["excl_gain_fp"]]
full = [d["fve_full_bitnet"], d["fve_full_fp"]]
xpos = np.arange(2)
ax[0].bar(xpos, base, width=0.55, color=[TERN_C, FP_C], alpha=0.55, label="shared-only FVE")
ax[0].bar(xpos, gain, width=0.55, bottom=base, color=[TERN_C, FP_C], alpha=1.0, hatch="//", edgecolor="white", label="exclusive gain")
for xi, b, g, f in zip(xpos, base, gain, full):
    ax[0].text(xi, b / 2, "%.2f" % b, ha="center", va="center", fontsize=9, weight="bold", color="white")
    ax[0].text(xi, b + g / 2, "+%.2f" % g, ha="center", va="center", fontsize=9, weight="bold", color="black",
               bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.85))
    ax[0].text(xi, f + 0.02, "full %.2f" % f, ha="center", va="bottom", fontsize=9, weight="bold")
ax[0].set_xticks(xpos); ax[0].set_xticklabels(labels, fontsize=9)
ax[0].set_ylabel("held-out FVE")
ax[0].set_ylim(0, 1.22)                       
ax[0].set_title("(a) Held-out FVE — exclusive GAIN: Llama +0.22 $>$ BitNet +0.13")
ax[0].legend(loc="upper left")

keys = [("I_A_bitnet_excl", "BitNet-excl\nI_A", TERN_C), ("I_B_fp_excl", "Llama-excl\nI_B", FP_C), ("I_S_shared", "shared\nI_S", SHARED_C)]
xs = np.arange(3)
nalive = [part[k]["n_alive"] for k, _, _ in keys]; ntot = [part[k]["n"] for k, _, _ in keys]
cols = [c for _, _, c in keys]
ax[1].bar(xs, ntot, width=0.6, color=cols, alpha=0.3, label="slots allocated")
ax[1].bar(xs, nalive, width=0.6, color=cols, alpha=1.0, label="alive (used)")
for xi, a, t in zip(xs, nalive, ntot):
    ax[1].text(xi, a + max(ntot) * 0.01, "%d/%d\n(%.0f%%)" % (a, t, 100 * a / t), ha="center", va="bottom", fontsize=9)
ax[1].set_xticks(xs); ax[1].set_xticklabels([lab for _, lab, _ in keys], fontsize=9)
ax[1].set_ylabel("dictionary slots"); ax[1].set_ylim(0, max(ntot) * 1.15)
ax[1].set_title("(b) Both models use exclusive slots (BitNet 774, Llama 150 alive)")
ax[1].legend(loc="upper right")

e = np.array(sc["r_hist_edges"]); ctr = (e[:-1] + e[1:]) / 2
cnt = np.array(sc["r_hist_counts"], float); cnt = cnt / cnt.sum()
ax[2].bar(ctr, cnt, width=0.045, color=SHARED_C, alpha=0.9)
ax[2].axvline(0.5, color="k", ls=":", lw=1, label="shared (r=0.5)")
ax[2].axvline(sc["r_median"], color=TERN_C, ls="--", lw=1.5, label="median r=%.2f" % sc["r_median"])
ax[2].set_xlabel("relative decoder norm  r  (0=Llama-only, 1=BitNet-only)"); ax[2].set_ylabel("fraction of latents")
ax[2].set_title("(c) Std-crosscoder CONFOUND: %.0f%% BitNet-excl, %.0f%% Llama-excl (artifact)"
                % (100 * sc["frac_bitnet_excl"], 100 * sc["frac_fp_excl"]))
ax[2].legend(loc="upper left")

fig.tight_layout(h_pad=2.0); fig.savefig(OUT); plt.close(fig)
print("OK fig12_dfc: full bitnet %.2f / llama %.2f; std-crosscoder median r=%.2f" % (full[0], full[1], sc["r_median"]))
