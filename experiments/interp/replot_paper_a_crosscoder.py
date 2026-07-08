
import json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 13, "axes.titlesize": 12.5, "axes.labelsize": 12,
                     "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 10.5,
                     "axes.grid": True, "grid.alpha": 0.3, "savefig.dpi": 300})
TERN_C, FP_C, SHARED_C = "#c0392b", "#2c3e50", "#7f8c8d"
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "reports", "data")
FIGS = os.path.join(ROOT, "papers", "paper-a-circuits", "figures")
L = lambda n: json.load(open(os.path.join(DATA, n)))

def fig10(cj, path):
    m = cj["metrics"]
    edges = np.array(m["r_hist_edges"]); counts = np.array(m["r_hist_counts"])
    centers = (edges[:-1] + edges[1:]) / 2; w = edges[1] - edges[0]
    fig, ax = plt.subplots(2, 1, figsize=(7.4, 6.6))  
    colors = [TERN_C if c >= 0.8 else FP_C if c <= 0.2 else SHARED_C for c in centers]
    ax[0].bar(centers, counts, width=w * 0.95, color=colors, alpha=0.9)
    ax[0].axvspan(0.0, 0.2, color=FP_C, alpha=0.06); ax[0].axvspan(0.35, 0.65, color=SHARED_C, alpha=0.10)
    ax[0].axvspan(0.8, 1.0, color=TERN_C, alpha=0.06)
    ax[0].set_xlabel(r"relative decoder norm  $r=\|W^{tern}\|/(\|W^{tern}\|+\|W^{fp}\|)$")
    ax[0].set_ylabel("# latents (alive)"); ax[0].set_title("Feature sharing spectrum")
    ax[0].axvline(0.5, color="k", ls=":", lw=1)
    cats = ["FP-only\n(r≤0.2)", "shared\n(0.35–0.65)", "ternary-only\n(r≥0.8)"]
    vals = [m["n_fp_excl"], m["n_shared"], m["n_tern_excl"]]
    ax[1].bar(cats, vals, color=[FP_C, SHARED_C, TERN_C], alpha=0.9)
    for i, v in enumerate(vals):
        ax[1].text(i, v, str(v), ha="center", va="bottom")
    ax[1].set_ylabel("# latents"); ax[1].set_title("Shared vs model-exclusive features")
    ax[1].margins(y=0.12)
    fig.tight_layout(h_pad=2.0); fig.savefig(path); plt.close(fig)

def fig11(cj_diff, cj_ctrl, path):
    md, mc = cj_diff["metrics"], cj_ctrl["metrics"]
    fig, ax = plt.subplots(2, 1, figsize=(7.4, 6.6))  
    for m, c, lab in [(mc, SHARED_C, "FP-vs-FP control"), (md, TERN_C, "ternary-vs-FP")]:
        e = np.array(m["r_hist_edges"]); ctr = (e[:-1] + e[1:]) / 2
        cnt = np.array(m["r_hist_counts"], float); cnt = cnt / cnt.sum()
        ax[0].plot(ctr, cnt, color=c, marker="o", ms=4, label=lab)
    ax[0].axvline(0.5, color="k", ls=":", lw=1)
    ax[0].set_xlabel("relative decoder norm  r"); ax[0].set_ylabel("fraction of latents")
    ax[0].set_title("Norm-sharing: identical (both peak r≈0.5)"); ax[0].legend()
    vals = [mc["cos_shared_median"], md["cos_shared_median"]]
    bars = ax[1].bar(["FP-vs-FP\ncontrol", "ternary-vs-FP"], vals, color=[SHARED_C, TERN_C], alpha=0.9)
    ax[1].axhline(0.0, color="k", lw=1)
    ax[1].axhline(1.0, color="#27ae60", ls="--", lw=1, label="identity (=1)")
    for b, v in zip(bars, vals):
        ax[1].text(b.get_x() + b.get_width() / 2, v + 0.03, "%.3f" % v, ha="center", va="bottom")
    ax[1].set_ylabel("median decoder cosine (shared)"); ax[1].set_ylim(-0.1, 1.05)
    ax[1].set_title("Direction alignment: control ≈ 0, ternary ≈ 0.12"); ax[1].legend()
    fig.tight_layout(h_pad=2.0); fig.savefig(path); plt.close(fig)

if __name__ == "__main__":
    diff = L("crosscoder_diff.json")
    fig10(diff, os.path.join(FIGS, "fig10_crosscoder_diff.png"))
    fig11(diff, L("crosscoder_control.json"), os.path.join(FIGS, "fig11_crosscoder_control.png"))
    md, mc = diff["metrics"], L("crosscoder_control.json")["metrics"]
    print("OK paper-A crosscoder: fig10 (excl fp/sh/tern=%d/%d/%d), fig11 cos control=%.3f ternary=%.3f" % (
        diff["metrics"]["n_fp_excl"], diff["metrics"]["n_shared"], diff["metrics"]["n_tern_excl"],
        mc["cos_shared_median"], md["cos_shared_median"]))
