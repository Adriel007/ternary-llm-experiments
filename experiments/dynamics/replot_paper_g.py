
import json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 13, "axes.titlesize": 12.5, "axes.labelsize": 12.5,
                     "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 10.5,
                     "axes.grid": True, "grid.alpha": 0.3, "savefig.dpi": 300})
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "reports", "data")
FIGS = os.path.join(ROOT, "papers", "paper-g-sae-faithfulness", "figures")
L = lambda n: json.load(open(os.path.join(DATA, n)))
TC, GC, OC = "#c0392b", "#2c3e50", "#16a085"

def fig37(t_ml, g_ml, t_lr, g_lr, path):
    tf = [r["metrics"]["fve"] for r in t_ml["results"]]; gf = [r["metrics"]["fve"] for r in g_ml["results"]]
    tl = [r["loss_recovered"] for r in t_lr["results"]]; gl = [r["loss_recovered"] for r in g_lr["results"]]
    x = np.arange(3); w = 0.38
    fig, ax = plt.subplots(2, 1, figsize=(7.2, 8.8))
    for a, tv, gv, ylab, title in [
        (ax[0], tf, gf, "FVE (reconstruction)", "FVE: parity at matched depth/expansion"),
        (ax[1], tl, gl, "loss-recovered (spliced)", "Loss-recovered (Gemma BOS-inflated baseline*)")]:
        a.bar(x - w / 2, tv, w, color=TC, label="ternary BitNet-2B")
        a.bar(x + w / 2, gv, w, color=GC, label="FP-mirror Gemma-2-2B")
        for xi, v in zip(x - w / 2, tv): a.text(xi, v + 0.008, "%.2f" % v, ha="center", va="bottom", fontsize=10)
        for xi, v in zip(x + w / 2, gv): a.text(xi, v + 0.008, "%.2f" % v, ha="center", va="bottom", fontsize=10)
        a.set_xticks(x); a.set_xticklabels(["early\n~0.2", "mid\n~0.5", "late\n~0.8"])
        a.set_ylabel(ylab); a.set_ylim(0, 1.12); a.set_title(title)
        a.legend(loc="lower right")  
    fig.tight_layout(h_pad=2.0); fig.savefig(path); plt.close(fig)

def fig38(tern, g_shim, g_off, path):
    fig, ax = plt.subplots(2, 1, figsize=(7.2, 8.8))
    v = [g_off["mean_absorption_fraction"], g_shim["mean_absorption_fraction"]]
    bars = ax[0].bar(["HookedTransformer\n(official)", "HF shim\n(ours)"], v, color=[OC, "#7f8c8d"], alpha=0.9)
    for b, x0 in zip(bars, v):
        ax[0].text(b.get_x() + b.get_width() / 2, x0 + 0.005, "%.3f" % x0, ha="center", va="bottom")
    ax[0].set_ylabel("mean absorption fraction"); ax[0].set_ylim(0, 0.4)
    ax[0].set_title("Method validation (Gemma SAE): shim ≈ official, Δ<0.01")
    x = np.arange(2); w = 0.38
    tern_v = [tern["mean_absorption_fraction"], tern["mean_full_absorption_rate"]]
    g_v = [g_off["mean_absorption_fraction"], g_off["mean_full_absorption_rate"]]
    b1 = ax[1].bar(x - w / 2, tern_v, w, color=TC, label="ternary BitNet-2B (L15)")
    b2 = ax[1].bar(x + w / 2, g_v, w, color=GC, label="FP-mirror Gemma-2-2B (L13)")
    for bars2, vv in [(b1, tern_v), (b2, g_v)]:
        for b, x0 in zip(bars2, vv):
            ax[1].text(b.get_x() + b.get_width() / 2, x0 + 0.005, "%.3f" % x0, ha="center", va="bottom", fontsize=10)
    ax[1].set_xticks(x); ax[1].set_xticklabels(["mean absorption\nfraction", "full-absorption\nrate"])
    ax[1].set_ylabel("absorption (lower = better)"); ax[1].set_ylim(0, 0.4)
    ax[1].set_title("Mid-depth absorption: ternary vs FP mirror (GT-probe f1≈0.93 both)"); ax[1].legend()
    fig.tight_layout(h_pad=2.0); fig.savefig(path); plt.close(fig)

def fig39(tern, gem, path):
    xs = np.arange(3)
    fig, ax = plt.subplots(figsize=(8.0, 5.4))
    ax.plot(xs, tern, "-o", color=TC, lw=2.4, ms=9, label="ternary BitNet-2B (L6/L15/L23)")
    ax.plot(xs, gem, "-s", color=GC, lw=2.4, ms=8, label="FP-mirror Gemma-2-2B (L5/L13/L20)")
    for x, (t, g) in enumerate(zip(tern, gem)):
        ax.text(x, t + 0.018, "%.3f" % t, ha="center", color=TC, fontsize=11, fontweight="bold")
        ax.text(x, g - 0.032, "%.3f" % g, ha="center", color=GC, fontsize=11, fontweight="bold")
    ax.axvspan(-0.4, 0.5, color="#e74c3c", alpha=0.05); ax.axvspan(1.5, 2.4, color="#e74c3c", alpha=0.08)
    ax.text(1.0, 0.04, "ternary cleaner\n(mid only)", ha="center", fontsize=10, color=TC)
    ax.text(2.0, 0.40, "ternary 4× WORSE", ha="center", fontsize=10, color=TC)
    ax.set_xticks(xs); ax.set_xticklabels(["early\n(L6/L5)", "mid\n(L15/L13)", "late\n(L23/L20)"])
    ax.set_ylabel("mean absorption fraction (lower = better)")
    ax.set_xlim(-0.4, 2.4); ax.set_ylim(0, 0.70)
    ax.set_title("Feature absorption vs depth (Gemma mirror, single seed):\nmid-depth ternary advantage reverses at the last layer")
    ax.legend(loc="upper left")
    fig.tight_layout(h_pad=2.0); fig.savefig(path); plt.close(fig)

if __name__ == "__main__":
    fig37(L("sae_multilayer.json"), L("sae_multilayer_gemma.json"),
          L("sae_loss_recovered.json"), L("sae_loss_recovered_gemma.json"),
          os.path.join(FIGS, "fig37_sae_fpmirror.png"))
    fig38(L("sae_absorption_ternary.json"), L("sae_absorption_gemma_shim.json"),
          L("sae_absorption_gemma_official.json"), os.path.join(FIGS, "fig38_sae_absorption.png"))
    tern = [L("sae_absorption_ternary_L6.json")["mean_absorption_fraction"],
            L("sae_absorption_ternary.json")["mean_absorption_fraction"],
            L("sae_absorption_ternary_L23.json")["mean_absorption_fraction"]]
    gem = [L("sae_absorption_gemma_L5.json")["mean_absorption_fraction"],
           L("sae_absorption_gemma_shim.json")["mean_absorption_fraction"],   
           L("sae_absorption_gemma_L20.json")["mean_absorption_fraction"]]
    fig39(tern, gem, os.path.join(FIGS, "fig39_sae_absorption_depth.png"))
    print("OK paper-G: fig37/38/39 regeneradas. tern-depth=", [round(t, 3) for t in tern], "gem-depth=", [round(g, 3) for g in gem])
