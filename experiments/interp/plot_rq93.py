import json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "results/runpod-runs/RQ93/rq93_capability_vs_robustness.json"))
rows = d["data"]
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
for a, ycol, ylab in ((ax[0], "k2", "K2 GSM8K acc (pure PTQ)"),
                      (ax[1], None, "K2 retention  K2/FP")):
    for arch, c, m in (("dense", "#1f77b4", "o"), ("MoE", "#d62728", "s")):
        xs = [r["fp"] for r in rows if r["arch"] == arch]
        if ycol:
            ys = [r["k2"] for r in rows if r["arch"] == arch]
        else:
            ys = [r["k2"] / r["fp"] for r in rows if r["arch"] == arch]
        a.scatter(xs, ys, c=c, marker=m, s=70, label=arch, edgecolors="k", linewidths=0.5, zorder=3)
    for r in rows:
        y = r["k2"] if ycol else r["k2"] / r["fp"]
        a.annotate(r["model"].replace("-Instruct", "").replace("-Chat", "")[:14],
                   (r["fp"], y), fontsize=6, xytext=(3, 3), textcoords="offset points")
    a.axvspan(0, 0.69, color="gray", alpha=0.08, zorder=0)
    a.set_xlabel("FP GSM8K acc (capability)"); a.set_ylabel(ylab); a.legend(); a.grid(alpha=0.3)
ax[0].set_title("K2 survival vs capability")
ax[1].set_title("Retention: counterexamples refute a clean threshold")
fig.suptitle("#93 — ternary K2 robustness is NOT a clean capability threshold\n"
             "(Falcon3-3B survives where Qwen3-1.7B dies @FP~0.7; Qwen3-8B dies where 30B survives @FP~0.9)")
fig.tight_layout()
out = sys.argv[2] if len(sys.argv) > 2 else "results/runpod-runs/RQ93/rq93_fig.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print("wrote", out)
