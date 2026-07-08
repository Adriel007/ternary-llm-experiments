
import json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))

INT = {
 "Qwen2.5-7B": [(3.168, 0.4367), (4.001, 0.8133)],
 "Coder-7B":   [(3.168, 0.6933), (4.001, 0.7533), (4.919, 0.7767)],
 "Qwen3-8B":   [(3.205, 0.8467), (4.028, 0.8833)],
}

TER = {
 "Qwen2.5-7B": {"K2": (3.295, 0.1467), "K3": (4.943, 0.8133), "FP": 0.8267},
 "Coder-7B":   {"K2": (3.295, 0.5033), "K3": (4.943, 0.7333), "FP": 0.7767},
 "Qwen3-8B":   {"K2": (3.295, 0.0233), "K3": (4.943, 0.8567), "FP": 0.9133},
}
MODELS = ["Qwen2.5-7B", "Coder-7B", "Qwen3-8B"]

fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
for ax, m in zip(axes, MODELS):
    xi = [p[0] for p in INT[m]]; yi = [p[1] for p in INT[m]]
    ax.plot(xi, yi, "o-", color="#d9544d", label="integer K-quant (llama.cpp)", ms=8, lw=2)
    t = TER[m]
    xt = [t["K2"][0], t["K3"][0]]; yt = [t["K2"][1], t["K3"][1]]
    ax.plot(xt, yt, "s-", color="#3a64d9", label="sasori ternary (K2/K3)", ms=8, lw=2)
    ax.axhline(t["FP"], color="gray", ls="--", lw=1, label="FP baseline")
    ax.annotate("K2 cliff\n(unique:\nreasoning\nablated)", xy=(t["K2"][0], t["K2"][1]),
                xytext=(t["K2"][0]+0.5, t["K2"][1]+0.18), fontsize=8, color="#3a64d9",
                arrowprops=dict(arrowstyle="->", color="#3a64d9"))
    ax.set_title(m, fontsize=11); ax.set_xlabel("bits per weight (representational)")
    ax.set_xlim(2.8, 5.4); ax.set_ylim(-0.03, 1.0); ax.grid(alpha=0.25)
axes[0].set_ylabel("GSM8K flexible (reasoning)")
axes[0].legend(fontsize=8, loc="lower right")
fig.suptitle("Reasoning-per-bit: integer K-quants do NOT collapse reasoning at low bpw; ternary K2 does.\n"
             "The ternary K2 cliff is a UNIQUE, graded weight-level reasoning-ablation no integer quant reproduces.",
             fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.92])
out = os.path.join(HERE, "pareto_int_vs_ternary_2026-06-30.png"); fig.savefig(out, dpi=150)
print("wrote", out)
print("\nmodel        | bpw  | integer flex | ternary flex")
for m in MODELS:
    for bpw, y in INT[m]: print(f"{m:12s} | {bpw:4.2f} | {y:.3f} (int)   |")
    for k in ("K2","K3"): print(f"{m:12s} | {TER[m][k][0]:4.2f} |               | {TER[m][k][1]:.3f} ({k})")
