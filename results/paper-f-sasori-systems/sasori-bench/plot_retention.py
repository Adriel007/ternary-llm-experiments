
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, "clean_sweep_2026-06-30.json")))

MODELS = ["Qwen2.5-7B-Instruct", "Qwen3-4B", "Qwen3-8B", "Qwen2.5-Coder-7B-Instruct", "Qwen3-30B-A3B"]
SHORT  = ["Qwen2.5-7B", "Qwen3-4B", "Qwen3-8B", "Coder-7B", "30B-A3B (MoE)"]
TASKS  = [("hellaswag", "acc_norm,none", "HellaSwag (knowledge)"),
          ("arc_challenge", "acc_norm,none", "ARC-Challenge (knowledge)"),
          ("gsm8k", "exact_match,flexible-extract", "GSM8K (reasoning)")]

def val(model, K, task, metric):
    return R[f"{model}|K{K}|{task}"][metric]

fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
x = np.arange(len(MODELS)); w = 0.38
for ax, (task, metric, title) in zip(axes, TASKS):
    k2 = [100 * val(m, 2, task, metric) / val(m, 0, task, metric) for m in MODELS]
    k3 = [100 * val(m, 3, task, metric) / val(m, 0, task, metric) for m in MODELS]
    b2 = ax.bar(x - w/2, k2, w, label="K2 (TQ2P, ~4.1 bpw)", color="#d9544d")
    b3 = ax.bar(x + w/2, k3, w, label="K3 (TQ3P, ~6.2 bpw)", color="#3a7d44")
    ax.axhline(100, color="gray", ls="--", lw=1, zorder=0)
    ax.set_title(title, fontsize=11)
    ax.set_xticks(x); ax.set_xticklabels(SHORT, rotation=30, ha="right", fontsize=8)
    ax.set_ylim(0, 115)
    for b in list(b2) + list(b3):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+1.5, f"{b.get_height():.0f}",
                ha="center", va="bottom", fontsize=7)
axes[0].set_ylabel("capability retained vs FP (K0)  [%]")
axes[0].legend(fontsize=8, loc="lower left")
fig.suptitle("Post-hoc ternary K-plane retention (lm-eval official, faithful joint-ridge, data-free)\n"
             "K3 recovers 93-100% everywhere; K2 hurts reasoning>knowledge, magnitude is architecture-dependent",
             fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.93])
out = os.path.join(HERE, "retention_map_2026-06-30.png")
fig.savefig(out, dpi=150)
print("wrote", out)

print("\nmodel               | task        | K0    | K2-ret | K3-ret")
for m, s in zip(MODELS, SHORT):
    for task, metric, _ in TASKS:
        k0 = val(m, 0, task, metric)
        print(f"{s:19s} | {task:11s} | {k0:.3f} | {100*val(m,2,task,metric)/k0:5.1f}% | {100*val(m,3,task,metric)/k0:5.1f}%")
