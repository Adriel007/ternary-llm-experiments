
import json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 13, "axes.titlesize": 12.5, "axes.labelsize": 12,
                     "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 10.5,
                     "axes.grid": True, "grid.alpha": 0.3, "savefig.dpi": 300})
TERN_C = "#c0392b"
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
lj = json.load(open(os.path.join(ROOT, "reports", "data", "logit_lens_2b.json")))
OUT = os.path.join(ROOT, "papers", "paper-a-circuits", "figures", "fig02_logit_lens.png")

s = lj["summary"]; cfg = lj["config"]; n = cfg["n_layers"]
depths = list(range(len(s["rank_median"])))
rank_med = np.array(s["rank_median"], float)
prob_m = np.array(s["prob_mean"], float); prob_s = np.array(s["prob_std"], float)
cd = s["crystallization_depth_median"]

fig, ax = plt.subplots(2, 1, figsize=(7.2, 8.8))
ax[0].plot(depths, rank_med + 1.0, color=TERN_C, marker="o", ms=5)
ax[0].set_yscale("log")
ax[0].axhline(1.0, color="#7f8c8d", ls=":", lw=1, label="top-1 (rank 0)")
ax[0].axvline(cd, color="#27ae60", ls="--", label="crystallization (median %.0f/%d)" % (cd, n))
ax[0].set_xlabel("depth (0 = embedding, %d = after final layer)" % n)
ax[0].set_ylabel("median rank of final token\n(1+rank, log; lower = closer to top-1)")
ax[0].set_title("(a) The final token climbs to top-1 only late"); ax[0].legend()
ax[1].plot(depths, prob_m, color=TERN_C, marker="o", ms=5)
ax[1].fill_between(depths, np.clip(prob_m - prob_s, 0, 1), np.clip(prob_m + prob_s, 0, 1), color=TERN_C, alpha=0.15)
ax[1].axvline(cd, color="#27ae60", ls="--", label="crystallization (median %.0f/%d)" % (cd, n))
ax[1].set_xlabel("depth (0 = embedding, %d = after final layer)" % n)
ax[1].set_ylabel("mean probability of final token"); ax[1].set_ylim(0, 1)
ax[1].set_title("(b) Probability mass accrues in the last ~third"); ax[1].legend()
fig.tight_layout(); fig.savefig(OUT); plt.close(fig)
print("OK fig02_logit_lens: crystallization median %.0f/%d (~%.0f%% depth), n_prompts=%d" % (cd, n, 100.0 * cd / n, cfg["n_prompts"]))
