
from __future__ import annotations

import math
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  

plt.rcParams.update({"figure.dpi": 130, "font.size": 10, "axes.grid": True, "grid.alpha": 0.3})

TERN_C, FP_C = "#c0392b", "#2c3e50"

def _avg(runs, sect, key):
    xs = runs[0][sect]["step"]
    Y = np.array([r[sect][key] for r in runs], dtype=float)
    return np.array(xs), Y.mean(0), Y.std(0)

def fig_loss(results, path):
    tern, fp = results["runs"]["ternary"], results["runs"]["fp"]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    for runs, c, lab in [(fp, FP_C, "FP twin"), (tern, TERN_C, "Ternary (W1.58A8)")]:
        xs, m, s = _avg(runs, "train", "train_loss")
        ax[0].plot(xs, m, color=c, label=lab); ax[0].fill_between(xs, m - s, m + s, color=c, alpha=0.18)
        xs, m, s = _avg(runs, "eval", "val_loss")
        ax[1].plot(xs, m, color=c, marker="o", ms=3, label=lab); ax[1].fill_between(xs, m - s, m + s, color=c, alpha=0.18)
    ax[0].set_title("Training loss"); ax[1].set_title("Validation loss")
    for a in ax:
        a.set_xlabel("step"); a.set_ylabel("cross-entropy"); a.legend()
    tax = results["summary"]["ternarization_tax"]
    fig.suptitle(f"Ternary vs FP twin (n={tax['n']} seeds)  |  tax = {tax['mean']:.3f} ± {tax['std']:.3f} nats", y=1.02)
    fig.tight_layout(h_pad=1.9); fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    return path

def fig_ternarization(results, path):
    tern = results["runs"]["ternary"]
    xs, mz, sz = _avg(tern, "tern", "frac_zero")
    _, md, sd = _avg(tern, "tern", "dist_to_lattice")
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].plot(xs, mz, color=TERN_C); ax[0].fill_between(xs, mz - sz, mz + sz, color=TERN_C, alpha=0.18)
    ax[0].set_title("Fraction of zeroed weights"); ax[0].set_ylabel("% zeros"); ax[0].set_ylim(0, 1)
    ax[1].plot(xs, md, color=TERN_C); ax[1].fill_between(xs, md - sd, md + sd, color=TERN_C, alpha=0.18)
    ax[1].set_title("Distance of latent weights to ternary lattice"); ax[1].set_ylabel("mean |w·s − round(w·s)|")
    for a in ax:
        a.set_xlabel("step")
    fig.tight_layout(h_pad=1.9); fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    return path

def fig_lambda(results, path):
    s = results["summary"]
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    names = ["FP twin", "Ternary"]
    means = [s["lambda_max_fp"]["mean"], s["lambda_max_ternary"]["mean"]]
    stds = [s["lambda_max_fp"]["std"], s["lambda_max_ternary"]["std"]]
    ax.bar(names, means, yerr=stds, color=[FP_C, TERN_C], capsize=5, alpha=0.9)
    ax.set_ylabel(r"top Hessian eigenvalue $\lambda_{\max}$")
    ax.set_title("Sharpness of the found solution")
    fig.tight_layout(h_pad=1.9); fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    return path

def fig_h3(results, path):
    h = results["h3"]
    layers = h["layers"]; deltas = h["delta_per_layer"]; lambdas = h["lambda_per_layer"]
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.8))
    ax[0].bar(layers, deltas, color=TERN_C, alpha=0.85)
    ax[0].set_title("Per-layer precision sensitivity"); ax[0].set_xlabel("decoder layer"); ax[0].set_ylabel(r"$\Delta L_\ell$ (val-loss reduction)")
    ax[1].bar(layers, lambdas, color="#2980b9", alpha=0.85)
    ax[1].set_title(r"Per-layer curvature $\lambda_{\max}(\ell)$"); ax[1].set_xlabel("decoder layer"); ax[1].set_ylabel(r"$\lambda_{\max}$")
    ax[2].scatter(lambdas, deltas, color=TERN_C)
    for l, xx, yy in zip(layers, lambdas, deltas):
        ax[2].annotate(str(l), (xx, yy), fontsize=8, xytext=(3, 3), textcoords="offset points")
    ax[2].set_title(fr"curvature vs sensitivity (Spearman $\rho$={h['spearman_lambda_delta']:.2f})")
    ax[2].set_xlabel(r"$\lambda_{\max}(\ell)$"); ax[2].set_ylabel(r"$\Delta L_\ell$")
    fig.tight_layout(h_pad=1.9); fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    return path

def fig_pareto(results, path):
    h = results["h3"]
    k = np.arange(len(h["pareto_sens"]))
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    rm = np.array(h["pareto_random_mean"]); rs = np.array(h["pareto_random_std"])
    ax.plot(k, rm, color="#7f8c8d", label="random order")
    ax.fill_between(k, rm - rs, rm + rs, color="#7f8c8d", alpha=0.2)
    ax.plot(k, h["pareto_sens"], color=TERN_C, marker="o", ms=4, label="sensitivity order")
    ax.axhline(h["fp_twin_val_loss"], color=FP_C, ls="--", label="FP twin (lower bound)")
    ax.axhline(h["base_val_loss"], color="#27ae60", ls=":", label="full ternary")
    ax.set_xlabel("# layers kept in full precision"); ax.set_ylabel("validation loss")
    ax.set_title("Mixed-precision Pareto front"); ax.legend(fontsize=8)
    fig.tight_layout(h_pad=1.9); fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    return path

def fig_h3_corrected(h3, path):
    layers = h3["layers"]; ptq = h3["ptq_sensitivity"]; lam = h3["lambda_per_layer"]
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.8))
    ax[0].bar(layers, ptq, color=TERN_C, alpha=0.85)
    ax[0].set_title("PTQ per-layer sensitivity"); ax[0].set_xlabel("decoder layer")
    ax[0].set_ylabel(r"$\Delta L_\ell$ = loss rise when layer $\ell$ ternarized")
    ax[1].bar(layers, lam, color="#2980b9", alpha=0.85)
    ax[1].set_title(r"Per-layer curvature $\lambda_{\max}(\ell)$ (FP solution)")
    ax[1].set_xlabel("decoder layer"); ax[1].set_ylabel(r"$\lambda_{\max}$")
    ax[2].scatter(lam, ptq, color=TERN_C)
    for l, xx, yy in zip(layers, lam, ptq):
        ax[2].annotate(str(l), (xx, yy), fontsize=8, xytext=(3, 3), textcoords="offset points")
    ax[2].set_title(fr"curvature vs sensitivity (Spearman $\rho$={h3['spearman_lambda_ptq']:.2f})")
    ax[2].set_xlabel(r"$\lambda_{\max}(\ell)$"); ax[2].set_ylabel(r"$\Delta L_\ell$")
    fig.tight_layout(h_pad=1.9); fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    return path

def fig_pareto_corrected(h3, path):
    k = np.arange(len(h3["pareto_sens"]))
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    rm = np.array(h3["pareto_random_mean"]); rs = np.array(h3["pareto_random_std"])
    ax.plot(k, rm, color="#7f8c8d", label="random order")
    ax.fill_between(k, rm - rs, rm + rs, color="#7f8c8d", alpha=0.2)
    ax.plot(k, h3["pareto_curv"], color="#2980b9", marker="s", ms=4, label="curvature order")
    ax.plot(k, h3["pareto_sens"], color=TERN_C, marker="o", ms=4, label="sensitivity order")
    ax.axhline(h3["fp_val_loss"], color=FP_C, ls="--", label="all FP")
    ax.axhline(h3["ternary_val_loss"], color="#27ae60", ls=":", label="QAT all-ternary")
    ax.set_xlabel("# layers ternarized (post-hoc, from FP twin)"); ax.set_ylabel("validation loss")
    ax.set_title("Mixed-precision Pareto front (PTQ of the FP twin)"); ax.legend(fontsize=8)
    fig.tight_layout(h_pad=1.9); fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    return path

def fig_mixed(mp, path):
    s0 = mp["seed0"]; par = s0["pareto"]
    ks = [0, 1, 2, 3]
    guided = [s0["full_ternary_k0"]] + [par[str(k)]["guided"] for k in (1, 2, 3)]
    reverse = [s0["full_ternary_k0"]] + [par[str(k)]["reverse"] for k in (1, 2, 3)]
    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    ax.plot(ks, guided, color=TERN_C, marker="o", ms=5, label="guided (most-sensitive layers FP)")
    ax.plot(ks, reverse, color="#8e44ad", marker="s", ms=5, label="adversarial (least-sensitive FP)")
    if "random" in par["1"]:
        ax.scatter([1], [par["1"]["random"]], color="#7f8c8d", marker="D", s=45, zorder=5, label="random (k=1)")
    ax.axhline(s0["full_fp_k8"], color=FP_C, ls="--", label="all FP (lower bound)")
    ax.axhline(s0["full_ternary_k0"], color="#27ae60", ls=":", label="all ternary")
    ax.set_xlabel("# full-precision layers (budget k, of 8)"); ax.set_ylabel("validation loss")
    ax.set_title("Trained mixed-precision allocation (seed 0)"); ax.set_xticks(ks); ax.legend(fontsize=8)
    fig.tight_layout(h_pad=1.9); fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    return path

def fig_headline(mp, path):
    seeds = ["0", "1", "2"]
    g = np.array([mp["headline"][s]["guided_L0"] for s in seeds])
    r = np.array([mp["headline"][s]["reverse_L6"] for s in seeds])
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    x = np.arange(2)
    ax.bar(x, [g.mean(), r.mean()], yerr=[g.std(), r.std()], capsize=6,
           color=[TERN_C, "#8e44ad"], alpha=0.9)
    ax.set_xticks(x); ax.set_xticklabels(["guided\n(FP = layer 0)", "adversarial\n(FP = layer 6)"])
    ax.set_ylabel("validation loss (n=3 seeds)")
    gap = mp["headline_summary"]
    ax.set_title(f"1 FP-layer budget: where you spend it matters\n"
                 f"gap = {gap['gap_mean']:.3f} ± {gap['gap_std']:.3f} nats")
    ax.set_ylim(min(g.min(), r.min()) - 0.02, max(g.max(), r.max()) + 0.02)
    fig.tight_layout(h_pad=1.9); fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    return path

def make_mp(mp, out_dir):
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    return {
        "mixed": fig_mixed(mp, os.path.join(fig_dir, "fig06_mixed_precision.png")),
        "headline": fig_headline(mp, os.path.join(fig_dir, "fig07_mp_headline.png")),
    }

def fig_h2_rankings(h2, path):
    layers = h2["layers"]; s = np.array(h2["s_l_trace"], float); c = np.array(h2["c_l_causal"], float)
    sn = (s - s.min()) / (s.max() - s.min() + 1e-12)
    cn = (c - c.min()) / (c.max() - c.min() + 1e-12)
    x = np.arange(len(layers)); w = 0.4
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.bar(x - w / 2, sn, w, label=r"$s_\ell$ (HAWQ Hessian trace)", color="#2980b9")
    ax.bar(x + w / 2, cn, w, label=r"$c_\ell$ (causal, mean-ablation)", color=TERN_C)
    ax.set_xticks(x); ax.set_xticklabels(layers); ax.set_xlabel("decoder layer")
    ax.set_ylabel("importance (min-max normalized)")
    same = h2.get("orderings_identical", False)
    ax.set_title(f"Second-order vs causal per-layer importance\n(orderings identical: {same})")
    ax.legend(fontsize=8)
    fig.tight_layout(h_pad=1.9); fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    return path

def fig_h2_pareto(h2, path):
    k = h2["budgets"]
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    ax.plot(k, h2["pareto_s"], color="#2980b9", marker="s", ms=5, label=r"$s_\ell$ (HAWQ) allocation")
    ax.plot(k, h2["pareto_c"], color=TERN_C, marker="o", ms=5, label=r"$c_\ell$ (causal) allocation")
    ax.axhline(h2["full_fp"], color=FP_C, ls="--", label="all FP")
    ax.axhline(h2["full_ternary"], color="#27ae60", ls=":", label="all ternary")
    ax.set_xlabel("# full-precision layers (budget k, of 8)"); ax.set_ylabel("validation loss")
    ax.set_title("H2: causal- vs second-order-guided allocation"); ax.set_xticks(k); ax.legend(fontsize=8)
    fig.tight_layout(h_pad=1.9); fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    return path

def fig_flatness(fj, path):
    s = fj["summary"]
    fig, ax = plt.subplots(1, 2, figsize=(9.0, 4.0))
    for a, metric, title in [(ax[0], "trace", r"Hessian trace (avg curvature)"),
                             (ax[1], "lambda_max", r"top eigenvalue $\lambda_{\max}$")]:
        m = [s[metric]["fp"]["mean"], s[metric]["ternary"]["mean"]]
        sd = [s[metric]["fp"]["std"], s[metric]["ternary"]["std"]]
        a.bar(["FP twin", "Ternary"], m, yerr=sd, capsize=5, color=[FP_C, TERN_C], alpha=0.9)
        a.set_title(title)
    rt = s["trace_ratio_tern_over_fp"]; rl = s["lambda_ratio_tern_over_fp"]
    fig.suptitle(f"Flatness: ternary vs FP twin (3 seeds)  |  trace ratio={rt:.2f}, λ ratio={rl:.2f}", y=1.03)
    fig.tight_layout(h_pad=1.9); fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    return path

def make_flatness(fj, out_dir):
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    return {"flatness": fig_flatness(fj, os.path.join(fig_dir, "fig11_flatness.png"))}

def fig_sae(sj, path):
    h = sj["history"]; st = h["step"]
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.plot(st, h["fve"], color=TERN_C, marker="o", ms=4, label="FVE")
    ax.set_xlabel("SAE training step"); ax.set_ylabel("fraction of variance explained", color=TERN_C)
    ax.set_ylim(0, 1); ax.tick_params(axis="y", labelcolor=TERN_C)
    ax2 = ax.twinx()
    ax2.plot(st, h["l0"], color="#2980b9", marker="s", ms=4, ls="--", label="L0")
    ax2.set_ylabel("L0 (mean active latents)", color="#2980b9"); ax2.tick_params(axis="y", labelcolor="#2980b9")
    f = sj["final_metrics"]
    ax.set_title("First SAE on a ternary LM (BitNet-2B, layer %d)\nFVE=%.3f  L0=%.0f  dead=%d  (dict %d, %.1fx)"
                 % (sj["layer"], f["fve"], f["l0"], f["dead"], sj["sae"]["d_sae"], sj["sae"]["expansion"]))
    fig.tight_layout(h_pad=1.9); fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    return path

def make_sae(sj, out_dir):
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    return {"sae": fig_sae(sj, os.path.join(fig_dir, "fig10_sae_ternary.png"))}

def fig_crosscoder(cj, path):
    m = cj["metrics"]; setup = cj["setup"]
    edges = np.array(m["r_hist_edges"]); counts = np.array(m["r_hist_counts"])
    centers = (edges[:-1] + edges[1:]) / 2; w = edges[1] - edges[0]
    SHARED_C = "#7f8c8d"
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.0))
    
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
    cc = setup["crosscoder"]; cosmed = m.get("cos_shared_median")
    fig.suptitle("Ternary-vs-FP crosscoder diff (layer %d, dict %d @ %.0fx, %d alive)  |  "
                 "FVE t=%.2f/fp=%.2f  cos(shared)=%s"
                 % (setup["layer"], cc["d_sae"], cc["expansion"], m["n_alive"],
                    m["final_fve_tern"], m["final_fve_fp"],
                    "%.2f" % cosmed if cosmed is not None else "n/a"), y=1.04)
    fig.tight_layout(h_pad=1.9); fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    return path

def make_crosscoder(cj, out_dir):
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    return {"crosscoder": fig_crosscoder(cj, os.path.join(fig_dir, "fig12_crosscoder_diff.png"))}

def fig_crosscoder_control(cj_diff, cj_ctrl, path):
    SHARED_C = "#7f8c8d"
    md, mc = cj_diff["metrics"], cj_ctrl["metrics"]
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.0))
    
    for m, c, lab in [(mc, SHARED_C, "FP-vs-FP control"), (md, TERN_C, "ternary-vs-FP")]:
        e = np.array(m["r_hist_edges"]); ctr = (e[:-1] + e[1:]) / 2
        cnt = np.array(m["r_hist_counts"], float); cnt = cnt / cnt.sum()
        ax[0].plot(ctr, cnt, color=c, marker="o", ms=3, label=lab)
    ax[0].axvline(0.5, color="k", ls=":", lw=1)
    ax[0].set_xlabel("relative decoder norm  r"); ax[0].set_ylabel("fraction of latents")
    ax[0].set_title("Norm-sharing: identical (both peak r≈0.5)"); ax[0].legend()
    
    vals = [mc["cos_shared_median"], md["cos_shared_median"]]
    bars = ax[1].bar(["FP-vs-FP\ncontrol", "ternary-vs-FP"], vals, color=[SHARED_C, TERN_C], alpha=0.9)
    ax[1].axhline(0.0, color="k", lw=1)
    ax[1].axhline(1.0, color="#27ae60", ls="--", lw=1, label="identity (=1)")
    for b, v in zip(bars, vals):
        ax[1].text(b.get_x() + b.get_width() / 2, v + 0.01, "%.3f" % v, ha="center", va="bottom")
    ax[1].set_ylabel("median decoder cosine (shared)")
    ax[1].set_ylim(-0.1, 1.05)
    ax[1].set_title("Direction alignment: control ≈ 0, ternary ≈ 0.12"); ax[1].legend()
    fig.suptitle("Ternary-vs-FP diff vs FP-vs-FP control (layer 4): same features, "
                 "ternary directions drift well above the run-to-run floor", y=1.03)
    fig.tight_layout(h_pad=1.9); fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    return path

def make_crosscoder_control(cj_diff, cj_ctrl, out_dir):
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    return {"control": fig_crosscoder_control(cj_diff, cj_ctrl,
            os.path.join(fig_dir, "fig13_crosscoder_control.png"))}

def fig_crosscoder_poscontrol(cj_diff, cj_ctrl, cj_pos, path):
    POS_C = "#8e44ad"
    md, mc, mp = cj_diff["metrics"], cj_ctrl["metrics"], cj_pos["metrics"]
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.0))
    
    series = [(mc, "#7f8c8d", "FP-vs-FP, same data (control)"),
              (md, TERN_C, "ternary-vs-FP, same data (diff)"),
              (mp, POS_C, "FP-vs-FP, diff. corpora (pos. control)")]
    for m, c, lab in series:
        e = np.array(m["r_hist_edges"]); ctr = (e[:-1] + e[1:]) / 2
        cnt = np.array(m["r_hist_counts"], float); cnt = cnt / cnt.sum()
        ax[0].plot(ctr, cnt, color=c, marker="o", ms=3, label=lab)
    for xv in (0.2, 0.8):
        ax[0].axvspan(0 if xv < 0.5 else 0.8, 0.2 if xv < 0.5 else 1.0, color=POS_C, alpha=0.06)
    ax[0].axvline(0.5, color="k", ls=":", lw=1)
    ax[0].set_yscale("log")
    ax[0].set_xlabel("relative decoder norm  r  (shaded = exclusive tails)")
    ax[0].set_ylabel("fraction of latents (log)")
    ax[0].set_title("Spectrum: only different-corpora populates the tails"); ax[0].legend(fontsize=7)
    
    labs = ["control\n(same data)", "diff\n(same data)", "pos. control\n(diff corpora)"]
    excl = [mc["n_tern_excl"] + mc["n_fp_excl"], md["n_tern_excl"] + md["n_fp_excl"],
            mp["n_tern_excl"] + mp["n_fp_excl"]]
    bars = ax[1].bar(labs, excl, color=["#7f8c8d", TERN_C, POS_C], alpha=0.9)
    for b, v in zip(bars, excl):
        ax[1].text(b.get_x() + b.get_width() / 2, v + 4, str(v), ha="center", va="bottom")
    ax[1].set_ylabel("model-exclusive latents (r≥0.8 or ≤0.2)")
    ax[1].set_title("Metric is alive: 0 / 0 / %d" % excl[2])
    fig.suptitle("A1 positive control: the relative-norm metric detects genuine inventory "
                 "differences (344) — so ternary-vs-FP's 0 is meaningful", y=1.03)
    fig.tight_layout(h_pad=1.9); fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    return path

def make_crosscoder_poscontrol(cj_diff, cj_ctrl, cj_pos, out_dir):
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    return {"poscontrol": fig_crosscoder_poscontrol(cj_diff, cj_ctrl, cj_pos,
            os.path.join(fig_dir, "fig36_crosscoder_poscontrol.png"))}

def fig_sae_fpmirror(t_ml, g_ml, t_lr, g_lr, path):
    TC, GC = "#c0392b", "#2c3e50"
    nT, nG = 30, 26
    tf = [(r["layer"] / (nT - 1), r["metrics"]["fve"]) for r in t_ml["results"]]
    gf = [(r["layer"] / (nG - 1), r["metrics"]["fve"]) for r in g_ml["results"]]
    tl = [(r["layer"] / (nT - 1), r["loss_recovered"]) for r in t_lr["results"]]
    gl = [(r["layer"] / (nG - 1), r["loss_recovered"]) for r in g_lr["results"]]
    import numpy as _np
    x = _np.arange(3); w = 0.38
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.0))
    ax[0].bar(x - w / 2, [v for _, v in tf], w, color=TC, label="ternary BitNet-2B")
    ax[0].bar(x + w / 2, [v for _, v in gf], w, color=GC, label="FP-mirror Gemma-2-2B")
    for xi, (_, v) in zip(x - w / 2, tf):
        ax[0].text(xi, v + 0.008, "%.2f" % v, ha="center", va="bottom", fontsize=8)
    for xi, (_, v) in zip(x + w / 2, gf):
        ax[0].text(xi, v + 0.008, "%.2f" % v, ha="center", va="bottom", fontsize=8)
    ax[0].set_xticks(x); ax[0].set_xticklabels(["early\n~0.2", "mid\n~0.5", "late\n~0.8"])
    ax[0].set_ylabel("FVE (reconstruction)"); ax[0].set_ylim(0, 1.0)
    ax[0].set_title("FVE: parity at matched depth/expansion"); ax[0].legend(fontsize=8)
    ax[1].bar(x - w / 2, [v for _, v in tl], w, color=TC, label="ternary BitNet-2B")
    ax[1].bar(x + w / 2, [v for _, v in gl], w, color=GC, label="FP-mirror Gemma-2-2B")
    for xi, (_, v) in zip(x - w / 2, tl):
        ax[1].text(xi, v + 0.008, "%.2f" % v, ha="center", va="bottom", fontsize=8)
    for xi, (_, v) in zip(x + w / 2, gl):
        ax[1].text(xi, v + 0.008, "%.2f" % v, ha="center", va="bottom", fontsize=8)
    ax[1].set_xticks(x); ax[1].set_xticklabels(["early\n~0.2", "mid\n~0.5", "late\n~0.8"])
    ax[1].set_ylabel("loss-recovered (spliced)"); ax[1].set_ylim(0, 1.0)
    ax[1].set_title("Loss-recovered (Gemma BOS-inflated baseline*)"); ax[1].legend(fontsize=8)
    fig.suptitle("SAE faithfulness at matched depth: ternary BitNet-2B vs FP-mirror "
                 "Gemma-2-2B (matched expansion 6.4x, k=32)", y=1.03)
    fig.tight_layout(h_pad=1.9); fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    return path

def make_sae_fpmirror(t_ml, g_ml, t_lr, g_lr, out_dir):
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    return {"fpmirror": fig_sae_fpmirror(t_ml, g_ml, t_lr, g_lr,
            os.path.join(fig_dir, "fig37_sae_fpmirror.png"))}

def fig_sae_absorption(tern, g_shim, g_off, path):
    TC, GC, OC = "#c0392b", "#2c3e50", "#16a085"
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.0))
    
    v = [g_off["mean_absorption_fraction"], g_shim["mean_absorption_fraction"]]
    bars = ax[0].bar(["HookedTransformer\n(official)", "HF shim\n(ours)"], v, color=[OC, "#7f8c8d"], alpha=0.9)
    for b, x in zip(bars, v):
        ax[0].text(b.get_x() + b.get_width() / 2, x + 0.005, "%.3f" % x, ha="center", va="bottom", fontsize=9)
    ax[0].set_ylabel("mean absorption fraction"); ax[0].set_ylim(0, 0.4)
    ax[0].set_title("Method validation (Gemma SAE):\nshim ≈ official, Δ<0.01", fontsize=10)
    
    import numpy as _np
    x = _np.arange(2); w = 0.38
    tern_v = [tern["mean_absorption_fraction"], tern["mean_full_absorption_rate"]]
    g_v = [g_off["mean_absorption_fraction"], g_off["mean_full_absorption_rate"]]
    b1 = ax[1].bar(x - w / 2, tern_v, w, color=TC, label="ternary BitNet-2B (L15)")
    b2 = ax[1].bar(x + w / 2, g_v, w, color=GC, label="FP-mirror Gemma-2-2B (L13)")
    for bars2, vv in [(b1, tern_v), (b2, g_v)]:
        for b, x0 in zip(bars2, vv):
            ax[1].text(b.get_x() + b.get_width() / 2, x0 + 0.005, "%.3f" % x0, ha="center", va="bottom", fontsize=8)
    ax[1].set_xticks(x); ax[1].set_xticklabels(["mean absorption\nfraction", "full-absorption\nrate"])
    ax[1].set_ylabel("absorption (lower = better)"); ax[1].set_ylim(0, 0.4)
    ax[1].set_title("Mid-depth absorption: ternary vs FP mirror\n(GT-probe f1≈0.93 both)", fontsize=10); ax[1].legend(fontsize=8)
    fig.suptitle("Feature absorption at mid-depth (SAEBench first-letter metric; "
                 "shim validated against official SAEBench)", y=1.03)
    fig.tight_layout(h_pad=1.9); fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    return path

def make_sae_absorption(tern, g_shim, g_off, out_dir):
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    return {"absorption": fig_sae_absorption(tern, g_shim, g_off,
            os.path.join(fig_dir, "fig38_sae_absorption.png"))}

def fig_sae_absorption_depth(tern, gem, path):
    import numpy as _np
    TC, GC = "#c0392b", "#2c3e50"
    xs = _np.arange(3)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(xs, tern, "-o", color=TC, lw=2.2, ms=8, label="ternary BitNet-2B (L6/L15/L23)")
    ax.plot(xs, gem, "-s", color=GC, lw=2.2, ms=7, label="FP-mirror Gemma-2-2B (L5/L13/L20)")
    for x, (t, g) in enumerate(zip(tern, gem)):
        ax.text(x, t + 0.018, "%.3f" % t, ha="center", color=TC, fontsize=9, fontweight="bold")
        ax.text(x, g - 0.030, "%.3f" % g, ha="center", color=GC, fontsize=9, fontweight="bold")
    
    ax.axvspan(-0.4, 0.5, color="#e74c3c", alpha=0.05)
    ax.axvspan(1.5, 2.4, color="#e74c3c", alpha=0.08)
    ax.text(1.0, 0.04, "ternary cleaner\n(mid only)", ha="center", fontsize=8, color=TC)
    ax.text(2.0, 0.40, "ternary 4× WORSE", ha="center", fontsize=8, color=TC)
    ax.set_xticks(xs); ax.set_xticklabels(["early\n(L6/L5)", "mid\n(L15/L13)", "late\n(L23/L20)"])
    ax.set_ylabel("mean absorption fraction (lower = better)")
    ax.set_xlim(-0.4, 2.4); ax.set_ylim(0, 0.70)
    ax.set_title("Feature absorption vs depth (Gemma mirror, single seed):\n"
                 "the mid-depth ternary advantage reverses at the last layer", fontsize=10)
    ax.legend(fontsize=8, loc="upper left"); ax.grid(alpha=0.25)
    fig.tight_layout(h_pad=1.9); fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    return path

def make_sae_absorption_depth(tern, gem, out_dir):
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    return {"absorption_depth": fig_sae_absorption_depth(tern, gem,
            os.path.join(fig_dir, "fig39_sae_absorption_depth.png"))}

def fig_soft_quant(sj, path):
    SOFT_C = "#e67e22"
    s = sj["summary"]; p = s["paired"]
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 4.0))
    for a, metric, title, fmt in [
        (ax[0], "tax", "Ternarization tax (nats) — lower better", "%.3f"),
        (ax[1], "trace", "Hessian trace (avg curvature) — lower flatter", "%.0f")]:
        m = [s["ste"][metric]["mean"], s["soft"][metric]["mean"]]
        sd = [s["ste"][metric]["std"], s["soft"][metric]["std"]]
        bars = a.bar(["STE\n(baseline)", "soft-annealed\n(H4)"], m, yerr=sd, capsize=5,
                     color=[TERN_C, SOFT_C], alpha=0.9)
        for b, v in zip(bars, m):
            a.text(b.get_x() + b.get_width() / 2, v, fmt % v, ha="center", va="bottom")
        a.set_title(title)
    fig.suptitle("Soft differentiable quantization (QP4/H4, 3 seeds): "
                 "Δtax=%+.3f±%.3f, trace ×%.2f±%.2f — soft is sharper AND costlier (refuted)"
                 % (p["dtax_mean"], p["dtax_std"], p["trace_ratio_mean"], p["trace_ratio_std"]), y=1.03)
    fig.tight_layout(h_pad=1.9); fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    return path

def make_soft_quant(sj, out_dir):
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    return {"soft_quant": fig_soft_quant(sj, os.path.join(fig_dir, "fig14_soft_quant_h4.png"))}

def fig_sae_multilayer(mj, path):
    rs = mj["results"]
    layers = [r["layer"] for r in rs]
    fve = [r["metrics"]["fve"] for r in rs]
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    bars = ax.bar([str(l) for l in layers], fve, color=TERN_C, alpha=0.9)
    for b, r in zip(bars, rs):
        ax.text(b.get_x() + b.get_width() / 2, r["metrics"]["fve"],
                "FVE %.3f\nL0=%.0f, %d dead" % (r["metrics"]["fve"], r["metrics"]["l0"], r["metrics"]["dead"]),
                ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0, 1.0); ax.set_xlabel("BitNet-2B residual-stream layer (of 30)")
    ax.set_ylabel("fraction of variance explained")
    ax.set_title("Multi-layer SAEs on a ternary LM (dict %d, k=%d): FVE %.2f–%.2f, 0 dead across depth"
                 % (rs[0]["sae"]["d_sae"] if "sae" in rs[0] else 16384,
                    rs[0]["metrics"]["k"], min(fve), max(fve)))
    fig.tight_layout(h_pad=1.9); fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    return path

def make_sae_multilayer(mj, out_dir):
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    return {"sae_multilayer": fig_sae_multilayer(mj, os.path.join(fig_dir, "fig15_sae_multilayer.png"))}

def fig_t_star(tj, path):
    s = tj["summary"]
    fr = sorted(float(k) for k in s)
    mean = [s[str(f) if str(f) in s else ("%.1f" % f)]["tax_mean"] for f in fr]
    std = [s[str(f) if str(f) in s else ("%.1f" % f)]["tax_std"] for f in fr]
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.0))
    
    ax[0].errorbar(fr, mean, yerr=std, color=TERN_C, marker="o", ms=6, capsize=4)
    ax[0].set_yscale("log"); ax[0].set_xlabel("t*  (fraction of training in FP before the ternary switch)")
    ax[0].set_ylabel("ternarization tax (nats, log)"); ax[0].set_title("Full sweep — t*=1.0 (PTQ) is catastrophic")
    ax[0].annotate("PTQ\n(no QAT)", (1.0, mean[-1]), textcoords="offset points", xytext=(-10, -28), ha="center")
    
    z = [(f, m, e) for f, m, e in zip(fr, mean, std) if f <= 0.75]
    zf, zm, ze = [a[0] for a in z], [a[1] for a in z], [a[2] for a in z]
    ax[1].errorbar(zf, zm, yerr=ze, color=TERN_C, marker="o", ms=6, capsize=4)
    bi = int(np.argmin(zm))
    ax[1].scatter([zf[bi]], [zm[bi]], color="#27ae60", zorder=5, s=90, label="optimum t*=%.2f" % zf[bi])
    ax[1].set_xlabel("t*"); ax[1].set_ylabel("ternarization tax (nats)")
    ax[1].set_title("Zoom (t*≤0.75): U-shape, optimum at t*≈0.5"); ax[1].legend()
    fig.suptitle("QP3/H3 — optimal 16→1.58 transition step (3 seeds): tax halved at t*≈0.5 "
                 "(%.3f vs %.3f pure QAT)" % (s["0.5"]["tax_mean"], s["0.0"]["tax_mean"]), y=1.03)
    fig.tight_layout(h_pad=1.9); fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    return path

def make_t_star(tj, out_dir):
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    return {"t_star": fig_t_star(tj, os.path.join(fig_dir, "fig16_t_star.png"))}

def fig_crosscoder_depth(dj, path):
    rs = dj["results"]
    layers = [r["layer"] for r in rs]
    cos = [r["metrics"]["cos_shared_median"] for r in rs]
    fve = [r["metrics"]["final_fve_tern"] for r in rs]
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.0))
    ax[0].plot(layers, cos, color=TERN_C, marker="o", ms=7)
    ax[0].axhline(0.0, color="#7f8c8d", ls="--", lw=1, label="orthogonal floor (FP-vs-FP control ≈ 0)")
    ax[0].set_xlabel("residual-stream layer (of 8)"); ax[0].set_ylabel("decoder cosine of shared features")
    ax[0].set_title("Drift across depth: early layers drift most\n(cosine ↑ with depth ⇒ less drift deeper)")
    ax[0].set_ylim(-0.02, max(cos) * 1.2); ax[0].legend(fontsize=8)
    ax[1].plot(layers, fve, color="#2980b9", marker="s", ms=7)
    ax[1].set_xlabel("residual-stream layer (of 8)"); ax[1].set_ylabel("SAE FVE (ternary side)")
    ax[1].set_title("Reconstruction across depth (0 exclusive features at every layer)")
    ax[1].set_ylim(0.8, 1.0)
    fig.suptitle("Ternary-vs-FP crosscoder diff across depth (tiny twins): drift is largest early, "
                 "shrinks 4× toward the output (cos %.2f→%.2f)" % (cos[0], cos[-1]), y=1.03)
    fig.tight_layout(h_pad=1.9); fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    return path

def make_crosscoder_depth(dj, out_dir):
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    return {"depth": fig_crosscoder_depth(dj, os.path.join(fig_dir, "fig17_crosscoder_depth.png"))}

def fig_t_star_emergence(ej, path):
    pf = ej["summary"]["per_frac"]; cp = ej["summary"]["coupling"]
    fr = sorted(float(k) for k in pf)
    K = lambda f: "%.3f" % f
    tax = [pf[K(f)]["final_tax_mean"] for f in fr]
    txs = [pf[K(f)]["final_tax_std"] for f in fr]
    lfp = [pf[K(f)]["L_fp_mean"] for f in fr]
    rptq = [pf[K(f)]["R_ptq_mean"] for f in fr]
    dfp = [pf[K(f)]["D_fp_mean"] for f in fr]
    zfp = [pf[K(f)]["z_fp_mean"] for f in fr]
    PURPLE, BLUE = "#8e44ad", "#2980b9"
    fig, ax = plt.subplots(2, 2, figsize=(11.0, 8.2))

    a = ax[0, 0]
    a.errorbar(fr, tax, yerr=txs, color=TERN_C, marker="o", ms=6, capsize=4)
    bi = int(np.argmin(tax))
    a.scatter([fr[bi]], [tax[bi]], color="#27ae60", zorder=5, s=95,
              label="optimum t*=%.2f" % fr[bi])
    a.set_xlabel("t*  (FP fraction before the ternary switch)")
    a.set_ylabel("ternarization tax (nats)")
    a.set_title("(a) The U-shape to explain (13-pt grid, 3 seeds)"); a.legend(fontsize=8)

    a = ax[0, 1]
    a.plot(fr, lfp, color=FP_C, marker="o", ms=5, label=r"$L_{fp}$ FP basin (readiness)")
    a.set_yscale("log"); a.set_xlabel("t*"); a.set_ylabel(r"$L_{fp}$ (nats, log)", color=FP_C)
    a.tick_params(axis="y", labelcolor=FP_C)
    a2 = a.twinx()
    a2.plot(fr, rptq, color=TERN_C, marker="s", ms=5, ls="--", label=r"$R_{ptq}$ instant PTQ gap")
    a2.set_ylabel(r"$R_{ptq}$ (nats)", color=TERN_C); a2.tick_params(axis="y", labelcolor=TERN_C)
    a2.grid(False)
    a.set_title("(b) Readiness × budget: basin improves, PTQ gap grows")
    h1, l1 = a.get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels()
    a.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper center")

    a = ax[1, 0]
    a.plot(fr, dfp, color=PURPLE, marker="o", ms=5, label=r"$D_{fp}$ dist-to-lattice")
    a.set_xlabel("t*"); a.set_ylabel(r"$D_{fp}$  mean$|w/\beta-\mathrm{round}|$", color=PURPLE)
    a.tick_params(axis="y", labelcolor=PURPLE)
    dr, zr = cp["D_fp_range"], cp["z_fp_range"]
    a.set_ylim(dr[0] - 0.02, dr[1] + 0.02)
    a2 = a.twinx()
    a2.plot(fr, zfp, color=BLUE, marker="s", ms=5, ls="--", label=r"$z_{fp}$ zero-fraction")
    a2.set_ylabel(r"$z_{fp}$ zero-fraction", color=BLUE); a2.tick_params(axis="y", labelcolor=BLUE)
    a2.set_ylim(zr[0] - 0.02, zr[1] + 0.02); a2.grid(False)
    a.set_title("(c) Modes do NOT emerge: D_fp, z_fp flat across t\n"
                r"($\Delta D{\approx}%.0e$, $\Delta z{\approx}%.0e$ over all 39 measurements)"
                % (dr[1] - dr[0], zr[1] - zr[0]))
    h1, l1 = a.get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels()
    a.legend(h1 + h2, l1 + l2, fontsize=7, loc="center right")

    a = ax[1, 1]
    hb = np.array(ej["config"]["hist_bins"]); ctr = (hb[:-1] + hb[1:]) / 2
    hist = ej["hist_seed0_by_step"]
    cols = ["#bdc3c7", "#e67e22", "#1a5276"]
    for st, c in zip(["0", "1000", "1900"], cols):
        if st in hist:
            h = np.array(hist[st], float); h = h / h.sum()
            a.plot(ctr, h, color=c, lw=1.7, label="FP step %s" % st)
    for x in (-1.0, 0.0, 1.0):
        a.axvline(x, color="k", ls=":", lw=0.8, alpha=0.5)
    a.set_xlabel(r"$w/\beta$  (latent weight / absmean scale)")
    a.set_ylabel("fraction of weights"); a.set_xlim(-3, 3)
    a.set_title("(d) FP weight histogram keeps its shape\n(no trimodal {-1,0,+1} emergence)")
    a.legend(fontsize=7)

    fig.suptitle("QP3/H3 part 2 — t* is a readiness×budget optimum, NOT a marker of weight-mode "
                 "emergence (3 seeds)\n"
                 r"Spearman(tax,$R_{ptq}$)=%.2f, Spearman(tax,$L_{fp}$)=%.2f — no single monotone "
                 "signal tracks the U" % (cp["spearman_tax_vs_Rptq"], cp["spearman_tax_vs_Lfp"]),
                 y=1.02)
    fig.tight_layout(h_pad=1.9); fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    return path

def make_t_star_emergence(ej, out_dir):
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    return {"t_star_emergence": fig_t_star_emergence(
        ej, os.path.join(fig_dir, "fig18_t_star_emergence.png"))}

def fig_h3_corr_multiseed(cj, path):
    s = cj["summary"]; rows = cj["rows"]
    rhos = s["rho_per_seed"]; seeds = [r["seed"] for r in rows]
    ptq_m, ptq_e = np.array(s["ptq_mean"]), np.array(s["ptq_std"])
    lam_m, lam_e = np.array(s["lambda_mean"]), np.array(s["lambda_std"])
    layers = list(range(len(ptq_m)))
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.0))
    
    bars = ax[0].bar([str(x) for x in seeds], rhos, color=TERN_C, alpha=0.9)
    for b, v in zip(bars, rhos):
        ax[0].text(b.get_x() + b.get_width() / 2, v, "%.2f" % v, ha="center", va="bottom")
    ax[0].axhline(s["rho_mean"], color=FP_C, ls="--",
                  label="mean %.2f ± %.2f" % (s["rho_mean"], s["rho_std"]))
    ax[0].set_ylim(0, 1.08); ax[0].set_xlabel("seed")
    ax[0].set_ylabel(r"Spearman $\rho(\lambda_{\max}(\ell),\,\Delta L_\ell)$")
    ax[0].set_title("(a) Curvature↔sensitivity ρ per seed\n(positive on average, seed-variable)")
    ax[0].legend(fontsize=8)
    
    ax[1].errorbar(lam_m, ptq_m, xerr=lam_e, yerr=ptq_e, fmt="o", color=TERN_C, capsize=3, ms=5)
    for l, xx, yy in zip(layers, lam_m, ptq_m):
        ax[1].annotate("L%d" % l, (xx, yy), fontsize=8, xytext=(4, 3), textcoords="offset points")
    ax[1].set_xlabel(r"per-layer curvature $\lambda_{\max}(\ell)$ (mean±std, 3 seeds)")
    ax[1].set_ylabel(r"PTQ sensitivity $\Delta L_\ell$ (mean±std)")
    ax[1].set_title("(b) Layer 0 dominates both (all 3 seeds)\npooled ρ=%.2f" % s["rho_pooled"])
    fig.suptitle("H3 curvature↔sensitivity replicated across 3 seeds: ρ=%.2f±%.2f — "
                 "positive but seed-variable; layer-0 dominance is the robust invariant" %
                 (s["rho_mean"], s["rho_std"]), y=1.03)
    fig.tight_layout(h_pad=1.9); fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    return path

def make_h3_corr_multiseed(cj, out_dir):
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    return {"h3_corr": fig_h3_corr_multiseed(
        cj, os.path.join(fig_dir, "fig19_h3_corr_multiseed.png"))}

def fig_logit_lens(lj, path):
    s = lj["summary"]; cfg = lj["config"]
    n = cfg["n_layers"]
    depths = list(range(len(s["rank_median"])))   
    rank_med = np.array(s["rank_median"], float)
    prob_m = np.array(s["prob_mean"], float); prob_s = np.array(s["prob_std"], float)
    cd = s["crystallization_depth_median"]
    fig, ax = plt.subplots(1, 2, figsize=(11.0, 4.2))
    
    ax[0].plot(depths, rank_med + 1.0, color=TERN_C, marker="o", ms=4)
    ax[0].set_yscale("log")
    ax[0].axhline(1.0, color="#7f8c8d", ls=":", lw=1, label="top-1 (rank 0)")
    ax[0].axvline(cd, color="#27ae60", ls="--", label="crystallization (median %.0f/%d)" % (cd, n))
    ax[0].set_xlabel("depth (0 = embedding, %d = after final layer)" % n)
    ax[0].set_ylabel("median rank of final token (1+rank, log; lower = closer to top-1)")
    ax[0].set_title("(a) The final token climbs to top-1 only late"); ax[0].legend(fontsize=7)
    
    ax[1].plot(depths, prob_m, color=TERN_C, marker="o", ms=4)
    ax[1].fill_between(depths, np.clip(prob_m - prob_s, 0, 1), np.clip(prob_m + prob_s, 0, 1),
                       color=TERN_C, alpha=0.15)
    ax[1].axvline(cd, color="#27ae60", ls="--", label="crystallization (median %.0f/%d)" % (cd, n))
    ax[1].set_xlabel("depth (0 = embedding, %d = after final layer)" % n)
    ax[1].set_ylabel("mean probability of final token")
    ax[1].set_ylim(0, 1); ax[1].set_title("(b) Probability mass accrues in the last ~third")
    ax[1].legend(fontsize=7)
    fig.suptitle("First logit-lens of a ternary LM (BitNet-2B, %d prompts): the model commits to its "
                 "next token LATE\nmedian crystallization at depth %.0f/%d (~%.0f%% depth); prob ~0→0.8 "
                 "over the last ~%d layers" % (cfg["n_prompts"], cd, n, 100.0 * cd / n, n - int(cd)),
                 y=1.04)
    fig.tight_layout(h_pad=1.9); fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    return path

def make_logit_lens(lj, out_dir):
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    return {"logit_lens": fig_logit_lens(lj, os.path.join(fig_dir, "fig20_logit_lens_2b.png"))}

def fig_eap_ig(ej, path, ll_crys=22):
    ee = ej["exact_effect_mean"]; ei = ej["eapig_score_mean"]; fa = ej["faithfulness"]
    n = ej["config"]["n_layers"]
    ATTN_C = "#2980b9"
    a_eff = [ee["a%d" % i] for i in range(n)]
    m_eff = [ee["m%d" % i] for i in range(n)]
    x = np.arange(n)
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.3))

    ax[0].bar(x - 0.2, m_eff, width=0.4, color=TERN_C, label="MLP write")
    ax[0].bar(x + 0.2, a_eff, width=0.4, color=ATTN_C, label="attention write")
    ax[0].axhline(0, color="k", lw=0.7)
    mid = ej["depth"]["median_importance_depth"]
    ax[0].axvline(mid, color="#27ae60", ls="--", lw=1.3, label="median |importance| depth %d" % mid)
    ax[0].axvline(ll_crys, color="#8e44ad", ls=":", lw=1.3, label="logit-lens crystallization %d" % ll_crys)
    ax[0].set_xlabel("decoder layer (0..%d)" % (n - 1))
    ax[0].set_ylabel("causal denoising effect")
    ax[0].set_title("(a) Causal map by depth: early-injection spike + late read-out cluster")
    ax[0].legend(fontsize=7)

    comp = ["a%d" % i for i in range(n)] + ["m%d" % i for i in range(n)]
    ex = np.array([ee[k] for k in comp]); ey = np.array([ei[k] for k in comp])
    lo, hi = float(min(ex.min(), ey.min())), float(max(ex.max(), ey.max()))
    pad = 0.05 * (hi - lo)
    ax[1].plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="#7f8c8d", ls="--", lw=1, label="identity")
    ax[1].scatter(ex, ey, color=TERN_C, alpha=0.8, s=28)
    ax[1].set_xlabel("exact denoising effect (gold standard)")
    ax[1].set_ylabel("EAP-IG score (gradient approximation)")
    ax[1].set_title("(b) EAP-IG is faithful on the ternary 2B\n"
                    r"Spearman %.2f, Pearson %.2f, IG-completeness %.2f" %
                    (fa["spearman_eapig_vs_exact"], fa["pearson_eapig_vs_exact"], fa["ig_completeness_mean"]))
    ax[1].legend(fontsize=8)

    tk = ej["topk_recovery_exact"]
    Ks = sorted(int(k) for k in tk); rec = [tk[str(k)] for k in Ks]
    ax[2].plot(Ks, rec, color=TERN_C, marker="o", ms=6)
    ax[2].axhline(1.0, color="#7f8c8d", ls="--", lw=1, label="full recovery")
    ax[2].set_xscale("log")
    ax[2].set_xlabel("top-K nodes jointly patched (by |exact effect|)")
    ax[2].set_ylabel("recovered fraction of logit-diff")
    ax[2].set_ylim(0, 1.08)
    ax[2].set_title("(c) The causal circuit is highly concentrated\n(K=1 already ~%.0f%%)" % (100 * rec[0]))
    ax[2].legend(fontsize=8)

    fig.suptitle("First causal component attribution of a ternary LM (BitNet-2B, %d contrastive pairs): "
                 "EAP-IG faithfully (ρ=%.2f) recovers the exact causal map; information enters early "
                 "(layer-0 MLP) and is read out late (~L%d), echoing the logit-lens" %
                 (ej["config"]["n_pairs"], fa["spearman_eapig_vs_exact"], ll_crys), y=1.04)
    fig.tight_layout(h_pad=1.9); fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    return path

def make_eap_ig(ej, out_dir):
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    return {"eap_ig": fig_eap_ig(ej, os.path.join(fig_dir, "fig21_eap_ig_2b.png"))}

def fig_eap_circuits(ej, path):
    ATTN_C = "#2980b9"
    fams = ej["families"]
    n = ej["config"]["n_layers"]
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.3))

    for col, (fam, title, mech) in enumerate([
            ("ioi", "(a) IOI: late ATTENTION read-out (name-movers ~L19-24)", "attention"),
            ("antonym", "(b) Antonyms: late MLP read-out (contrast)", "MLP")]):
        ee = fams[fam]["denoise_mean"]
        a_eff = [ee["a%d" % i] for i in range(n)]
        m_eff = [ee["m%d" % i] for i in range(n)]
        x = np.arange(n)
        ax[col].bar(x - 0.2, m_eff, width=0.4, color=TERN_C, label="MLP write")
        ax[col].bar(x + 0.2, a_eff, width=0.4, color=ATTN_C, label="attention write")
        ax[col].axhline(0, color="k", lw=0.7)
        mid = fams[fam]["median_importance_depth"]
        ax[col].axvline(mid, color="#27ae60", ls="--", lw=1.2, label="median |importance| depth %d" % mid)
        ax[col].axvspan(18.5, 24.5, color=ATTN_C, alpha=0.07)
        ax[col].set_xlabel("decoder layer (0..%d)" % (n - 1))
        ax[col].set_ylabel("causal denoising effect")
        ax[col].set_title(title, fontsize=9.5)
        ax[col].legend(fontsize=7)

    c = ej["contrast"]
    fams_o = ["antonym", "ioi"]
    sp = [c["eapig_spearman"][f] for f in fams_o]
    bars = ax[2].bar(["antonym\n(minimal pair)", "IOI\n(distributed)"], sp,
                     color=[TERN_C, ATTN_C], alpha=0.9)
    for b, v in zip(bars, sp):
        ax[2].text(b.get_x() + b.get_width() / 2, v, "%.2f" % v, ha="center", va="bottom")
    ax[2].set_ylim(0, 1.0)
    ax[2].set_ylabel(r"EAP-IG vs exact (Spearman)")
    ax[2].set_title("(c) EAP-IG faithfulness is task-dependent\n"
                    "(emb & layer-0 MLP stay ~100% sufficient in BOTH)", fontsize=9.5)

    fig.suptitle("Hardening the ternary-2B causal map (§3.16): the early layer-0 spike is whole-sequence "
                 "early-detok (NOT a minimal-pair artifact), but the LATE read-out is mechanism-specific — "
                 "attention name-movers for IOI vs MLPs for antonyms; EAP-IG fidelity is task-dependent",
                 y=1.04, fontsize=10.5)
    fig.tight_layout(h_pad=1.9); fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    return path

def make_eap_circuits(ej, out_dir):
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    return {"eap_circuits": fig_eap_circuits(ej, os.path.join(fig_dir, "fig22_eap_circuits_2b.png"))}

def fig_eap_positions(ej, path):
    ATTN_C = "#2980b9"; CONTENT_C = "#16a085"; END_C = "#e67e22"
    fams = ej["families"]; n = ej["config"]["n_layers"]
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.3))

    labels, content, end = [], [], []
    for fam in ("ioi", "antonym"):
        ed = fams[fam]["early_detok"]
        for node in ("emb", "m0"):
            labels.append("%s\n%s" % (fam, node)); content.append(ed[node]["content"]); end.append(ed[node]["end"])
    xx = np.arange(len(labels))
    ax[0].bar(xx - 0.2, content, 0.4, color=CONTENT_C, label="patched at CONTENT positions")
    ax[0].bar(xx + 0.2, end, 0.4, color=END_C, label="patched at END position")
    ax[0].set_xticks(xx); ax[0].set_xticklabels(labels, fontsize=8)
    ax[0].set_ylabel("causal denoising effect")
    ax[0].set_ylim(-0.05, 1.1)
    ax[0].set_title("(a) Early-detok node: ~100% at CONTENT, ~0 at END\n(the early spike is detok, not read-out)", fontsize=9.5)
    ax[0].legend(fontsize=7)

    for col, (fam, title) in enumerate([("ioi", "(b) IOI END read-out: ATTENTION (name-movers ~L19-24)"),
                                        ("antonym", "(c) Antonym END read-out: a19 + late MLPs")]):
        ee = fams[fam]["effect"]["end"]
        a_eff = [ee["a%d" % i] for i in range(n)]; m_eff = [ee["m%d" % i] for i in range(n)]
        x = np.arange(n)
        ax[col + 1].bar(x - 0.2, m_eff, 0.4, color=TERN_C, label="MLP write")
        ax[col + 1].bar(x + 0.2, a_eff, 0.4, color=ATTN_C, label="attention write")
        ax[col + 1].axhline(0, color="k", lw=0.7)
        ax[col + 1].axvspan(18.5, 24.5, color=ATTN_C, alpha=0.07)
        ax[col + 1].set_xlabel("decoder layer (0..%d)" % (n - 1))
        ax[col + 1].set_ylabel("END-position causal effect")
        ax[col + 1].set_title(title, fontsize=9.5); ax[col + 1].legend(fontsize=7)

    fig.suptitle("Position-resolved causal attribution (§3.17 fix): the early layer-0/embedding spike is "
                 "detokenization at the CONTENT token (~0 at END), while the next-token decision is READ OUT "
                 "at the END position by late attention (IOI name-movers) — entry and read-out cleanly separate",
                 y=1.04, fontsize=10.5)
    fig.tight_layout(h_pad=1.9); fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    return path

def make_eap_positions(ej, out_dir):
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    return {"eap_positions": fig_eap_positions(ej, os.path.join(fig_dir, "fig23_eap_positions_2b.png"))}

def fig_eap_heads(ej, path):
    fams = ej["families"]; n = ej["config"]["n_layers"]; H = ej["config"]["n_query_heads"]
    arrs = {f: np.array(fams[f]["head_effect_end"]) for f in ("ioi", "antonym")}
    vmax = max(float(np.abs(a).max()) for a in arrs.values())
    fig, ax = plt.subplots(1, 2, figsize=(13.0, 5.6))
    for col, (fam, title) in enumerate([("ioi", "(a) IOI name-mover heads"),
                                        ("antonym", "(b) Antonym read-out heads")]):
        a = arrs[fam]
        im = ax[col].imshow(a, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax, origin="lower")
        for d in fams[fam]["top_heads"][:3]:
            _hi = d["layer"] > n - 4  
            _dy, _va = (-1.7, "top") if _hi else (1.6, "bottom")
            ax[col].text(d["head"], d["layer"] + _dy, "%.2f" % d["effect"], ha="center", va=_va,
                         fontsize=8, fontweight="bold", color="black",
                         bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="0.6", lw=0.5, alpha=0.9))
            ax[col].add_patch(plt.Rectangle((d["head"] - 0.5, d["layer"] - 0.5), 1, 1,
                                            fill=False, edgecolor="#111111", lw=1.8))
        _box = plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor="#111111", lw=1.8, label="top-3 causal head")
        ax[col].legend(handles=[_box], loc="lower right", fontsize=7.5, framealpha=0.9)
        ax[col].set_xlabel("query head (0..%d)" % (H - 1)); ax[col].set_ylabel("decoder layer (0..%d)" % (n - 1))
        top = fams[fam]["top_heads"][0]
        ax[col].set_title("%s — top: L%dH%d (%.2f)" % (title, top["layer"], top["head"], top["effect"]), fontsize=10)
        fig.colorbar(im, ax=ax[col], fraction=0.046, pad=0.04, label="END-position causal effect")
    fig.suptitle("Per-head attribution of the END read-out (ternary BitNet-2B): a SPARSE set of attention "
                 "heads carries the next-token decision — IOI name-movers (L22H17/H19, L27H13) vs antonym "
                 "heads (L19H12/H6, L27H11), task-specific with partial overlap", y=1.02, fontsize=10)
    fig.tight_layout(h_pad=1.9); fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    return path

def make_eap_heads(ej, out_dir):
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    return {"eap_heads": fig_eap_heads(ej, os.path.join(fig_dir, "fig24_eap_heads_2b.png"))}

def fig_eap_heads_path(ej, path):
    CONTENT_C = "#16a085"; END_C = "#e67e22"
    fams = ej["families"]; n = ej["config"]["n_layers"]; H = ej["config"]["n_heads"]
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.6))

    labels, aio, asx = [], [], []
    for fam in ("ioi", "antonym"):
        for hk, v in fams[fam]["causal_name_movers"].items():
            labels.append("%s\n%s" % (fam[:3], hk)); aio.append(v["attn_to_io"]); asx.append(v["attn_to_s"])
    xx = np.arange(len(labels))
    ax[0].bar(xx - 0.2, aio, 0.4, color=CONTENT_C, label="attn END→IO / content")
    ax[0].bar(xx + 0.2, asx, 0.4, color=END_C, label="attn END→subject")
    ax[0].set_xticks(xx); ax[0].set_xticklabels(labels, fontsize=7.5)
    ax[0].set_ylabel("attention weight from END query"); ax[0].set_ylim(0, 1.0)
    ax[0].set_title("(a) Name-mover QK: IOI is heterogeneous\n(L22H19 reads IO; L22H17 reads SUBJECT)", fontsize=9.5)
    ax[0].legend(fontsize=7)

    io = np.array(fams["ioi"]["attn_to_io"]); sx = np.array(fams["ioi"]["attn_to_s"])
    vmax = max(float(io.max()), float(sx.max()))
    for col, (arr, title) in enumerate([(io, "(b) IOI: attention END→IO name"),
                                        (sx, "(c) IOI: attention END→subject name")]):
        im = ax[col + 1].imshow(arr, aspect="auto", cmap="Greens", vmin=0, vmax=vmax, origin="lower")
        for (l, h) in ej["config"]["causal_heads"]["ioi"]:
            ax[col + 1].add_patch(plt.Rectangle((h - 0.5, l - 0.5), 1, 1, fill=False, edgecolor="#c0392b", lw=1.8))
        ax[col + 1].set_xlabel("query head (0..%d)" % (H - 1)); ax[col + 1].set_ylabel("decoder layer (0..%d)" % (n - 1))
        ax[col + 1].set_title(title, fontsize=9.5)
        _cbox = plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor="#c0392b", lw=1.8, label="causal name-mover head")
        ax[col + 1].legend(handles=[_cbox], loc="lower right", fontsize=7.5, framealpha=0.9)
        fig.colorbar(im, ax=ax[col + 1], fraction=0.046, pad=0.04)

    fig.suptitle("QK / path of the name-mover heads (BitNet-2B): the IOI END read-out is a STRUCTURED circuit "
                 "with distinct roles — an IO-reader (L22H19) and a subject-reader (L22H17) — not a uniform "
                 "name-mover set; antonym heads cleanly read the content word (red boxes = §3.17 causal heads)",
                 y=1.03, fontsize=10)
    fig.tight_layout(h_pad=1.9); fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    return path

def make_eap_heads_path(ej, out_dir):
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    return {"eap_heads_path": fig_eap_heads_path(ej, os.path.join(fig_dir, "fig25_eap_heads_path_2b.png"))}

def fig_eap_heads_ov(ej, path):
    COPY_C, SUP_C = "#16a085", "#e67e22"   
    fams = ej["families"]
    specs = {"ioi": ("io", "s", "IO (answer)", "subject (competitor)"),
             "antonym": ("cue", "ans", "cue word X", "answer ¬X")}
    titles = {"ioi": "IOI", "antonym": "antonym"}
    fig, ax = plt.subplots(2, 2, figsize=(13.5, 8.4))
    for col, fam in enumerate(["ioi", "antonym"]):
        heads = list(fams[fam]["heads"].keys())
        ka, kb, la, lb = specs[fam]
        x = np.arange(len(heads))
        for row, (field, semf) in enumerate([("dla", "dla_sem"), ("d_logit_meanabl", "d_logit_meanabl_sem")]):
            sgn = 1.0 if row == 0 else -1.0   
            va = [sgn * fams[fam]["heads"][h][field][ka] for h in heads]
            vb = [sgn * fams[fam]["heads"][h][field][kb] for h in heads]
            ea = [fams[fam]["heads"][h][semf][ka] for h in heads]
            eb = [fams[fam]["heads"][h][semf][kb] for h in heads]
            ax[row, col].bar(x - 0.2, va, 0.4, yerr=ea, capsize=3, color=COPY_C, label=la, alpha=0.9)
            ax[row, col].bar(x + 0.2, vb, 0.4, yerr=eb, capsize=3, color=SUP_C, label=lb, alpha=0.9)
            ax[row, col].axhline(0, color="k", lw=0.8)
            ax[row, col].set_xticks(x); ax[row, col].set_xticklabels(heads, fontsize=8.5)
            ax[row, col].legend(fontsize=7.5, loc="best")
            ax[row, col].grid(axis="y", ls=":", alpha=0.4)
        ax[0, col].set_ylabel("direct logit attribution")
        ax[1, col].set_ylabel("causal contribution\n(clean − mean-ablated logit)")
    ax[0, 0].set_title("(a) DLA — IOI: L22H19/L27H13 COPY the IO (+); L22H17 SUPPRESSES the subject (−)", fontsize=9)
    ax[0, 1].set_title("(b) DLA — antonym: heads WRITE the answer (+), SUPPRESS the cue (−) — not copy heads", fontsize=9)
    ax[1, 0].set_title("(c) causal (mean-ablation) confirms (a)", fontsize=9)
    ax[1, 1].set_title("(d) causal: cue-suppression dominates (answer-writing is direct-path)", fontsize=9)
    fig.suptitle("OV / copy-direction of the name-mover heads (BitNet-2B): the IOI END read-out is a "
                 "COPY+SUPPRESS circuit (positive name-movers L22H19/L27H13 + an S-inhibition head L22H17), "
                 "while antonym heads are NOT copy heads — they suppress the cue X and write the answer ¬X. "
                 "Direct (top) and causal (bottom) agree on the dominant channels; for antonyms cue-suppression "
                 "dominates the full causal footprint while answer-writing is a direct-path effect.",
                 y=1.02, fontsize=9.4)
    fig.tight_layout(h_pad=1.9); fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    return path

def make_eap_heads_ov(ej, out_dir):
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    return {"eap_heads_ov": fig_eap_heads_ov(ej, os.path.join(fig_dir, "fig26_eap_heads_ov_2b.png"))}

def make_h2alloc(h2, out_dir):
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    return {
        "rankings": fig_h2_rankings(h2, os.path.join(fig_dir, "fig08_h2_rankings.png")),
        "pareto": fig_h2_pareto(h2, os.path.join(fig_dir, "fig09_h2_pareto.png")),
    }

def make_h3(h3, out_dir):
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    return {
        "h3": fig_h3_corrected(h3, os.path.join(fig_dir, "fig04_h3_sensitivity.png")),
        "pareto": fig_pareto_corrected(h3, os.path.join(fig_dir, "fig05_pareto.png")),
    }

def make_all(results, out_dir):
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    
    paths = {
        "loss": fig_loss(results, os.path.join(fig_dir, "fig01_loss.png")),
        "ternarization": fig_ternarization(results, os.path.join(fig_dir, "fig02_ternarization.png")),
        "lambda": fig_lambda(results, os.path.join(fig_dir, "fig03_lambda.png")),
    }
    return paths

def fig_h2_2b(h2, path):
    RND, HAWQ, CAU, ORA = "#7f8c8d", "#e67e22", "#c0392b", "#2c3e50"
    cols = {"random": RND, "s_hawq": HAWQ, "c_causal": CAU, "oracle": ORA}
    lab = {"random": "random", "s_hawq": "HAWQ  $s_\\ell$ (2nd-order)",
           "c_causal": "causal  $c_\\ell$", "oracle": "oracle (per-layer)"}
    B = h2["config"]["budgets"]
    base = h2["base"]["ppl_ternary_held"]
    layers = sorted(int(k) for k in h2["recovered"])
    rec = [h2["recovered"][str(l)] for l in layers]
    cl = [h2["c_l"][str(l)] for l in layers]
    sl = [h2["s_l"][str(l)]["trace"] for l in layers]
    rho_c = h2["spearman"]["c_vs_recovered"]; rho_s = h2["spearman"]["s_vs_recovered"]

    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.5))
    
    for nm in ["random", "s_hawq", "c_causal", "oracle"]:
        y = [h2["pareto_joint"][nm][str(b)]["ppl"] for b in B]
        ax[0].plot(B, y, "-o", ms=4, color=cols[nm], label=lab[nm], lw=2 if nm in ("c_causal", "s_hawq") else 1.4)
    ax[0].axhline(base, color="k", ls="--", lw=0.9, label="ternary (no correction)")
    ax[0].set_xlabel("budget  B  (# layers corrected, of 30)")
    ax[0].set_ylabel("held-out test PPL")
    ax[0].set_title("(a) Pareto fronts — HAWQ is WORST; random ≈ causal", fontsize=9.5)
    ax[0].legend(fontsize=7.5, loc="upper right"); ax[0].grid(ls=":", alpha=0.4)
    
    ax[1].scatter(cl, rec, c=CAU, s=28)
    ax[1].set_xscale("symlog", linthresh=0.05)
    ax[1].set_xlabel("$c_\\ell$  (causal importance, mean-ablation Δloss)")
    ax[1].set_ylabel("recovered Δloss (per-layer adapter)")
    ax[1].set_title("(b) $c_\\ell$ vs recoverability:  ρ=%+.2f (p=%.2f)" % (rho_c[0], rho_c[1]), fontsize=9.5)
    ax[1].grid(ls=":", alpha=0.4)
    
    ax[2].scatter(sl, rec, c=HAWQ, s=28)
    ax[2].set_xlabel("$s_\\ell$  (HAWQ Hessian trace)")
    ax[2].set_ylabel("recovered Δloss (per-layer adapter)")
    ax[2].set_title("(c) $s_\\ell$ vs recoverability:  ρ=%+.2f (p=%.2f)  — anti-correlated" % (rho_s[0], rho_s[1]), fontsize=9.5)
    ax[2].grid(ls=":", alpha=0.4)
    fig.suptitle("H2 on the real BitNet-2B (low-rank FP16 correction, joint per-allocation training): causal "
                 "importance $c_\\ell$ Pareto-dominates the HAWQ 2nd-order baseline $s_\\ell$ at every budget — "
                 "but a cheap random spread matches/beats $c_\\ell$. The actionable finding is that HAWQ's "
                 "curvature signal is actively misleading here (it concentrates redundant early layers; $s_\\ell$ "
                 "anti-correlates with recoverability), so spreading the budget beats concentrating it.",
                 y=1.04, fontsize=9.2)
    fig.tight_layout(h_pad=1.9); fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    return path

def make_h2_2b(h2, out_dir):
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    return {"h2_2b": fig_h2_2b(h2, os.path.join(fig_dir, "fig27_h2_correction_2b.png"))}

def fig_h2_2b_hardened(h, path):
    HAWQ, CAU, ORA, RND = "#e67e22", "#c0392b", "#2c3e50", "#95a5a6"
    s = h["summary"]; B = h["config"]["budgets"]; base = s["base_ppl"]
    det = {"s_hawq": HAWQ, "c_causal": CAU, "oracle": ORA}
    lab = {"s_hawq": "HAWQ  $s_\\ell$ (2nd-order)", "c_causal": "causal  $c_\\ell$",
           "oracle": "oracle (per-layer)"}

    fig, ax = plt.subplots(2, 1, figsize=(7.4, 9.6))
    
    rmed = [s["random"][str(b)]["ppl_median"] for b in B]
    rlo = [s["random"][str(b)]["ppl_p05"] for b in B]
    rhi = [s["random"][str(b)]["ppl_p95"] for b in B]
    ax[0].fill_between(B, rlo, rhi, color=RND, alpha=0.25, label="random  p05–p95")
    ax[0].plot(B, rmed, "--", color=RND, lw=1.6, label="random  median")
    for nm, c in det.items():
        y = [s[nm][str(b)]["ppl_mean"] for b in B]
        e = [s[nm][str(b)]["ppl_sd"] for b in B]
        ax[0].errorbar(B, y, yerr=e, fmt="-o", ms=4, color=c, lw=2 if nm != "oracle" else 1.4,
                       capsize=2.5, label=lab[nm])
    ax[0].axhline(base, color="k", ls=":", lw=0.9, label="ternary (no correction) %.1f" % base)
    ax[0].set_xlabel("budget  B  (# layers corrected, of 30)")
    ax[0].set_ylabel("held-out test PPL  (512 seqs, mean±sd over 3 seeds)")
    ax[0].set_title("(a) Pareto fronts with uncertainty — HAWQ robustly WORST", fontsize=9.5)
    ax[0].legend(fontsize=7.3, loc="upper right"); ax[0].grid(ls=":", alpha=0.4)
    
    for i, b in enumerate(B):
        allr = s["random"][str(b)]["ppl_all"]
        xj = i + (np.random.RandomState(b).rand(len(allr)) - 0.5) * 0.28
        ax[1].scatter(xj, allr, s=14, color=RND, alpha=0.7, zorder=2,
                      label="random draws" if i == 0 else None)
        for nm, c in det.items():
            ax[1].scatter(i, s[nm][str(b)]["ppl_mean"], s=46, color=c, zorder=3,
                          marker={"s_hawq": "^", "c_causal": "o", "oracle": "D"}[nm],
                          label=lab[nm] if i == 0 else None)
    ax[1].set_xticks(range(len(B))); ax[1].set_xticklabels(B)
    ax[1].set_xlabel("budget  B")
    ax[1].set_ylabel("held-out test PPL")
    ax[1].set_title("(b) ranking vs the random spread (10 draws/budget)", fontsize=9.5)
    ax[1].legend(fontsize=7.3, loc="upper right"); ax[1].grid(ls=":", alpha=0.4)
    fig.suptitle("H2 hardened (3 seeds + 512-seq test CIs + random as a distribution): causal $c_\\ell$ "
                 "ROBUSTLY beats the HAWQ 2nd-order baseline (6/7 budgets, >2sd at mid-B), confirming "
                 "curvature is the wrong signal at 1.58-bit. But $c_\\ell$'s edge over a random spread is "
                 "modest (beats the random median 4/7) — decisive only at mid budget (B=5: $c_\\ell$ beats all "
                 "10 random draws), washing out at the extremes. Budget dominates ranking; spread > concentrate.",
                 y=1.05, fontsize=9.0)
    fig.tight_layout(h_pad=1.9); fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    return path

def make_h2_2b_hardened(h, out_dir):
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    return {"h2_2b_hardened": fig_h2_2b_hardened(h, os.path.join(fig_dir, "fig28_h2_hardening_2b.png"))}

def fig_h2_2b_greedy(g, h, path):
    GRE, CAU, HAWQ, ORA, RND = "#27ae60", "#c0392b", "#e67e22", "#2c3e50", "#95a5a6"
    s = h["summary"]; gp = g["greedy_pareto"]
    B = [int(b) for b in gp]; B.sort()
    base = g["base"]["ppl"]
    gy = [gp[str(b)]["ppl_mean"] for b in B]
    cy = [s["c_causal"][str(b)]["ppl_mean"] for b in B]
    sy = [s["s_hawq"][str(b)]["ppl_mean"] for b in B]
    oy = [s["oracle"][str(b)]["ppl_mean"] for b in B]
    rmed = [s["random"][str(b)]["ppl_median"] for b in B]
    rlo = [s["random"][str(b)]["ppl_p05"] for b in B]
    rhi = [s["random"][str(b)]["ppl_p95"] for b in B]

    fig, ax = plt.subplots(2, 1, figsize=(7.4, 9.6))
    
    ax[0].fill_between(B, rlo, rhi, color=RND, alpha=0.22, label="random  p05–p95")
    ax[0].plot(B, rmed, "--", color=RND, lw=1.5, label="random  median")
    ax[0].plot(B, sy, "-^", color=HAWQ, lw=1.4, ms=5, label="HAWQ  $s_\\ell$")
    ax[0].plot(B, oy, "-D", color=ORA, lw=1.4, ms=5, label="oracle (per-layer)")
    ax[0].plot(B, cy, "-o", color=CAU, lw=1.6, ms=5, label="causal  $c_\\ell$")
    ax[0].plot(B, gy, "-o", color=GRE, lw=2.6, ms=7, label="greedy (combinatorial)", zorder=5)
    ax[0].axhline(base, color="k", ls=":", lw=0.9, label="ternary %.1f" % base)
    ax[0].set_xlabel("budget  B  (# layers corrected, of 30)")
    ax[0].set_ylabel("held-out test PPL  (512 seqs)")
    ax[0].set_title("(a) the greedy ceiling — below every cheap ranking AND random", fontsize=9.5)
    ax[0].legend(fontsize=7.3, loc="upper right"); ax[0].grid(ls=":", alpha=0.4)
    
    adv_c = [cy[i] - gy[i] for i in range(len(B))]
    adv_r = [rmed[i] - gy[i] for i in range(len(B))]
    x = np.arange(len(B)); w = 0.36
    ax[1].bar(x - w / 2, adv_c, w, color=CAU, label="greedy advantage over $c_\\ell$")
    ax[1].bar(x + w / 2, adv_r, w, color=RND, label="greedy advantage over random median")
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set_xticks(x); ax[1].set_xticklabels(B)
    ax[1].set_xlabel("budget  B")
    ax[1].set_ylabel("PPL reduction vs greedy ceiling")
    ax[1].set_title("(b) ranking matters most at LOW budget (gap shrinks with B)", fontsize=9.5)
    ax[1].legend(fontsize=7.5, loc="upper right"); ax[1].grid(ls=":", alpha=0.4, axis="y")
    fig.suptitle("H2 greedy-joint (combinatorial) oracle on BitNet-2B: a forward-greedy allocation BEATS c_l, HAWQ, "
                 "the single-layer oracle AND the random spread at every budget (random median 5/5, best random draws "
                 "p05 4/5). The gap over the cheap signals is ~2–3 PPL at low budget and shrinks as B grows — so the "
                 "earlier 'budget dominates ranking' was an artefact of WEAK signals: ranking matters a lot, but c_l/"
                 "HAWQ are inadequate proxies for the optimal choice. Greedy path: 28,10,29,26,1,0,27,18 (spreads late+early+mid).",
                 y=1.05, fontsize=8.8)
    fig.tight_layout(h_pad=1.9); fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    return path

def make_h2_2b_greedy(g, h, out_dir):
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    return {"h2_2b_greedy": fig_h2_2b_greedy(g, h, os.path.join(fig_dir, "fig29_h2_greedy_2b.png"))}

def fig_h2_2b_predictor(p, path):
    GRE, CAU, HAWQ, ORA, RND, GEL = "#27ae60", "#c0392b", "#e67e22", "#2c3e50", "#95a5a6", "#8e44ad"
    t = p["table"]; v = p["verdict"]; corr = p["correlations"]
    B = sorted(int(b) for b in t)
    gel = [p["g_front"][str(b)]["ppl_mean"] for b in B]
    gy = [t[str(b)]["greedy"] for b in B]
    cy = [t[str(b)]["c_causal"] for b in B]
    sy = [t[str(b)]["s_hawq"] for b in B]
    rmed = [t[str(b)]["random_median"] for b in B]

    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.8))
    
    ax[0].plot(B, rmed, "--", color=RND, lw=1.5, label="random  median")
    ax[0].plot(B, sy, "-^", color=HAWQ, lw=1.4, ms=5, label="HAWQ  $s_\\ell$")
    ax[0].plot(B, cy, "-o", color=CAU, lw=1.6, ms=5, label="causal  $c_\\ell$")
    ax[0].plot(B, gel, "-s", color=GEL, lw=2.0, ms=6, label="init-grad  $g_\\ell$ (new cheap)", zorder=4)
    ax[0].plot(B, gy, "-o", color=GRE, lw=2.6, ms=7, label="greedy ceiling", zorder=5)
    ax[0].set_xlabel("budget  B  (# layers corrected, of 30)")
    ax[0].set_ylabel("held-out test PPL  (512 seqs)")
    ax[0].set_title("(a) the new cheap signal $g_\\ell$ is WORSE than $c_\\ell$ — 0/5", fontsize=9.5)
    ax[0].legend(fontsize=7.3, loc="upper right"); ax[0].grid(ls=":", alpha=0.4)
    
    sigs = ["$g_\\ell$ (init-grad)", "$c_\\ell$ (causal)", "$s_\\ell$ (HAWQ)", "random med"]
    cols = [GEL, CAU, HAWQ, RND]
    gaps = [v["mean_gap_g_to_greedy"], v["mean_gap_c_to_greedy"],
            float(np.mean([sy[i] - gy[i] for i in range(len(B))])),
            float(np.mean([rmed[i] - gy[i] for i in range(len(B))]))]
    rho = [corr["spearman_greedyorder_vs_g"], corr["spearman_greedyorder_vs_c"],
           corr["spearman_greedyorder_vs_s"], None]
    x = np.arange(len(sigs))
    ax[1].bar(x, gaps, 0.62, color=cols)
    for i, (gp, r) in enumerate(zip(gaps, rho)):
        lab = "gap %.2f" % gp + ("" if r is None else "\n$\\rho_{greedy}$=%.2f" % r)
        ax[1].text(i, gp + 0.05, lab, ha="center", va="bottom", fontsize=7.2)
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set_xticks(x); ax[1].set_xticklabels(sigs, fontsize=8)
    ax[1].set_ylabel("mean PPL gap above greedy ceiling")
    ax[1].set_ylim(0, max(gaps) * 1.3)
    ax[1].set_title("(b) all single-layer scalars leave a gap; $g_\\ell$ the largest", fontsize=9.5)
    ax[1].grid(ls=":", alpha=0.4, axis="y")
    fig.suptitle("H2 cheap-predictor of the greedy choice on BitNet-2B (NEGATIVE): the init-gradient $g_\\ell=\\|\\partial L/\\partial B_\\ell\\|$ "
                 "at the B=0 no-op (one backward) is a WORSE proxy than $c_\\ell$ at every budget (beats $c_\\ell$/random/greedy 0/5; mean gap "
                 "3.15 vs 1.80 ppl). It is stable across A-inits ($\\rho$=0.96) but robustly biased to early layers (top picks L0–L5), while greedy "
                 "spreads late+mid. So curvature, causal-importance AND init-gradient — all single-layer scalars — fail to reach the combinatorial "
                 "ceiling: 'which layers' is genuinely non-local.",
                 y=1.04, fontsize=8.6)
    fig.tight_layout(h_pad=1.9); fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    return path

def make_h2_2b_predictor(p, out_dir):
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    return {"h2_2b_predictor": fig_h2_2b_predictor(p, os.path.join(fig_dir, "fig30_h2_predictor_2b.png"))}

def fig_h2_2b_eap_alloc(p, path):
    EAP, EXA, CAU, HAWQ, RND, GRE, GEL = "#2980b9", "#16a085", "#c0392b", "#e67e22", "#95a5a6", "#27ae60", "#8e44ad"
    t = p["table"]; B = sorted(int(b) for b in t)
    ey = [t[str(b)]["eap_ig"] for b in B]; xy = [t[str(b)]["exact"] for b in B]
    gy = [t[str(b)]["greedy"] for b in B]; cy = [t[str(b)]["c_causal"] for b in B]
    sy = [t[str(b)]["s_hawq"] for b in B]; ry = [t[str(b)]["random_median"] for b in B]

    fig, ax = plt.subplots(2, 1, figsize=(7.4, 9.6))
    
    ax[0].plot(B, ry, "--", color=RND, lw=1.4, label="random  median")
    ax[0].plot(B, sy, "-^", color=HAWQ, lw=1.4, ms=5, label="HAWQ  $s_\\ell$")
    ax[0].plot(B, cy, "-o", color=CAU, lw=1.5, ms=5, label="mean-abl  $c_\\ell$")
    ax[0].plot(B, ey, "-s", color=EAP, lw=1.6, ms=5, label="EAP-IG (proposal's method)")
    ax[0].plot(B, xy, "-D", color=EXA, lw=2.2, ms=6, label="exact task-import (gold)", zorder=4)
    ax[0].plot(B, gy, "-o", color=GRE, lw=2.6, ms=7, label="greedy ceiling", zorder=5)
    ax[0].set_xlabel("budget  B  (# layers corrected, of 30)")
    ax[0].set_ylabel("held-out test PPL  (512 seqs)")
    ax[0].set_title("(a) gold-standard task importance is the best cheap signal — nears the ceiling at low B", fontsize=8.7)
    ax[0].legend(fontsize=7.0, loc="upper right"); ax[0].grid(ls=":", alpha=0.4)
    
    names = ["exact\n(gold)", "$c_\\ell$\n(mean-abl)", "random\nmedian", "EAP-IG\n(proposal)", "HAWQ\n$s_\\ell$", "$g_\\ell$\n(init-grad)"]
    keys = ["exact", "c_causal", "random_median", "eap_ig", "s_hawq", "g_ell"]
    cols = [EXA, CAU, RND, EAP, HAWQ, GEL]
    gaps = [float(np.mean([t[str(b)][k] - t[str(b)]["greedy"] for b in B])) for k in keys]
    x = np.arange(len(names))
    bars = ax[1].bar(x, gaps, 0.66, color=cols)
    bars[0].set_edgecolor("k"); bars[0].set_linewidth(1.6)
    for i, g in enumerate(gaps):
        ax[1].text(i, g + 0.04, "%.2f" % g, ha="center", va="bottom", fontsize=8)
    ax[1].set_xticks(x); ax[1].set_xticklabels(names, fontsize=7.6)
    ax[1].set_ylabel("mean PPL gap above greedy ceiling")
    ax[1].set_ylim(0, max(gaps) * 1.25)
    ax[1].set_title("(b) exact task importance < $c_\\ell$ < random ≈ EAP-IG < HAWQ < $g_\\ell$", fontsize=8.7)
    ax[1].grid(ls=":", alpha=0.4, axis="y")
    fig.suptitle("H2 the proposal's LITERAL signal — EAP-IG task-causal importance for ternary bit-allocation (BitNet-2B, identical 512-test). "
                 "The GOLD-STANDARD exact task importance Pareto-dominates HAWQ (5/5) and is the BEST cheap signal (mean gap to the greedy ceiling "
                 "0.90 ppl, beating $c_\\ell$ at B=1,2,3, near-ceiling at B=2,3) — H2's CORE (causal>curvature) holds. But the scalable EAP-IG "
                 "approximation collapses to ~random (gap 2.06): circuit-ranking fidelity (ρ=0.85) does NOT transfer to allocation; and at B≥5 no "
                 "per-layer signal reaches the ceiling (non-locality persists).",
                 y=1.05, fontsize=8.5)
    fig.tight_layout(h_pad=1.9); fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    return path

def make_h2_2b_eap_alloc(p, out_dir):
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    return {"h2_2b_eap_alloc": fig_h2_2b_eap_alloc(p, os.path.join(fig_dir, "fig31_h2_eap_alloc_2b.png"))}

_DOWNSTREAM_N = {"hellaswag": 10042, "arc_challenge": 1172, "arc_easy": 2376,
                 "winogrande": 1267, "piqa": 1838, "openbookqa": 500}

def fig_h2_2b_downstream(p, path):
    BASEC, CEILC = "#7f8c8d", "#27ae60"
    A = p["allocations"]; base = A["base"]["primary"]; ceil = A["greedy@8"]["primary"]
    tasks = [t for t in p["tasks"] if t in base and t in ceil]
    b = np.array([base[t]["value"] for t in tasks])
    c = np.array([ceil[t]["value"] for t in tasks])
    seb = np.array([math.sqrt(v * (1 - v) / _DOWNSTREAM_N[t]) for v, t in zip(b, tasks)])
    sec = np.array([math.sqrt(v * (1 - v) / _DOWNSTREAM_N[t]) for v, t in zip(c, tasks)])
    z = (c - b) / np.sqrt(seb ** 2 + sec ** 2)
    mb, mc = A["base"]["mean_primary_acc"], A["greedy@8"]["mean_primary_acc"]

    labels = [t.replace("_", "\n") for t in tasks] + ["MEAN\n(6 tasks)"]
    bv = np.append(b, mb); cv = np.append(c, mc)
    bse = np.append(seb, 0.0); cse = np.append(sec, 0.0)
    x = np.arange(len(labels)); w = 0.38
    fig, ax = plt.subplots(figsize=(11.0, 4.9))
    ax.bar(x - w / 2, bv, w, yerr=bse, color=BASEC, capsize=2.5, label="ternary base (B=0)")
    ax.bar(x + w / 2, cv, w, yerr=cse, color=CEILC, capsize=2.5,
           label="greedy@8 — empirical CEILING (best the correction can do)")
    for i in range(len(tasks)):
        ax.text(i, max(bv[i], cv[i]) + max(bse[i], cse[i]) + 0.012,
                "Δ%+.3f\n%.1fσ" % (c[i] - b[i], z[i]), ha="center", va="bottom", fontsize=7.0)
    ax.text(len(tasks), max(mb, mc) + 0.014, "Δ%+.4f" % (mc - mb), ha="center", va="bottom",
            fontsize=7.4, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7.8)
    ax.set_ylabel("accuracy  (acc_norm where defined, else acc)")
    ax.set_ylim(0, 0.92); ax.axhline(0.0, color="k", lw=0.6)
    ax.legend(fontsize=8.0, loc="upper right"); ax.grid(ls=":", alpha=0.4, axis="y")
    fig.suptitle("H2 / C3 the DOWNSTREAM-ACCURACY half — does the perplexity-recovering correction lift accuracy? "
                 "NO (BitNet-2B, full lm-eval test sets). Even the empirical CEILING (greedy@8) moves mean accuracy "
                 "by +0.0037 over the ternary base; NO task reaches significance (max |z|=1.41, hellaswag n=10042). "
                 "The deployed ternary 2B is already near-FP downstream → little gap to recover; a wikitext-LM "
                 "correction does not transfer to QA accuracy. H2's perplexity half holds (fig27–31); its accuracy "
                 "half is a floor at this scale.", y=1.13, fontsize=8.4)
    fig.tight_layout(h_pad=1.9); fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    return path

def make_h2_2b_downstream(p, out_dir):
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    return {"h2_2b_downstream": fig_h2_2b_downstream(p, os.path.join(fig_dir, "fig32_h2_downstream_2b.png"))}

def fig_h2_downstream_named(p, path):
    BASEC, CEILC = "#7f8c8d", "#27ae60"
    A = p["allocations"]

    def g(tag, t, k):
        return A[tag]["metrics"][t][k]
    N = {"mmlu": 14042, "gsm8k": 1319}
    rows = [  
        ("MMLU\n(loglik acc,\nformat-free)", "mmlu", "acc,none", N["mmlu"], "reason"),
        ("GSM8K\nflexible-extract\n(reasoning)", "gsm8k", "exact_match,flexible-extract", N["gsm8k"], "reason"),
        ("GSM8K\nstrict-match\n(format+reasoning)", "gsm8k", "exact_match,strict-match", N["gsm8k"], "format"),
    ]
    b = np.array([g("base", t, k) for _, t, k, _, _ in rows])
    c = np.array([g("greedy@8", t, k) for _, t, k, _, _ in rows])
    ns = np.array([n for *_, n, _ in rows])
    seb = np.sqrt(b * (1 - b) / ns)
    sec = np.sqrt(c * (1 - c) / ns)
    z = (c - b) / np.sqrt(seb ** 2 + sec ** 2)
    labels = [r[0] for r in rows]
    x = np.arange(len(labels)); w = 0.38
    fig, ax = plt.subplots(figsize=(9.6, 5.0))
    ax.bar(x - w / 2, b, w, yerr=seb, color=BASEC, capsize=3, label="ternary base (B=0)")
    ax.bar(x + w / 2, c, w, yerr=sec, color=CEILC, capsize=3,
           label="greedy@8 — empirical CEILING (best the correction can do)")
    for i in range(len(rows)):
        flag = "  ← format only" if rows[i][4] == "format" else ""
        ax.text(i, max(b[i], c[i]) + max(seb[i], sec[i]) + 0.015,
                "Δ%+.3f\n%.1fσ%s" % (c[i] - b[i], z[i], flag),
                ha="center", va="bottom", fontsize=8.0,
                fontweight="bold" if rows[i][4] == "format" else "normal",
                color="#c0392b" if rows[i][4] == "format" else "k")
    
    ax.axvspan(1.5, 2.5, color="#c0392b", alpha=0.06)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8.2)
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 0.78); ax.axhline(0.0, color="k", lw=0.6)
    ax.legend(fontsize=8.4, loc="upper left"); ax.grid(ls=":", alpha=0.4, axis="y")
    fig.suptitle("§3.18 downstream — H2's two LITERALLY-named tasks (MMLU + GSM8K, 5-shot, full sets): the accuracy "
                 "half is a FLOOR even here.\nThe only metric that moves is GSM8K strict-match (+0.063, 3.3σ); MMLU "
                 "and GSM8K flexible-extract — the format-robust, reasoning-capturing metrics — are flat (−0.4σ, −0.5σ).\n"
                 "strict vs flexible score the SAME greedy@8 generations → the gain is a generation-FORMAT shift "
                 "(wikitext LM-adaptation), NOT recovered reasoning.", y=1.16, fontsize=8.2)
    fig.tight_layout(h_pad=1.9); fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    return path

def make_h2_downstream_named(p, out_dir):
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    return {"h2_downstream_named": fig_h2_downstream_named(p, os.path.join(fig_dir, "fig45_h2_downstream_named.png"))}

def fig_soft_quant_tuned(sj, path):
    SOFT_C = "#e67e22"
    s = sj["summary"]; sch = s["schedules"]
    ste_tax, ste_trace = s["ste_tax"], s["ste_trace"]
    
    names = sorted(sch.keys(), key=lambda k: sch[k]["tax_mean"])
    x = np.arange(len(names))
    tax_m = [sch[k]["tax_mean"] for k in names]
    tax_se = [sch[k]["tax_sd"] / math.sqrt(3.0) for k in names]
    dtax = [sch[k]["dtax_vs_ste_mean"] for k in names]
    dse = [sch[k]["dtax_vs_ste_se"] for k in names]
    tr_ratio = [sch[k]["trace_over_ste"] for k in names]
    labels = [n.replace("_", "\n") for n in names]

    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.6))

    bars = ax[0].bar(x, tax_m, yerr=tax_se, capsize=3.5, color=SOFT_C, alpha=0.9)
    ax[0].axhline(ste_tax, color=TERN_C, ls="--", lw=1.6, label="STE baseline tax=%.4f" % ste_tax)
    for i, b in enumerate(bars):
        sig = "" if abs(dtax[i]) > 2 * dse[i] else "  (n.s.)"
        ax[0].text(b.get_x() + b.get_width() / 2, tax_m[i] + tax_se[i] + 0.004,
                   "Δ%+.4f\n±%.4f%s" % (dtax[i], dse[i], sig), ha="center", va="bottom", fontsize=7.0)
    ax[0].set_xticks(x); ax[0].set_xticklabels(labels, fontsize=7.6)
    ax[0].set_ylim(0, max(tax_m) * 1.28)
    ax[0].set_ylabel("ternarization tax (nats) — lower better")
    ax[0].set_title("(a) tax vs STE: cheapest schedules only TIE STE (Δ n.s.)")
    ax[0].legend(fontsize=8.0, loc="upper left")

    barb = ax[1].bar(x, tr_ratio, color=SOFT_C, alpha=0.9)
    ax[1].axhline(1.0, color=TERN_C, ls="--", lw=1.6, label="STE baseline (trace=%.0f)" % ste_trace)
    for i, b in enumerate(barb):
        ax[1].text(b.get_x() + b.get_width() / 2, tr_ratio[i] + 0.01,
                   "×%.2f" % tr_ratio[i], ha="center", va="bottom", fontsize=7.6)
    ax[1].set_xticks(x); ax[1].set_xticklabels(labels, fontsize=7.6)
    ax[1].set_ylabel("Hessian trace ÷ STE — >1 means SHARPER than STE")
    ax[1].set_ylim(0, max(tr_ratio) * 1.18)
    ax[1].set_title("(b) curvature: EVERY schedule is sharper than STE (ratio >1)")
    ax[1].legend(fontsize=8.0, loc="lower right")

    fig.suptitle("Soft-quant tau-schedule sweep (QP4/H4 hardening, 6 schedules × 3 seeds): "
                 "no schedule beats STE on both axes. Best tax (cosine Δ%+.4f, n.s.) is still ×%.2f sharper. "
                 "VERDICT: the naive soft-quant negative (fig14) is NOT a schedule artefact — "
                 "soft-then-hard mismatch is fundamental." % (sch["cosine"]["dtax_vs_ste_mean"],
                 sch["cosine"]["trace_over_ste"]), y=1.04, fontsize=8.6)
    fig.tight_layout(h_pad=1.9); fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    return path

def make_soft_quant_tuned(sj, out_dir):
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    return {"soft_quant_tuned": fig_soft_quant_tuned(sj, os.path.join(fig_dir, "fig33_soft_quant_h4_tuned.png"))}

def fig_h2_2b_cheapgreedy(froz, jw, path):
    FROZ_C, JW_C, GRD_C, C_C, H_C, R_C = "#7f8c8d", "#e67e22", "#2c3e50", "#c0392b", "#16a085", "#95a5a6"
    tab = jw["table"]                                   
    Bs = [int(b) for b in jw["config"]["report_budgets"]]
    x = np.array(Bs, dtype=float)

    def col(t, key):
        return np.array([t[str(B)][key] for B in Bs], dtype=float)

    fig, ax = plt.subplots(2, 1, figsize=(7.4, 9.6))

    ax[0].plot(x, col(tab, "greedy"), "o--", color=GRD_C, lw=2.0, label="greedy oracle (ceiling)")
    ax[0].plot(x, col(jw["table"], "c_causal"), "s-", color=C_C, lw=1.5, label="c_ℓ (causal, best scalar)")
    ax[0].plot(x, col(tab, "s_hawq"), "^-", color=H_C, lw=1.2, label="HAWQ s_ℓ")
    ax[0].plot(x, col(tab, "random_median"), ":", color=R_C, lw=1.4, label="random (median)")
    ax[0].plot(x, col(froz["table"], "cg_probe"), "D-", color=FROZ_C, lw=1.6,
               label="cheap-greedy frozen-S (gap %.2f)" % froz["verdict"]["mean_gap_cg_to_greedy"])
    ax[0].plot(x, col(jw["table"], "cg_probe"), "D-", color=JW_C, lw=2.2,
               label="cheap-greedy jointwarm (gap %.2f, BEST cheap)" % jw["verdict"]["mean_gap_cg_to_greedy"])
    ax[0].set_xlabel("budget B (FP16-corrected layers)")
    ax[0].set_ylabel("test perplexity (512 seqs, 3 seeds) — lower better")
    ax[0].set_xticks(Bs)
    ax[0].set_title("(a) Pareto fronts: cheap jointwarm beats c_ℓ at low B,\nbut never reaches the ceiling (refutes cheap rescue)")
    ax[0].legend(fontsize=7.4, loc="upper right")

    kk = np.arange(1, len(jw["cg_probe_path"]) + 1)
    g_froz = [p["gain"] for p in froz["cg_probe_path"]]
    g_jw = [p["gain"] for p in jw["cg_probe_path"]]
    w = 0.38
    ax[1].bar(kk - w / 2, g_froz, w, color=FROZ_C, label="frozen-S (S can't re-adapt) — NOISE FLOOR")
    ax[1].bar(kk + w / 2, g_jw, w, color=JW_C, label="jointwarm (S re-adapts) — resolved")
    ax[1].axhline(0, color="k", lw=0.6)
    ax[1].set_xlabel("greedy search step k (layer added)")
    ax[1].set_ylabel("candidate marginal gain on SCORE set (nats)")
    ax[1].set_xticks(kk)
    ax[1].set_title("(b) Self-correction: the frozen-S probe was mis-specified\n(letting S re-adapt resolves gains 10–50×)")
    ax[1].legend(fontsize=8.0, loc="upper left")

    fig.suptitle("H2/C3 §3.18 — a cheap, faithful approximation of the greedy allocation oracle "
                 "(BitNet-2B, test-512, 3 seeds). Well-powered jointwarm cheap-greedy is the best cheap "
                 "signal (mean gap-to-ceiling %.2f vs c_ℓ %.2f) and beats c_ℓ for B≤3, yet 0/5 layer "
                 "overlap with greedy and ~random by B≥5 → 'which layers' stays hard; the optimum is "
                 "likely non-unique." % (jw["verdict"]["mean_gap_cg_to_greedy"],
                 jw["verdict"]["mean_gap_c_to_greedy"]), y=1.05, fontsize=8.8)
    fig.tight_layout(h_pad=1.9); fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    return path

def make_h2_2b_cheapgreedy(froz, jw, out_dir):
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    return {"h2_2b_cheapgreedy": fig_h2_2b_cheapgreedy(
        froz, jw, os.path.join(fig_dir, "fig34_h2_cheapgreedy_2b.png"))}

def fig_h2_2b_domain(a2, r1, r2, path):
    WIKI_C, FW_C = "#2c3e50", "#27ae60"
    runs = [("train wiki\n(A2)", a2), ("train fineweb\n(R1)", r1), ("train mix\n(R2)", r2)]
    base_w = a2["base"]["wikitext_ppl"]
    base_f = a2["base"]["fineweb_ppl"]
    x = np.arange(len(runs)); w = 0.38
    fig, ax = plt.subplots(2, 1, figsize=(7.4, 9.6))

    rec_w = [d["verdict"]["wikitext"]["frac_recovered_greedy8"] * 100 for _, d in runs]
    rec_f = [d["verdict"]["fineweb"]["frac_recovered_greedy8"] * 100 for _, d in runs]
    bw = ax[0].bar(x - w / 2, rec_w, w, color=WIKI_C, label="eval wikitext")
    bf = ax[0].bar(x + w / 2, rec_f, w, color=FW_C, label="eval FineWeb-Edu")
    ax[0].axhline(0, color="k", lw=0.9)
    for bars, vals in ((bw, rec_w), (bf, rec_f)):
        for b, v in zip(bars, vals):
            ax[0].text(b.get_x() + b.get_width() / 2, v + (1.2 if v >= 0 else -1.2),
                       "%+.1f%%" % v, ha="center", va="bottom" if v >= 0 else "top", fontsize=8.0)
    ax[0].set_xticks(x); ax[0].set_xticklabels([n for n, _ in runs], fontsize=8.4)
    ax[0].set_ylabel("% of base PPL recovered (greedy@8)")
    ax[0].set_title("(a) The correction recovers ONLY its training domain\n(the mix recovers both; <0 = worse than ternary)")
    ax[0].legend(fontsize=8.4, loc="lower right")
    ax[0].set_ylim(min(rec_f) - 8, max(rec_w) * 1.18)

    ppl_w = [d["fronts"]["greedy@8"]["wikitext"]["ppl_mean"] for _, d in runs]
    ppl_f = [d["fronts"]["greedy@8"]["fineweb"]["ppl_mean"] for _, d in runs]
    cw = ax[1].bar(x - w / 2, ppl_w, w, color=WIKI_C, label="wikitext test")
    cf = ax[1].bar(x + w / 2, ppl_f, w, color=FW_C, label="FineWeb-Edu test")
    ax[1].axhline(base_w, color=WIKI_C, ls="--", lw=1.5, label="wiki base (ternary) %.1f" % base_w)
    ax[1].axhline(base_f, color=FW_C, ls="--", lw=1.5, label="fineweb base (ternary) %.1f" % base_f)
    for bars, vals in ((cw, ppl_w), (cf, ppl_f)):
        for b, v in zip(bars, vals):
            ax[1].text(b.get_x() + b.get_width() / 2, v + 0.5, "%.1f" % v, ha="center", va="bottom", fontsize=8.0,
                       bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85))  
    ax[1].set_xticks(x); ax[1].set_xticklabels([n for n, _ in runs], fontsize=8.4)
    ax[1].set_ylabel("test PPL (greedy@8) — lower better")
    ax[1].set_title("(b) absolute levels: a bar BELOW its dashed base = improved")
    ax[1].legend(fontsize=7.8, loc="upper right", ncol=1)
    ax[1].set_ylim(0, max(max(ppl_w), base_w) * 1.22)

    fig.suptitle("H2/C3 §3.18 — domain-control trilogy (BitNet-2B, rank-8 LoRA, greedy@8, test-512, 3 seeds): "
                 "the low-rank correction's PPL gain is DOMAIN-ADAPTATION to its training corpus, not a "
                 "transferable quantization correction. Wiki-train recovers wiki %+.0f%% but fineweb %+.0f%%; "
                 "fineweb-train is the mirror; the wiki+fineweb MIX recovers both (%+.0f%% / %+.0f%%) → the "
                 "method is domain-general with a representative corpus."
                 % (rec_w[0], rec_f[0], rec_w[2], rec_f[2]), y=1.04, fontsize=8.5)
    fig.tight_layout(h_pad=1.9); fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    return path

def make_h2_2b_domain(a2, r1, r2, out_dir):
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    return {"h2_2b_domain": fig_h2_2b_domain(
        a2, r1, r2, os.path.join(fig_dir, "fig35_h2_domain_control_2b.png"))}

def make_granularity(gran, out_dir):
    P = gran["pareto"]["c_causal"]
    rec = gran["per_layer_recoverability"]
    budgets = gran["budgets"]
    n = len(rec)
    colors = {"attn": "#2980b9", "mlp": "#c0392b", "both": "#27ae60"}
    fig, ax = plt.subplots(2, 1, figsize=(7.4, 9.6))
    for blk in ["attn", "mlp", "both"]:
        xs = [P[blk][str(B)]["extra_mem_frac"] for B in budgets]
        ys = [P[blk][str(B)]["test_ppl"] for B in budgets]
        ax[0].plot(xs, ys, marker="o", color=colors[blk],
                   label=(blk + "-only" if blk != "both" else "both (attn+mlp)"))
        for B, x, y in zip(budgets, xs, ys):
            ax[0].annotate("B%d" % B, (x, y), fontsize=7, alpha=0.6,
                           xytext=(2, 3), textcoords="offset points")
    base = gran["base_test_ppl"]
    ax[0].axhline(base, ls=":", color="gray", label="ternary base %.1f" % base)
    ax[0].set_xlabel("extra memory fraction"); ax[0].set_ylabel("held-out PPL")
    ax[0].set_title("(a) Pareto by block — equal parameter budget"); ax[0].legend(fontsize=8)
    layers = list(range(n))
    att = [rec[str(l)]["attn"]["recovered_dloss"] for l in layers]
    mlp = [rec[str(l)]["mlp"]["recovered_dloss"] for l in layers]
    ax[1].plot(layers, att, marker=".", color=colors["attn"], label="attn-only")
    ax[1].plot(layers, mlp, marker=".", color=colors["mlp"], label="mlp-only")
    ax[1].set_xlabel("layer"); ax[1].set_ylabel("recovered Δloss (test)")
    ax[1].set_title("(b) per-layer recoverability attn vs mlp"); ax[1].legend(fontsize=8)
    fig.suptitle("H2/C3 §3.18 — attn-vs-MLP granularity (BitNet-2B, rank-8 LoRA, c_causal ranking, single seed)",
                 y=1.03, fontsize=10)
    path = os.path.join(out_dir, "figures/fig40_h2_granularity_2b.png")
    fig.tight_layout(h_pad=1.9); fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    return path

def make_gate(gate, out_dir):
    P = gate["pareto"]
    budgets = gate["budgets"]
    variants = ["static", "scalar_gate", "channel_gate", "input_gate"]
    colors = {"static": "#7f8c8d", "scalar_gate": "#2980b9",
              "channel_gate": "#e67e22", "input_gate": "#c0392b"}
    labs = {"static": "static (§3.18)", "scalar_gate": "scalar gate (free)",
            "channel_gate": "channel gate", "input_gate": "input gate (§4.1)"}
    fig, ax = plt.subplots(2, 1, figsize=(7.4, 9.6))
    for v in variants:
        ys = [P[v][str(B)]["test_ppl"] for B in budgets]
        ax[0].plot(budgets, ys, marker="o", color=colors[v], label=labs[v])
        xs = [P[v][str(B)]["extra_mem_frac"] for B in budgets]
        ax[1].plot(xs, ys, marker="o", color=colors[v], label=labs[v])
    base = gate["base_test_ppl"]
    for a in ax:
        a.axhline(base, ls=":", color="gray", label="ternary base %.1f" % base)
        a.set_ylabel("held-out PPL"); a.legend(fontsize=8)
    ax[0].set_xlabel("budget B (layers corrected)"); ax[0].set_title("(a) PPL vs budget — equal layers")
    ax[1].set_xlabel("extra memory fraction"); ax[1].set_title("(b) PPL vs memory — honest equal-cost")
    fig.suptitle("H2/C3 §3.18 — learned HGF gate (BitNet-2B, rank-8 LoRA, c_causal ranking, single seed)",
                 y=1.03, fontsize=10)
    path = os.path.join(out_dir, "figures/fig41_h2_gate_2b.png")
    fig.tight_layout(h_pad=1.9); fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    return path

def make_crosscoder_scale(cc, out_dir):
    m = cc["metrics"]
    edges = np.array(m["r_hist_edges"]); counts = np.array(m["r_hist_counts"])
    centers = 0.5 * (edges[:-1] + edges[1:])
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].bar(centers, counts, width=(edges[1] - edges[0]) * 0.9, color="#8e44ad", alpha=0.8)
    ax[0].axvspan(0.35, 0.65, color="gray", alpha=0.15, label="shared band")
    ax[0].axvline(0.2, ls=":", color="#2c3e50"); ax[0].axvline(0.8, ls=":", color="#c0392b")
    ax[0].set_xlabel("relative decoder norm r  (0=FP-excl, 1=BitNet-excl)")
    ax[0].set_ylabel("# latents"); ax[0].legend(fontsize=8)
    ax[0].set_title("(a) r distribution — bimodal, ZERO fp-side (r_med=%.2f)" % m["r_median"])
    
    labels = ["FVE\nBitNet", "FVE\nFP(Llama)"]
    vals = [m["final_fve_bitnet"], m["final_fve_fp"]]
    bars = ax[1].bar(labels, vals, color=["#c0392b", "#2c3e50"], alpha=0.85)
    ax[1].set_ylim(0, 1.05); ax[1].set_ylabel("fraction variance explained")
    for b, v in zip(bars, vals):
        ax[1].text(b.get_x() + b.get_width() / 2, v + 0.02, "%.3f" % v, ha="center", fontsize=10)
    sc = cc["scales"]
    ax[1].set_title("(b) reconstruction asymmetry confounds r\n(raw act-norm scale: BitNet %.0f vs FP %.2f)"
                    % (sc[0], sc[1]), fontsize=9)
    fig.suptitle("C1 §3.13 — scaled diff BitNet-2B (d%d) vs %s (d%d): confounded by reconstruction asymmetry"
                 % (cc["dims"][0], cc["models"]["fp"].split("/")[-1], cc["dims"][1]), y=1.03, fontsize=9.5)
    path = os.path.join(out_dir, "figures/fig42_crosscoder_scale.png")
    fig.tight_layout(h_pad=1.9); fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    return path

def make_dynamics_scale(ladder, tstar, out_dir):
    order = ["S", "M", "L"]
    sc = ladder["scales"]
    x = [np.mean([s["params_nonembed"] for s in sc[n]["seeds"].values()]) / 1e6 for n in order]

    def col(key):  
        out = []
        for n in order:
            v = [sd[key] for sd in sc[n]["seeds"].values()]
            out.append((float(np.mean(v)), float(np.std(v))))
        return [m for m, _ in out], [e for _, e in out]

    tax_m, tax_e = col("tax")
    tr_m, tr_e = [], []
    l0_m, l0_e = [], []
    for n in order:
        tr = [sd["trace"]["ratio_tern_over_fp"] for sd in sc[n]["seeds"].values()]
        l0 = [sd["l0_over_median"] for sd in sc[n]["seeds"].values()]
        tr_m.append(float(np.mean(tr))); tr_e.append(float(np.std(tr)))
        l0_m.append(float(np.mean(l0))); l0_e.append(float(np.std(l0)))

    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    panels = [(ax[0, 0], tax_m, tax_e, "ternarization tax  (nats, tern-FP val loss)",
               "(a) tax SHRINKS with scale", TERN_C),
              (ax[0, 1], tr_m, tr_e, "Hessian-trace ratio  tern / FP",
               "(b) curvature gap GROWS (>1)", "#8e44ad"),
              (ax[1, 0], l0_m, l0_e, "layer-0 PTQ sensitivity / median layer",
               "(c) layer-0 dominance INTENSIFIES", "#16a085")]
    for a, m, e, ylab, title, c in panels:
        a.errorbar(x, m, yerr=e, marker="o", ms=8, lw=2, color=c, capsize=4)
        for xi, mi, n in zip(x, m, order):
            a.annotate(n, (xi, mi), textcoords="offset points", xytext=(6, 6), fontsize=10, weight="bold")
        a.set_xscale("log"); a.set_xlabel("non-embedding params (M, log)")
        a.set_ylabel(ylab); a.set_title(title, fontsize=10)
    panels[1][0].axhline(1.0, ls=":", color="gray", lw=1)  

    rows = tstar["rows"]
    fracs = sorted({r["t_frac"] for r in rows})
    fp = float(np.mean([r["final_val_loss"] for r in rows if r["t_frac"] == 1.0]))
    fm, fe = [], []
    for f in fracs:
        v = [r["final_val_loss"] - fp for r in rows if r["t_frac"] == f]
        fm.append(float(np.mean(v))); fe.append(float(np.std(v)))
    ad = ax[1, 1]
    ad.errorbar(fracs, fm, yerr=fe, marker="o", ms=7, lw=2, color="#c0392b", capsize=4)
    imin = int(np.argmin([m for m, f in zip(fm, fracs) if f < 1.0]))
    fmin = [f for f in fracs if f < 1.0][imin]
    ad.axvline(fmin, ls="--", color="#16a085", lw=1.5, label="min @ t*=%.2f" % fmin)
    ad.axhline(0.0, ls=":", color="gray", lw=1, label="FP floor (t*=1.0)")
    qat = [m for m, f in zip(fm, fracs) if f == 0.0][0]
    red = 100 * (1 - min(m for m, f in zip(fm, fracs) if f < 1.0) / qat)
    ad.set_xlabel("t*  (FP→ternary switch fraction; 0=QAT-from-start, 1=pure FP)")
    ad.set_ylabel("tax over FP floor  (nats)")
    ad.set_title("(d) t*-switch U-curve @ M (125M): min t*=0.5, -%.0f%% tax" % red, fontsize=10)
    ad.legend(fontsize=8)
    fig.suptitle("T2 §3.19 — dynamics at scale: the toy ternary story persists S→M→L "
                 "(wikitext-103, ternary QAT vs FP twin, 3 seeds)", y=1.02, fontsize=11)
    path = os.path.join(out_dir, "figures/fig43_dynamics_scale.png")
    fig.tight_layout(h_pad=1.9); fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    return path

def make_dfc_scale(dj, out_dir):
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    d = dj["dfc"]; part = d["partition"]; sc = dj["std_crosscoder"]["metrics"]
    BIT_C, FP_LAB_C, SHARED_C = TERN_C, FP_C, "#7f8c8d"
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.3))

    labels = ["BitNet-2B\n(ternary)", "Llama-3.2-1B\n(FP twin-of-method)"]
    base = [d["fve_shared_only_bitnet"], d["fve_shared_only_fp"]]
    gain = [d["excl_gain_bitnet"], d["excl_gain_fp"]]
    full = [d["fve_full_bitnet"], d["fve_full_fp"]]
    xpos = np.arange(2)
    ax[0].bar(xpos, base, width=0.55, color=[BIT_C, FP_LAB_C], alpha=0.55, label="shared-only FVE")
    ax[0].bar(xpos, gain, width=0.55, bottom=base, color=[BIT_C, FP_LAB_C], alpha=1.0,
              hatch="//", edgecolor="white", label="exclusive gain")
    for xi, b, g, f in zip(xpos, base, gain, full):
        ax[0].text(xi, b / 2, "%.2f" % b, ha="center", va="center", fontsize=9, weight="bold", color="white")
        ax[0].text(xi, b + g / 2, "+%.2f" % g, ha="center", va="center", fontsize=9, weight="bold", color="black",
                   bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.85))
        ax[0].text(xi, f + 0.02, "full %.2f" % f, ha="center", va="bottom", fontsize=9, weight="bold")
    ax[0].set_xticks(xpos); ax[0].set_xticklabels(labels, fontsize=8)
    ax[0].set_ylabel("held-out fraction of variance explained")
    ax[0].set_ylim(0, 1.08)
    ax[0].set_title("(a) DFC held-out FVE: exclusive GAIN\nLlama gains MORE (+0.22) than BitNet (+0.13)", fontsize=9.5)
    ax[0].legend(fontsize=8, loc="upper left")

    keys = [("I_A_bitnet_excl", "BitNet-excl\nI_A", BIT_C),
            ("I_B_fp_excl", "Llama-excl\nI_B", FP_LAB_C),
            ("I_S_shared", "shared\nI_S", SHARED_C)]
    xs = np.arange(3)
    nalive = [part[k]["n_alive"] for k, _, _ in keys]
    ntot = [part[k]["n"] for k, _, _ in keys]
    cols = [c for _, _, c in keys]
    ax[1].bar(xs, ntot, width=0.6, color=cols, alpha=0.3, label="slots allocated")
    ax[1].bar(xs, nalive, width=0.6, color=cols, alpha=1.0, label="alive (used)")
    for xi, a, t in zip(xs, nalive, ntot):
        ax[1].text(xi, a + max(ntot) * 0.01, "%d/%d\n(%.0f%%)" % (a, t, 100 * a / t),
                   ha="center", va="bottom", fontsize=8)
    ax[1].set_xticks(xs); ax[1].set_xticklabels([lab for _, lab, _ in keys], fontsize=8)
    ax[1].set_ylabel("dictionary slots")
    ax[1].set_title("(b) Both models use exclusive slots\n(BitNet 774, Llama 150 alive)", fontsize=9.5)
    ax[1].legend(fontsize=8)

    e = np.array(sc["r_hist_edges"]); ctr = (e[:-1] + e[1:]) / 2
    cnt = np.array(sc["r_hist_counts"], float); cnt = cnt / cnt.sum()
    ax[2].bar(ctr, cnt, width=0.045, color=SHARED_C, alpha=0.9)
    ax[2].axvline(0.5, color="k", ls=":", lw=1, label="shared (r=0.5)")
    ax[2].axvline(sc["r_median"], color=BIT_C, ls="--", lw=1.5, label="median r=%.2f" % sc["r_median"])
    ax[2].set_xlabel("relative decoder norm  r  (0=Llama-only, 1=BitNet-only)")
    ax[2].set_ylabel("fraction of latents")
    ax[2].set_title("(c) Std-crosscoder CONFOUND: %.0f%% labelled\nBitNet-excl, %.0f%% Llama-excl (artifact)"
                    % (100 * sc["frac_bitnet_excl"], 100 * sc["frac_fp_excl"]), fontsize=9.5)
    ax[2].legend(fontsize=8)

    fig.suptitle("§3.13 DFC scaled cross-architecture diff (BitNet-2B vs Llama-3.2-1B): the standard "
                 "crosscoder's '51%% BitNet-exclusive / 0%% Llama-exclusive' is a reconstruction-asymmetry "
                 "artifact;\nDFC severs it and finds BOTH models carry exclusive content (Llama MORE) "
                 "-- but a cross-ARCH diff is not a ternary-attribution (that stays the §3.9 same-arch twin, ~0)",
                 y=1.05, fontsize=9.5)
    path = os.path.join(fig_dir, "fig44_dfc_scale.png")
    fig.tight_layout(h_pad=1.9); fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    return path

def make_ha1_transfer(base, fp16, lora, ppl, out_dir):
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    BASEC, LORAC, CEILC = "#7f8c8d", TERN_C, "#27ae60"
    tasks = [("MMLU\n(loglik acc)", "mmlu_acc", 0.25),
             ("GSM8K\nstrict-match", "gsm8k_strict", 0.0),
             ("GSM8K\nflexible-extract", "gsm8k_flexible", 0.0)]
    b = np.array([base[k] for _, k, _ in tasks], float)
    l = np.array([lora[k] for _, k, _ in tasks], float)
    f = np.array([fp16[k] for _, k, _ in tasks], float)
    chance = [c for *_, c in tasks]
    
    gap = f - b
    rec = np.where(np.abs(gap) > 1e-9, (l - b) / gap, 0.0)

    fig, ax = plt.subplots(1, 2, figsize=(12.4, 5.0))
    x = np.arange(len(tasks)); w = 0.26
    ax[0].bar(x - w, b, w, color=BASEC, label="ternary base (PT-BitNet, no LoRA)")
    ax[0].bar(x, l, w, color=LORAC, label="ternary + LoRA-64 KD (full pipeline)")
    ax[0].bar(x + w, f, w, color=CEILC, label="FP16 teacher (ceiling)")
    for i, c in enumerate(chance):
        ax[0].plot([x[i] - 1.6 * w, x[i] + 1.6 * w], [c, c], ls="--", lw=1.0,
                   color="k", alpha=0.55, label="chance" if i == 0 else None)
    ax[0].set_xticks(x); ax[0].set_xticklabels([t[0] for t in tasks], fontsize=8.6)
    ax[0].set_ylabel("accuracy"); ax[0].set_ylim(0, 0.72)
    ax[0].legend(fontsize=8.0, loc="upper left"); ax[0].grid(ls=":", alpha=0.4, axis="y")
    ax[0].set_title("(a) Downstream levels: base AND +LoRA at the FLOOR; only FP16 has the skill",
                    fontsize=9.2)

    ppl_rec = (ppl["base_ratio"] - ppl["lora_ratio"]) / (ppl["base_ratio"] - 1.0)
    bars = list(100 * rec) + [100 * ppl_rec]
    blabs = ["MMLU\nacc", "GSM8K\nstrict", "GSM8K\nflex", "WikiText-2\nPPL\n(in-run)"]
    cols = [LORAC, LORAC, LORAC, "#2980b9"]
    xb = np.arange(len(bars))
    ax[1].bar(xb, bars, color=cols, alpha=0.9)
    ax[1].axhline(0, color="k", lw=0.6)
    for i, v in enumerate(bars):
        ax[1].text(i, v + (2 if v >= 0 else -2), "%.0f%%" % v, ha="center",
                   va="bottom" if v >= 0 else "top", fontsize=9, fontweight="bold")
    ax[1].axvspan(2.5, 3.5, color="#2980b9", alpha=0.06)
    ax[1].set_xticks(xb); ax[1].set_xticklabels(blabs, fontsize=8.4)
    ax[1].set_ylabel("fraction of FP gap recovered by +LoRA (%)")
    ax[1].set_ylim(-12, 112); ax[1].grid(ls=":", alpha=0.4, axis="y")
    ax[1].set_title("(b) (iii) PPL recovered $\\gg$ (ii) accuracy recovered $\\approx 0$\n"
                    "PPL says 'recovered', the task says 'dead'", fontsize=9.2)

    fig.suptitle("§4 Aposta 1 / H-A1 -- the PPL$\\neq$accuracy dissociation TRANSFERS QAT$\\to$PTQ "
                 "(BitNet-2B $\\to$ Phi-2 ternarized train-free).\nPhi-2-ternary base collapses BOTH downstream "
                 "(MMLU $\\approx$chance, GSM8K 0/0) AND in PPL (71$\\times$ FP, in-run); the LoRA-64 KD "
                 "correction recovers $\\sim$99% of the (huge) PPL gap but $\\sim$0% of the accuracy gap.\n"
                 "strict==flexible==0 at base AND +LoRA -> no format artifact (the arithmetic skill is genuinely "
                 "absent, not mis-extracted). Optimizing perplexity optimizes the wrong metric -- now shown in PTQ too.",
                 y=1.13, fontsize=8.6)
    path = os.path.join(fig_dir, "fig46_ha1_transfer.png")
    fig.tight_layout(h_pad=1.9); fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    return path

def make_ha2_ladder(arms, base, fp16, out_dir):
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    order = ["base", "wikitext", "instruct", "task", "fp16"]
    cols = {"base": "#7f8c8d", "wikitext": "#95a5a6", "instruct": "#f39c12", "task": TERN_C, "fp16": "#27ae60"}

    def vals(arm, key):
        if arm == "base":
            return [base[key]]
        if arm == "fp16":
            return [fp16[key]]
        return [j[key] for j in arms[arm]]

    def mean_range(arm, key):
        v = vals(arm, key); m = sum(v) / len(v)
        return m, m - min(v), max(v) - m

    tasks = [("MMLU\n(loglik acc)", "mmlu_acc", 0.25),
             ("GSM8K\nflexible-extract", "gsm8k_flexible", 0.0),
             ("GSM8K\nstrict-match", "gsm8k_strict", 0.0)]
    fig, ax = plt.subplots(1, 2, figsize=(13.0, 5.2))

    x = np.arange(len(tasks)); w = 0.16
    for i, arm in enumerate(order):
        m = [mean_range(arm, k)[0] for _, k, _ in tasks]
        lo = [mean_range(arm, k)[1] for _, k, _ in tasks]
        hi = [mean_range(arm, k)[2] for _, k, _ in tasks]
        ax[0].bar(x + (i - 2) * w, m, w, yerr=[lo, hi], capsize=2, color=cols[arm],
                  label=arm if arm not in ("base", "fp16") else (arm + " (anchor)"))
    for i, (_, _, ch) in enumerate(tasks):
        if ch > 0:
            ax[0].plot([x[i] - 2.6 * w, x[i] + 2.6 * w], [ch, ch], ls="--", lw=1.0, color="k",
                       alpha=0.55, label="chance" if i == 0 else None)
    ax[0].set_xticks(x); ax[0].set_xticklabels([t[0] for t in tasks], fontsize=8.6)
    ax[0].set_ylabel("accuracy"); ax[0].set_ylim(0, 0.66)
    ax[0].legend(fontsize=7.6, loc="upper right", ncol=2); ax[0].grid(ls=":", alpha=0.4, axis="y")
    ax[0].set_title("(a) Ladder: wikitext $<$ instruct $<$ task (all far below FP); task-aligned lifts most",
                    fontsize=9.0)

    keys = [("MMLU", "mmlu_acc"), ("GSM8K-flex", "gsm8k_flexible")]
    arms3 = ["wikitext", "instruct", "task"]
    xb = np.arange(len(arms3)); ww = 0.36
    for j, (lab, k) in enumerate(keys):
        rec = [100 * (mean_range(a, k)[0] - base[k]) / (fp16[k] - base[k]) for a in arms3]
        ax[1].bar(xb + (j - 0.5) * ww, rec, ww, color=["#2980b9", TERN_C][j], alpha=0.9, label=lab)
        for i, v in enumerate(rec):
            ax[1].text(xb[i] + (j - 0.5) * ww, v + 0.4, "%.0f%%" % v, ha="center", va="bottom", fontsize=8.2)
    ax[1].set_xticks(xb); ax[1].set_xticklabels(arms3, fontsize=8.8)
    ax[1].set_ylabel("fraction of FP gap recovered (%)"); ax[1].set_ylim(0, 14)
    ax[1].axvspan(1.5, 2.5, color=TERN_C, alpha=0.06)
    ax[1].legend(fontsize=8.4, loc="upper left"); ax[1].grid(ls=":", alpha=0.4, axis="y")
    ax[1].set_title("(b) task recovers $\\sim$10% of the FP gap (z=7.2/5.3 vs wikitext, 2 seeds);\n"
                    "instruct $>$ wikitext (z=2.7/3.0) but 2.3$\\times$ $<$ task $\\Rightarrow$ TASK-DOMINANT", fontsize=9.0)

    fig.suptitle("§4 Aposta 2 / H-A2 -- a TASK-ALIGNED correction (same rank-64 budget, only the KD corpus "
                 "changes) LIFTS the Aposta-1 downstream floor -- modestly and task-DOMINANTLY.\n"
                 "Phi-2 ternary: task-corpus correction recovers MMLU +10.6% / GSM8K-flex +8.3% of the FP gap "
                 "(strict==flex $\\Rightarrow$ real, not format); wikitext stays at the floor, generic-instruction "
                 "lifts slightly (z$\\approx$2.7-3.0) but task lifts $2.3\\times$ more.\n"
                 "So the floor is PARTLY an objective/data-alignment artifact, NOT fully intrinsic to rank-64 "
                 "ternary -- but $\\sim$90% of the gap remains (lifts off the floor $\\neq$ recovers the skill).",
                 y=1.13, fontsize=8.6)
    path = os.path.join(fig_dir, "fig47_ha2_ladder.png")
    fig.tight_layout(h_pad=1.9); fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    return path
