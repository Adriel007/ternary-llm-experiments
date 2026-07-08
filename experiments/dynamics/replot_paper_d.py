
import json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 13, "axes.titlesize": 13, "axes.labelsize": 13,
    "xtick.labelsize": 11.5, "ytick.labelsize": 11.5, "legend.fontsize": 10.5,
    "axes.grid": True, "grid.alpha": 0.3, "savefig.dpi": 300,
})
TERN_C, FP_C = "#c0392b", "#2c3e50"
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "reports", "data")
FIGS = os.path.join(ROOT, "papers", "paper-d-dynamics", "figures")
L = lambda n: json.load(open(os.path.join(DATA, n)))

def _avg(runs, sect, key):
    xs = runs[0][sect]["step"]
    Y = np.array([r[sect][key] for r in runs], dtype=float)
    return np.array(xs), Y.mean(0), Y.std(0)

def fig1_tax(results, path):                                    
    tern, fp = results["runs"]["ternary"], results["runs"]["fp"]
    fig, ax = plt.subplots(2, 1, figsize=(6.8, 8.6))
    for runs, c, lab in [(fp, FP_C, "FP twin"), (tern, TERN_C, "Ternary (W1.58A8)")]:
        xs, m, s = _avg(runs, "train", "train_loss")
        ax[0].plot(xs, m, color=c, label=lab); ax[0].fill_between(xs, m - s, m + s, color=c, alpha=0.18)
        xs, m, s = _avg(runs, "eval", "val_loss")
        ax[1].plot(xs, m, color=c, marker="o", ms=4, label=lab); ax[1].fill_between(xs, m - s, m + s, color=c, alpha=0.18)
    ax[0].set_title("Training loss"); ax[1].set_title("Validation loss")
    for a in ax:
        a.set_xlabel("step"); a.set_ylabel("cross-entropy"); a.legend()
    fig.tight_layout(); fig.savefig(path); plt.close(fig)

def fig2_lattice(results, path):                                
    tern = results["runs"]["ternary"]
    xs, mz, sz = _avg(tern, "tern", "frac_zero")
    _, md, sd = _avg(tern, "tern", "dist_to_lattice")
    fig, ax = plt.subplots(2, 1, figsize=(6.8, 8.4))
    ax[0].plot(xs, mz, color=TERN_C); ax[0].fill_between(xs, mz - sz, mz + sz, color=TERN_C, alpha=0.18)
    ax[0].set_title("Fraction of zeroed weights"); ax[0].set_ylabel("% zeros"); ax[0].set_ylim(0, 1)
    ax[1].plot(xs, md, color=TERN_C); ax[1].fill_between(xs, md - sd, md + sd, color=TERN_C, alpha=0.18)
    ax[1].set_title("Distance of latent weights to ternary lattice"); ax[1].set_ylabel("mean |w·s − round(w·s)|")
    for a in ax:
        a.set_xlabel("step")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)

def fig3_flatness(fj, path):                                    
    s = fj["summary"]
    fig, ax = plt.subplots(2, 1, figsize=(6.4, 8.4))
    for a, metric, title in [(ax[0], "trace", r"Hessian trace (avg curvature)"),
                             (ax[1], "lambda_max", r"top eigenvalue $\lambda_{\max}$")]:
        m = [s[metric]["fp"]["mean"], s[metric]["ternary"]["mean"]]
        sd = [s[metric]["fp"]["std"], s[metric]["ternary"]["std"]]
        a.bar(["FP twin", "Ternary"], m, yerr=sd, capsize=6, color=[FP_C, TERN_C], alpha=0.9)
        a.set_title(title)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)

def fig4_softquant(sj, path):                                   
    SOFT_C = "#e67e22"
    s = sj["summary"]
    fig, ax = plt.subplots(2, 1, figsize=(6.4, 8.6))
    for a, metric, title, fmt in [
        (ax[0], "tax", "Ternarization tax (nats) — lower better", "%.3f"),
        (ax[1], "trace", "Hessian trace (avg curvature) — lower flatter", "%.0f")]:
        m = [s["ste"][metric]["mean"], s["soft"][metric]["mean"]]
        sd = [s["ste"][metric]["std"], s["soft"][metric]["mean"] * 0 + s["soft"][metric]["std"]]
        bars = a.bar(["STE\n(baseline)", "soft-annealed\n(H4)"], m, yerr=sd, capsize=6,
                     color=[TERN_C, SOFT_C], alpha=0.9)
        off = 0.04 * max(m)
        for b, v, e in zip(bars, m, sd):                         
            a.text(b.get_x() + b.get_width() / 2, v + e + off, fmt % v, ha="center", va="bottom")
        a.set_ylim(0, max(mi + ei for mi, ei in zip(m, sd)) * 1.18)
        a.set_title(title)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)

def fig5_tstar(tj, path):                                       
    s = tj["summary"]
    fr = sorted(float(k) for k in s)
    mean = [s[str(f) if str(f) in s else ("%.1f" % f)]["tax_mean"] for f in fr]
    std = [s[str(f) if str(f) in s else ("%.1f" % f)]["tax_std"] for f in fr]
    fig, ax = plt.subplots(2, 1, figsize=(6.8, 8.6))
    ax[0].errorbar(fr, mean, yerr=std, color=TERN_C, marker="o", ms=7, capsize=4)
    ax[0].set_yscale("log"); ax[0].set_xlabel("t*  (fraction of training in FP before the ternary switch)")
    ax[0].set_ylabel("ternarization tax (nats, log)"); ax[0].set_title("Full sweep — t*=1.0 (PTQ) is catastrophic")
    ax[0].annotate("PTQ\n(no QAT)", (1.0, mean[-1]), textcoords="offset points", xytext=(-12, -30), ha="center")
    z = [(f, m, e) for f, m, e in zip(fr, mean, std) if f <= 0.75]
    zf, zm, ze = [a[0] for a in z], [a[1] for a in z], [a[2] for a in z]
    ax[1].errorbar(zf, zm, yerr=ze, color=TERN_C, marker="o", ms=7, capsize=4)
    bi = int(np.argmin(zm))
    ax[1].scatter([zf[bi]], [zm[bi]], color="#27ae60", zorder=5, s=110, label="optimum t*=%.2f" % zf[bi])
    ax[1].set_xlabel("t*"); ax[1].set_ylabel("ternarization tax (nats)")
    ax[1].set_title("Zoom (t*≤0.75): U-shape, optimum at t*≈0.5"); ax[1].legend()
    fig.tight_layout(); fig.savefig(path); plt.close(fig)

def fig6_tstar_emergence(ej, path):                             
    pf = ej["summary"]["per_frac"]; cp = ej["summary"]["coupling"]
    fr = sorted(float(k) for k in pf); K = lambda f: "%.3f" % f
    tax = [pf[K(f)]["final_tax_mean"] for f in fr]; txs = [pf[K(f)]["final_tax_std"] for f in fr]
    lfp = [pf[K(f)]["L_fp_mean"] for f in fr]; rptq = [pf[K(f)]["R_ptq_mean"] for f in fr]
    dfp = [pf[K(f)]["D_fp_mean"] for f in fr]; zfp = [pf[K(f)]["z_fp_mean"] for f in fr]
    PURPLE, BLUE = "#8e44ad", "#2980b9"
    fig, ax = plt.subplots(2, 2, figsize=(12.5, 10.0))
    a = ax[0, 0]
    a.errorbar(fr, tax, yerr=txs, color=TERN_C, marker="o", ms=6, capsize=4)
    bi = int(np.argmin(tax))
    a.scatter([fr[bi]], [tax[bi]], color="#27ae60", zorder=5, s=95, label="optimum t*=%.2f" % fr[bi])
    a.set_xlabel("t*  (FP fraction before the ternary switch)"); a.set_ylabel("ternarization tax (nats)")
    a.set_title("(a) The U-shape to explain (13-pt grid, 3 seeds)"); a.legend()
    a = ax[0, 1]
    a.plot(fr, lfp, color=FP_C, marker="o", ms=5, label=r"$L_{fp}$ FP basin (readiness)")
    a.set_yscale("log"); a.set_xlabel("t*"); a.set_ylabel(r"$L_{fp}$ (nats, log)", color=FP_C); a.tick_params(axis="y", labelcolor=FP_C)
    a2 = a.twinx(); a2.plot(fr, rptq, color=TERN_C, marker="s", ms=5, ls="--", label=r"$R_{ptq}$ instant PTQ gap")
    a2.set_ylabel(r"$R_{ptq}$ (nats)", color=TERN_C); a2.tick_params(axis="y", labelcolor=TERN_C); a2.grid(False)
    a.set_title("(b) Readiness × budget: basin improves, PTQ gap grows")
    h1, l1 = a.get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels()
    a.legend(h1 + h2, l1 + l2, loc="upper center")
    a = ax[1, 0]
    a.plot(fr, dfp, color=PURPLE, marker="o", ms=5, label=r"$D_{fp}$ dist-to-lattice")
    a.set_xlabel("t*"); a.set_ylabel(r"$D_{fp}$  mean$|w/\beta-\mathrm{round}|$", color=PURPLE); a.tick_params(axis="y", labelcolor=PURPLE)
    dr, zr = cp["D_fp_range"], cp["z_fp_range"]; a.set_ylim(dr[0] - 0.02, dr[1] + 0.02)
    a2 = a.twinx(); a2.plot(fr, zfp, color=BLUE, marker="s", ms=5, ls="--", label=r"$z_{fp}$ zero-fraction")
    a2.set_ylabel(r"$z_{fp}$ zero-fraction", color=BLUE); a2.tick_params(axis="y", labelcolor=BLUE)
    a2.set_ylim(zr[0] - 0.02, zr[1] + 0.02); a2.grid(False)
    a.set_title("(c) Modes do NOT emerge: $D_{fp}$, $z_{fp}$ flat across t")
    h1, l1 = a.get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels()
    a.legend(h1 + h2, l1 + l2, loc="center right")
    a = ax[1, 1]
    hb = np.array(ej["config"]["hist_bins"]); ctr = (hb[:-1] + hb[1:]) / 2
    hist = ej["hist_seed0_by_step"]; cols = ["#bdc3c7", "#e67e22", "#1a5276"]
    for st, c in zip(["0", "1000", "1900"], cols):
        if st in hist:
            h = np.array(hist[st], float); h = h / h.sum()
            a.plot(ctr, h, color=c, lw=1.9, label="FP step %s" % st)
    for x in (-1.0, 0.0, 1.0):
        a.axvline(x, color="k", ls=":", lw=0.8, alpha=0.5)
    a.set_xlabel(r"$w/\beta$  (latent weight / absmean scale)"); a.set_ylabel("fraction of weights"); a.set_xlim(-3, 3)
    a.set_title("(d) FP weight histogram keeps its shape\n(no trimodal {-1,0,+1} emergence)"); a.legend()
    fig.tight_layout(); fig.savefig(path); plt.close(fig)

def fig7_scale(ladder, tstar, path):                            
    order = ["S", "M", "L"]; sc = ladder["scales"]
    x = [np.mean([s["params_nonembed"] for s in sc[n]["seeds"].values()]) / 1e6 for n in order]

    def col(key):
        out = [(float(np.mean([sd[key] for sd in sc[n]["seeds"].values()])),
                float(np.std([sd[key] for sd in sc[n]["seeds"].values()]))) for n in order]
        return [m for m, _ in out], [e for _, e in out]

    tax_m, tax_e = col("tax"); tr_m, tr_e, l0_m, l0_e = [], [], [], []
    for n in order:
        tr = [sd["trace"]["ratio_tern_over_fp"] for sd in sc[n]["seeds"].values()]
        l0 = [sd["l0_over_median"] for sd in sc[n]["seeds"].values()]
        tr_m.append(float(np.mean(tr))); tr_e.append(float(np.std(tr)))
        l0_m.append(float(np.mean(l0))); l0_e.append(float(np.std(l0)))
    fig, ax = plt.subplots(2, 2, figsize=(12.5, 9.4))
    panels = [(ax[0, 0], tax_m, tax_e, "ternarization tax (nats)", "(a) tax SHRINKS with scale", TERN_C),
              (ax[0, 1], tr_m, tr_e, "Hessian-trace ratio tern / FP", "(b) curvature gap GROWS (>1)", "#8e44ad"),
              (ax[1, 0], l0_m, l0_e, "layer-0 PTQ sens. / median layer", "(c) layer-0 dominance INTENSIFIES", "#16a085")]
    for a, m, e, ylab, title, c in panels:
        a.errorbar(x, m, yerr=e, marker="o", ms=8, lw=2, color=c, capsize=4)
        for xi, mi, n in zip(x, m, order):
            a.annotate(n, (xi, mi), textcoords="offset points", xytext=(6, 6), fontsize=11, weight="bold")
        a.set_xscale("log"); a.set_xlabel("non-embedding params (M, log)"); a.set_ylabel(ylab); a.set_title(title)
    panels[1][0].axhline(1.0, ls=":", color="gray", lw=1)
    rows = tstar["rows"]; fracs = sorted({r["t_frac"] for r in rows})
    fp = float(np.mean([r["final_val_loss"] for r in rows if r["t_frac"] == 1.0]))
    fm, fe = [], []
    for f in fracs:
        v = [r["final_val_loss"] - fp for r in rows if r["t_frac"] == f]
        fm.append(float(np.mean(v))); fe.append(float(np.std(v)))
    ad = ax[1, 1]; ad.errorbar(fracs, fm, yerr=fe, marker="o", ms=7, lw=2, color="#c0392b", capsize=4)
    imin = int(np.argmin([m for m, f in zip(fm, fracs) if f < 1.0])); fmin = [f for f in fracs if f < 1.0][imin]
    ad.axvline(fmin, ls="--", color="#16a085", lw=1.5, label="min @ t*=%.2f" % fmin)
    ad.axhline(0.0, ls=":", color="gray", lw=1, label="FP floor (t*=1.0)")
    qat = [m for m, f in zip(fm, fracs) if f == 0.0][0]
    red = 100 * (1 - min(m for m, f in zip(fm, fracs) if f < 1.0) / qat)
    ad.set_xlabel("t*  (FP→ternary switch fraction)"); ad.set_ylabel("tax over FP floor (nats)")
    ad.set_title("(d) t*-switch U-curve @ M (125M): min t*=0.5, -%.0f%% tax" % red); ad.legend()
    fig.tight_layout(); fig.savefig(path); plt.close(fig)

if __name__ == "__main__":
    res = L("results.json")
    fig1_tax(res, os.path.join(FIGS, "fig1_tax.png"))
    fig2_lattice(res, os.path.join(FIGS, "fig2_lattice.png"))
    fig3_flatness(L("flatness.json"), os.path.join(FIGS, "fig3_flatness.png"))
    fig4_softquant(L("soft_quant_h4.json"), os.path.join(FIGS, "fig4_softquant.png"))
    fig5_tstar(L("t_star.json"), os.path.join(FIGS, "fig5_tstar.png"))
    fig6_tstar_emergence(L("t_star_emergence.json"), os.path.join(FIGS, "fig6_tstar_emergence.png"))
    fig7_scale(L("dynamics_scale_ladder.json"), L("dynamics_scale_tstar.json"),
               os.path.join(FIGS, "fig7_scale.png"))
    print("OK: 7 figuras do paper-D regeneradas em", FIGS)
