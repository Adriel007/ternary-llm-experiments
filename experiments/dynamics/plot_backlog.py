from __future__ import annotations

import json
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  

plt.rcParams.update({"figure.dpi": 130, "font.size": 10, "axes.grid": True, "grid.alpha": 0.3})

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "reports/data")
FIGDIR = os.path.join(ROOT, "reports/figures")
TERN_C, FP_C = "#c0392b", "#2c3e50"

def _load(name):
    p = os.path.join(DATA, name)
    return json.load(open(p)) if os.path.exists(p) else None

def fig48_gate_hard(d):
    S = d["summary"]; B = d["budgets"]; variants = d["variants"]
    cols = {"static": "#7f8c8d", "scalar_gate": "#2980b9", "channel_gate": "#e67e22", "input_gate": "#c0392b"}
    labs = {"static": "static (§3.18)", "scalar_gate": "scalar gate (free, 1 param)",
            "channel_gate": "channel gate", "input_gate": "input gate (§4.1)"}
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.6))
    x = np.arange(len(B)); w = 0.2
    for i, v in enumerate(variants):
        m = [S[v][str(b)]["mean_ppl"] for b in B]
        e = [S[v][str(b)]["ci95_halfwidth"] for b in B]
        ax[0].bar(x + (i - 1.5) * w, m, w, yerr=e, capsize=3, color=cols[v], label=labs[v])
    ax[0].set_xticks(x); ax[0].set_xticklabels(["B=%d" % b for b in B])
    ax[0].set_ylabel("held-out PPL (512-seq test)"); ax[0].set_ylim(17, 28.5)
    ax[0].legend(fontsize=8); ax[0].grid(ls=":", alpha=0.4, axis="y")
    ax[0].set_title("(a) PPL ± 95%% CI (3 seeds) — every gate < static at every B", fontsize=9.2)
    for v in variants:
        if v == "static":
            continue
        dm = [S[v][str(b)]["delta_mean"] for b in B]
        de = [S[v][str(b)]["delta_ci95_halfwidth"] for b in B]
        ax[1].errorbar(x, dm, yerr=de, marker="o", capsize=3, color=cols[v], label=labs[v])
    ax[1].axhline(0, ls="--", color="k", alpha=0.6)
    ax[1].set_xticks(x); ax[1].set_xticklabels(["B=%d" % b for b in B])
    ax[1].set_ylabel("PPL reduction vs static (paired)"); ax[1].legend(fontsize=8)
    ax[1].set_title("(b) paired Δ(static−gate) > 0, CI excludes 0 — 3/3 seeds, 5/5 budgets", fontsize=9.2)
    fig.suptitle("Gate HARDENING (§3.18/fig41 positive) — BitNet-2B, rank-8 LoRA, c_causal, 3 seeds + 512-seq test.\n"
                 "The single-seed gate win SURVIVES the honesty bar: learned gate robustly beats static; input gate dominates.",
                 y=1.06, fontsize=8.8)
    p = os.path.join(FIGDIR, "fig48_gate_hard.png")
    fig.tight_layout(h_pad=1.9); fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    return p

def fig49_b6_heads(d):
    acc = np.array(d["head_effect_end_pooled"])          
    named = d["named_heads"]; tmpls = list(d["config"]["templates"].keys())
    floor = d["noise_floor_median_abs"]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1.5, 1]})
    vmax = np.percentile(acc, 99.5)
    im = ax[0].imshow(acc, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax, origin="lower")
    ax[0].set_xlabel("query head"); ax[0].set_ylabel("layer")
    ax[0].set_title("(a) per-head END causal effect (pooled, %d pairs / 3 templates)"
                    % d["config"]["n_pairs_total"], fontsize=9.2)
    for i, k in enumerate(named):
        l, h = int(k.split("L")[1].split("H")[0]), int(k.split("H")[1])
        ax[0].add_patch(plt.Rectangle((h - .5, l - .5), 1, 1, fill=False, edgecolor="k", lw=1.8))
        
        ax[0].annotate(k, (h, l), color="k", fontsize=8, fontweight="bold", ha="center", va="top",
                       xytext=(h, l - 3.5 - 2.6 * i), textcoords="data",
                       bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="0.6", lw=0.5, alpha=0.9),
                       arrowprops=dict(arrowstyle="-", color="0.4", lw=0.7))
    _b6box = plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor="k", lw=1.8, label="named head (top-3)")
    ax[0].legend(handles=[_b6box], loc="lower right", fontsize=8, framealpha=0.9)
    fig.colorbar(im, ax=ax[0], fraction=0.046, label="normalized effect")
    x = np.arange(len(named)); w = 0.25
    cols = ["#c0392b", "#e67e22", "#2980b9"]
    for j, t in enumerate(tmpls):
        vals = [named[k]["per_template_effect"][t] for k in named]
        ax[1].bar(x + (j - 1) * w, vals, w, color=cols[j], label="template: %s" % t)
    ax[1].axhline(floor, ls=":", color="gray", label="noise floor %.4f" % floor)
    ax[1].set_xticks(x); ax[1].set_xticklabels(list(named.keys()))
    ax[1].set_ylabel("END causal effect"); ax[1].legend(fontsize=8)
    ax[1].set_title("(b) named heads survive ALL 3 templates\n(top-K each; %d–%d× the noise floor)"
                    % (int(min(named[k]["pooled_effect"] for k in named) / floor),
                       int(max(named[k]["pooled_effect"] for k in named) / floor)), fontsize=9.2)
    fig.suptitle("B6 — §3.17 IOI head anatomy HARDENED: 3 templates × %d pairs (vs 1×10). "
                 "L22H17 (S-inhibition) + L22H19/L27H13 (copy name-movers) are the pooled top-3 and in every "
                 "template's top-K." % d["config"]["n_pairs_total"], y=1.04, fontsize=8.8)
    p = os.path.join(FIGDIR, "fig49_b6_heads.png")
    fig.tight_layout(h_pad=1.9); fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    return p

def fig50_b4(d, ref):
    ioi = np.array(d["ioi_layer_importance"])
    rl = np.array(ref["depth"]["layer_importance_exact"])
    n = min(len(ioi), len(rl)); ioi, rl = ioi[:n], rl[:n]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    ax[0].scatter(rl, ioi, c=np.arange(n), cmap="viridis", s=40)
    for i in range(n):
        if i in d["ioi_top_layers"][:4] or i in d["ref_top_layers"][:4]:
            ax[0].annotate("L%d" % i, (rl[i], ioi[i]), fontsize=7.5)
    ax[0].set_xlabel("§3.16 minimal-pair layer importance (c_exact)")
    ax[0].set_ylabel("IOI layer importance (c_exact)")
    ax[0].set_title("(a) per-layer c_exact: IOI vs §3.16  (Spearman ρ=%.3f)"
                    % d["layer_spearman_ioi_vs_ref"], fontsize=9.4)
    x = np.arange(n); w = 0.42
    ax[1].bar(x - w / 2, rl / rl.sum(), w, color=FP_C, label="§3.16 minimal-pair")
    ax[1].bar(x + w / 2, ioi / ioi.sum(), w, color=TERN_C, label="IOI (B6 pool)")
    ax[1].set_xlabel("layer"); ax[1].set_ylabel("normalized importance"); ax[1].legend(fontsize=8.5)
    ax[1].set_title("(b) both rankings put L0 first + share the late cluster (18/19/21/23)", fontsize=9.2)
    fig.suptitle("B4 — c_exact STABILITY across task families. Layer-level ρ=%.3f (node-level ρ=%.3f); "
                 "the causal LAYER-allocation signal is model-level, not prompt-family-specific → supports "
                 "the H2 'causal > curvature' headline." % (d["layer_spearman_ioi_vs_ref"],
                 d["node_spearman_ioi_vs_ref"]), y=1.04, fontsize=8.8)
    p = os.path.join(FIGDIR, "fig50_b4_cexact_transfer.png")
    fig.tight_layout(h_pad=1.9); fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    return p

def fig51_robust(d):
    c6 = d["c6"]; B = d["c6_budgets"]; c7 = d["c7"]
    cols = {"c_causal": TERN_C, "s_hawq": "#e67e22", "random": "#7f8c8d"}
    fig, ax = plt.subplots(2, 1, figsize=(7.4, 9.4))
    for nm in c6:
        ax[0].plot(B, [c6[nm][str(b)]["ppl_128"] for b in B], marker="o", ls="-", color=cols[nm],
                   label="%s @128" % nm)
        ax[0].plot(B, [c6[nm][str(b)]["ppl_1024"] for b in B], marker="s", ls="--", color=cols[nm],
                   label="%s @1024" % nm, alpha=0.8)
    ax[0].axhline(d["base_ppl_128"], ls=":", color="gray", alpha=0.7, label="base @128 %.1f" % d["base_ppl_128"])
    ax[0].axhline(d["base_ppl_1024"], ls=":", color="black", alpha=0.5, label="base @1024 %.1f" % d["base_ppl_1024"])
    ax[0].set_xlabel("budget B (layers corrected)"); ax[0].set_ylabel("held-out PPL")
    ax[0].legend(fontsize=7.3, ncol=2); ax[0].set_title("(a) C6 — fronts hold at 8× context (c_causal > s_hawq at all B)", fontsize=9.0)
    
    labels = ["Spearman\n(c_ℓ vs full)", "top-5\nJaccard", "top-8\nJaccard"]
    meds = [c7["spearman_median"], c7["top5_jaccard_median"], c7["top8_jaccard_median"]]
    lo = [c7["spearman_median"] - c7["spearman_p05"], 0, 0]
    hi = [c7["spearman_p95"] - c7["spearman_median"], 0, 0]
    ax[1].bar(range(3), meds, color=[TERN_C, "#2980b9", "#27ae60"], yerr=[lo, hi], capsize=4)
    for i, v in enumerate(meds):
        ax[1].text(i, v + hi[i] + 0.02, "%.3f" % v, ha="center", fontsize=9)  
    ax[1].set_xticks(range(3)); ax[1].set_xticklabels(labels, fontsize=8.5)
    ax[1].set_ylim(min(0.8, min(meds) - 0.05), 1.10); ax[1].set_ylabel("stability (median over %d bootstraps)" % c7["n_boot"])
    ax[1].set_title("(b) C7 — c_ℓ ranking stable under calib resampling\n(%d calib seqs; min ρ=%.3f)"
                    % (c7["n_calib_seqs"], d["c7"].get("spearman_p05", 0)), fontsize=9.0)
    fig.suptitle("C6+C7 — robustness of the §3.18 allocation. Fronts survive 128→1024 context; "
                 "c_ℓ ranking Spearman %.3f under bootstrap." % c7["spearman_median"], y=1.04, fontsize=8.8)
    p = os.path.join(FIGDIR, "fig51_robust.png")
    fig.tight_layout(h_pad=1.9); fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    return p

def fig52_tstar_control(d):
    S = d["summary"]; U = d["u_curve"]; tf = d["config"]["t_fracs"]; modes = d["config"]["modes"]
    cols = {"cosine": FP_C, "constant": TERN_C, "rewarm": "#27ae60"}
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.6))
    for m in modes:
        ys = [S[m][str(t)]["tax_mean"] for t in tf]
        es = [S[m][str(t)]["tax_std"] for t in tf]
        ax[0].errorbar(tf, ys, yerr=es, marker="o", capsize=3, color=cols[m],
                       label="%s (best t*=%.2f)" % (m, U[m]["best_t_frac"]))
    ax[0].set_xlabel("t* (FP-pretrain fraction before ternary switch)")
    ax[0].set_ylabel("ternarization tax (nats)"); ax[0].legend(fontsize=8.5)
    ax[0].set_title("(a) tax vs t* by LR rule (3 seeds)", fontsize=9.4)
    
    for m in modes:
        ys = [S[m][str(t)]["tax_mean"] for t in tf if t < 1.0]
        ax[1].plot([t for t in tf if t < 1.0], ys, marker="o", color=cols[m], label=m)
    ax[1].set_xlabel("t* (excluding t*=1.0 PTQ collapse)"); ax[1].set_ylabel("tax (nats)")
    ax[1].legend(fontsize=8.5)
    ax[1].set_title("(b) interior: cosine dips at mid; constant/rewarm decline monotonically", fontsize=9.0)
    u_cos, u_con = U["cosine"]["u_shaped"], U["constant"]["u_shaped"]
    fig.suptitle("C2 — is the t* optimum real or an LR-schedule artefact? Cosine U-shaped=%s, "
                 "constant-LR U-shaped=%s.\nWhat is ROBUST: some FP pretrain helps, full PTQ (t*=1.0) "
                 "collapses; the PRECISE optimum location is schedule-dependent." % (u_cos, u_con),
                 y=1.05, fontsize=8.8)
    p = os.path.join(FIGDIR, "fig52_tstar_control.png")
    fig.tight_layout(h_pad=1.9); fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    return p

def fig53_adherence(d):
    A = d["allocations"]; V = d["verdict"]
    arms = ["base", "greedy@8"]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    metrics = [("strict-match", "strict_match"), ("flexible-extract", "flexible_extract"),
               ("#### adherence", None)]
    x = np.arange(len(metrics)); w = 0.36
    for i, arm in enumerate(arms):
        vals = [A[arm]["strict_match"], A[arm]["flexible_extract"], A[arm]["adherence"]["adherence_rate"]]
        ax[0].bar(x + (i - 0.5) * w, vals, w, color=[FP_C, TERN_C][i], label=arm)
    ax[0].set_xticks(x); ax[0].set_xticklabels([m[0] for m in metrics], fontsize=8.8)
    ax[0].set_ylabel("rate"); ax[0].legend(fontsize=9)
    ax[0].set_title("(a) GSM8K base vs greedy@8 (n=%d)" % A["base"]["adherence"]["n"], fontsize=9.4)
    deltas = [V["delta_strict"], V["delta_flexible"], V["delta_adherence"]]
    ax[1].bar(range(3), deltas, color=["#c0392b", "#7f8c8d", "#2980b9"])
    for i, v in enumerate(deltas):
        ax[1].text(i, v + (0.003 if v >= 0 else -0.006), "%+.3f" % v, ha="center", fontsize=9)
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set_xticks(range(3)); ax[1].set_xticklabels(["Δstrict", "Δflexible", "Δadherence"], fontsize=8.8)
    ax[1].set_ylabel("greedy@8 − base")
    ax[1].set_title("(b) Δstrict tracks Δadherence, Δflexible≈0\n→ strict gain is FORMAT (explains=%s)"
                    % V["adherence_explains_strict"], fontsize=9.0)
    fig.suptitle("§2.3 template-adherence — the fig45 GSM8K strict-match gain is a MEASURED format effect: "
                 "the correction shifts generation toward the `#### N` template (Δadherence), not arithmetic "
                 "(Δflexible≈0).", y=1.04, fontsize=8.8)
    p = os.path.join(FIGDIR, "fig53_adherence.png")
    fig.tight_layout(h_pad=1.9); fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    return p

def main():
    os.makedirs(FIGDIR, exist_ok=True)
    done = []
    g = _load("h2_2b_gate_hard.json");      done += [fig48_gate_hard(g)] if g else []
    b6 = _load("eap_ig_heads_b6.json");     done += [fig49_b6_heads(b6)] if b6 else []
    b4 = _load("eap_ig_b4.json"); ref = _load("eap_ig_2b.json")
    if b4 and ref:                          done += [fig50_b4(b4, ref)]
    rb = _load("h2_2b_robust.json");        done += [fig51_robust(rb)] if rb else []
    c2 = _load("t_star_control.json");      done += [fig52_tstar_control(c2)] if c2 else []
    ad = _load("gsm8k_adherence.json");     done += [fig53_adherence(ad)] if ad else []
    for p in done:
        print("wrote", p)

if __name__ == "__main__":
    main()
