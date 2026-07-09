
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 12, "axes.titlesize": 12, "axes.labelsize": 12,
                     "xtick.labelsize": 10.5, "ytick.labelsize": 10.5, "legend.fontsize": 9.5})

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATA = os.path.join(ROOT, "experiments", "ternary-quantization", "data")
OUTDIRS = [os.path.join(ROOT, "papers", "paper-b-kplane", "figures"),
           os.path.join(ROOT, "papers", "arxivorg", "paper-b-kplane", "figures")]

def load(name):
    p = os.path.join(DATA, name + ".jsonl")
    return [json.loads(l) for l in open(p) if l.strip()]

EXCLUDE_FLEX = {"Qwen/Qwen2.5-14B-Instruct"}

def fam(m):
    m = m.split("/")[-1]
    ml = m.lower()
    if "coder" in ml: return "Qwen2.5-Coder"
    if "qwen2.5" in ml: return "Qwen2.5"
    if "qwen3-30b" in ml: return "Qwen3-MoE"
    if "qwen3" in ml: return "Qwen3"
    if "granite" in ml: return "Granite"
    if "phi-3" in ml: return "Phi-3.5"
    if "phi-4" in ml: return "Phi-4"
    if "mistral" in ml: return "Mistral"
    if "glm" in ml: return "GLM"
    return m

def save(fig, stem):
    for d in OUTDIRS:
        fig.savefig(os.path.join(d, stem + ".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

def f1():
    
    def ret_series(records, scoref, keyfn=lambda r: r["model"]):
        by = {}
        for r in records:
            by.setdefault(keyfn(r), {})[int(r["K"])] = scoref(r)
        out = {2: [], 3: []}
        for m, d in by.items():
            if 0 in d and d[0] > 0:
                for K in (2, 3):
                    if K in d: out[K].append(100 * d[K] / d[0])
        return out
    s3 = load("lmeval_stage3")
    gsm = ret_series([r for r in s3 if r["task"] == "gsm8k"], lambda r: r["value"])
    f5 = [r for r in load("f5_qwen25inst") if r["model"] not in EXCLUDE_FLEX]
    gsm5 = ret_series(f5, lambda r: r["metrics"]["exact_match,flexible-extract"])
    for K in (2, 3): gsm[K] += gsm5[K]
    arc = ret_series([r for r in s3 if r["task"] == "arc_challenge"], lambda r: r["value"])
    mmlu = ret_series(load("mmlu_official"), lambda r: r["metrics"]["acc,none"])
    know = {K: arc[K] + mmlu[K] for K in (2, 3)}

    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    
    for name, ser, col in [("math (GSM8K)", gsm, "#c0392b"),
                           ("knowledge (ARC+MMLU, model-task pairs)", know, "#2e86de")]:
        xs = [0, 2, 3]
        mean = [100] + [np.mean(ser[2]), np.mean(ser[3])]
        lo = [100] + [min(ser[2]), min(ser[3])]
        hi = [100] + [max(ser[2]), max(ser[3])]
        ax.plot(xs, mean, "o-", color=col, lw=2, label=f"{name}  (n={len(ser[2])})")
        ax.fill_between(xs, lo, hi, color=col, alpha=0.15)
        print(f"F1 {name}: K2 {np.mean(ser[2]):.0f}% [{min(ser[2]):.0f}-{max(ser[2]):.0f}]  "
              f"K3 {np.mean(ser[3]):.0f}% [{min(ser[3]):.0f}-{max(ser[3]):.0f}]  n={len(ser[2])}")
    ax.axhline(100, ls=":", color="grey", lw=1)
    ax.set_xticks([0, 2, 3]); ax.set_xticklabels(["K=0 (FP)", "K=2 (~3.3 bpw)", "K=3 (~4.9 bpw)"])
    ax.set_ylabel("retention vs FP (%)"); ax.set_ylim(0, 108)
    ax.set_title("The K-lever: a 3rd trit-plane rescues reasoning specifically\n"
                 "(K2 collapses math to a cliff; knowledge degrades gracefully)")
    ax.legend(loc="lower right", fontsize=9)
    save(fig, "fig1_klever")

def f2():
    caps = []  
    def collect(records, scoref, task=None, keyfn=lambda r: r["model"]):
        recs = [r for r in records if (task is None or r.get("task") == task)]
        by = {}
        for r in recs: by.setdefault(keyfn(r), {})[int(r["K"])] = scoref(r)
        return [100 * d[2] / d[0] for d in by.values() if 0 in d and 2 in d and d[0] > 0]
    s3 = load("lmeval_stage3")

    caps.append(("Competition math\n(MATH-500, math_verify)", collect(load("math_official"), lambda r: r["metrics"]["math_verify,none"], "minerva_math500")))
    g = collect([r for r in s3 if r["task"] == "gsm8k"], lambda r: r["value"]) +        collect([r for r in load("f5_qwen25inst") if r["model"] not in EXCLUDE_FLEX],
                lambda r: r["metrics"]["exact_match,flexible-extract"])
    caps.append(("Grade-school math\n(GSM8K)", g))
    caps.append(("Knowledge\n(MMLU)", collect(load("mmlu_official"), lambda r: r["metrics"]["acc,none"])))
    caps.append(("Science QA\n(ARC-C)", collect([r for r in s3 if r["task"] == "arc_challenge"], lambda r: r["value"])))
    caps.append(("Instruction\n(IFEval)", collect(load("ifeval_subset"), lambda r: r["prompt_strict"])))
    caps.append(("Commonsense\n(HellaSwag)", collect(load("lmeval_capmap"), lambda r: r["acc"])))
    caps.append(("Code\n(HumanEval)", collect(load("humaneval"), lambda r: r["passat1"])))
    caps = [c for c in caps if c[1]]
    caps.sort(key=lambda c: np.median(c[1]))
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ys = np.arange(len(caps))
    meds = [np.median(c[1]) for c in caps]
    los = [min(c[1]) for c in caps]; his = [max(c[1]) for c in caps]
    colors = plt.cm.RdYlGn(np.array(meds) / 100.0)
    ax.barh(ys, meds, color=colors, edgecolor="0.3")
    ax.errorbar(meds, ys, xerr=[np.subtract(meds, los), np.subtract(his, meds)], fmt="none", ecolor="0.3", capsize=3)
    for y, c in zip(ys, caps):
        ax.text(max(c[1]) + 2, y, f"{min(c[1]):.0f}-{max(c[1]):.0f}% (n={len(c[1])})", va="center", fontsize=8)
    ax.axvline(100, ls=":", color="grey")
    ax.set_yticks(ys); ax.set_yticklabels([c[0] for c in caps], fontsize=8)
    ax.set_xlabel("K2 retention vs FP (%) — bar = median, whisker = range across models")
    ax.set_xlim(0, 125)
    ax.set_title("Capability stratification at K=2 (~3.3 bpw):\nreasoning collapses, knowledge/commonsense/code survive")
    for c in caps: print(f"F2 {c[0].splitlines()[0]}: K2 ret median {np.median(c[1]):.0f}% range {min(c[1]):.0f}-{max(c[1]):.0f}% n={len(c[1])}")
    save(fig, "fig2_capability")

def f3():
    rec = {int(r["K"]): r for r in load("niah_hardened")}
    lens = sorted(int(k) for k in rec[0]["retrieval"])
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    cols = {0: "#2e86de", 2: "#c0392b", 3: "#27ae60"}
    labs = {0: "K=0 (FP)", 2: "K=2 (~3.3 bpw)", 3: "K=3 (~4.9 bpw)"}
    for K in (0, 2, 3):
        ys = [100 * rec[K]["retrieval"][str(L)] for L in lens]
        xs = [L / 1000 for L in lens]
        if K == 3:  
            ax.plot(xs, ys, "s--", color=cols[K], lw=2.2, ms=9, dashes=(4, 3),
                    mfc="white", mec=cols[K], mew=1.8, label=labs[K], zorder=5)
        else:       
            ax.plot(xs, ys, "o-", color=cols[K], lw=(3.0 if K == 0 else 2), ms=6,
                    label=labs[K], zorder=2)
        print(f"F3 K{K} retrieval:", {L: round(100 * rec[K]['retrieval'][str(L)]) for L in lens})
    ax.set_xlabel("context length (k tokens)"); ax.set_ylabel("needle retrieval accuracy (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Long-context retrieval (NIAH, Qwen2.5-7B):\nK2 breaks at 32k; K3 holds")
    ax.legend(fontsize=9)
    save(fig, "fig3_niah")

def f4():

    sq = [r for r in load("mech_sqnr_validation") if int(r["K"]) == 2]
    
    def dB(a): return 10 * np.log10(np.clip(np.array(a, float), 1e-6, None))
    
    G = np.linspace(0, 1, 20)
    def resamp(key):
        return np.array([np.interp(G, np.linspace(0, 1, len(r[key])), dB(r[key])) for r in sq])
    mathg, knowg = resamp("sqnr_math"), resamp("sqnr_know")
    ml = np.array([dB(r["sqnr_math"][-5:]).mean() for r in sq])
    kl = np.array([dB(r["sqnr_know"][-5:]).mean() for r in sq])
    nbelow = int((ml < kl).sum())
    fig, (a, b) = plt.subplots(2, 1, figsize=(6.6, 8.2))  
    a.plot(G, mathg.mean(0), color="#c0392b", lw=2, label="math (GSM8K prompts)")
    a.fill_between(G, mathg.mean(0) - mathg.std(0), mathg.mean(0) + mathg.std(0), color="#c0392b", alpha=0.12)
    a.plot(G, knowg.mean(0), color="#2e86de", lw=2, label="knowledge (MMLU prompts)")
    a.fill_between(G, knowg.mean(0) - knowg.std(0), knowg.mean(0) + knowg.std(0), color="#2e86de", alpha=0.12)
    a.set_xlabel("fractional depth (0=first, 1=last layer)"); a.set_ylabel("activation SQNR at K=2 [dB]")
    a.set_title("(a) SQNR collapses early->mid (amplification\nlives in the layers); math sits lower late")
    a.legend(fontsize=8)
    lim = [min(kl.min(), ml.min()) - 1, max(kl.max(), ml.max()) + 1]
    b.scatter(kl, ml, s=45, color="#8e44ad", zorder=3)
    b.plot(lim, lim, "--", color="grey", label="equal (math = know)")
    b.set_xlim(lim); b.set_ylim(lim)
    b.set_xlabel("late SQNR, knowledge [dB]"); b.set_ylabel("late SQNR, math [dB]")
    b.set_title(f"(b) Math degrades MORE than knowledge\nin late activations: {nbelow}/{len(sq)} below the line")
    b.legend(fontsize=8, loc="upper left")
    print(f"F4 mechanism: n={len(sq)} models, math<know late {nbelow}/{len(sq)}; "
          f"late SQNR math mean {ml.mean():.1f} dB vs know {kl.mean():.1f} dB")
    fig.tight_layout(h_pad=2.0)  
    save(fig, "fig4_sqnr")

def f5():
    av = load("attn_vs_mlp_v3")
    conds = ["baseFP", "mlp_only_K2", "attnK3_mlpK2", "attnK2_mlpK3", "attn_only_K2"]
    clab = {"baseFP": "FP", "mlp_only_K2": "MLP=K2\nattn=FP", "attnK3_mlpK2": "attn=K3\nMLP=K2",
            "attnK2_mlpK3": "attn=K2\nMLP=K3", "attn_only_K2": "attn=K2\nMLP=FP"}
    by = {(r["model"], r["cond"]): r["flex"] for r in av}
    models = sorted({r["model"] for r in av if (r["model"], "baseFP") in
                     {(rr["model"], rr["cond"]) for rr in av}}, key=fam)
    models = [m for m in models if (m, "baseFP") in by]
    fig, (a, b) = plt.subplots(2, 1, figsize=(6.8, 8.4))  
    x = np.arange(len(conds)); w = 0.8 / len(models)
    for i, m in enumerate(models):
        base = by[(m, "baseFP")]
        vals = [(100 * by[(m, c)] / base) if (m, c) in by else np.nan for c in conds]
        a.bar(x + i * w, [0 if v != v else v for v in vals], w, label=fam(m))
        print(f"F5 attn-vs-mlp {fam(m)}:", {clab[c].replace(chr(10), ' '): (round(v) if v == v else None) for c, v in zip(conds, vals)})
    a.set_xticks(x + w * (len(models) - 1) / 2); a.set_xticklabels([clab[c] for c in conds], fontsize=7)
    a.set_ylabel("GSM8K retention vs FP (%)"); a.axhline(100, ls=":", color="grey")
    a.set_title("(a) Attention is the bottleneck\n(MLP-K2 keeps ~90%; attn-K2 collapses)")
    a.legend(fontsize=7, loc="upper right")
    
    mx = [r for r in load("mixedk_dual") if r["task"] == "gsm8k"]
    fs = sorted({r["f"] for r in mx})
    mods = sorted({r["model"] for r in mx}, key=fam)
    for m in mods:
        full = next((r["acc"] for r in mx if r["model"] == m and r["f"] == 1.0 and r["strategy"] == "K3-uniform"), None)
        if not full: continue
        ys = []
        for f in fs:
            if f == 0.0:
                vs = [r["acc"] for r in mx if r["model"] == m and r["f"] == 0.0]
            elif f == 1.0:
                vs = [full]
            else:
                vs = [r["acc"] for r in mx if r["model"] == m and r["f"] == f and r["strategy"] in ("random0", "random1", "SQNR-high", "SQNR-low")]
            ys.append(100 * np.mean(vs) / full if vs else np.nan)
        b.plot(fs, ys, "o-", label=m.split("/")[-1].replace("-Instruct", ""))
        print(f"F5 mixedK {fam(m)}: acc/full-K3 by f", {f: (round(y) if y == y else None) for f, y in zip(fs, ys)})
    b.set_xlabel("fraction of blocks at K=3"); b.set_ylabel("GSM8K acc vs full-K3 (%)")
    b.set_title("(b) Mixed-K is gradual, not all-or-nothing\n(allocation signal not robustly > random)")
    b.legend(fontsize=8)
    fig.tight_layout(h_pad=2.0)  
    save(fig, "fig5_allocation")

def f6():
    rec = {int(r["K"]): r for r in load("paperE_amp")}
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    cols = {2: "#c0392b", 3: "#27ae60"}
    for K in (2, 3):
        q = rec[K]["surprise_quartis"]
        ax.plot([1, 2, 3, 4], q, "o-", color=cols[K], lw=2,
                label=f"K={K} (slope {rec[K]['slope']:+.2f})")
        print(f"F6 K{K} surprise quartiles:", [round(x, 2) for x in q], "slope", rec[K]["slope"])
    ax.set_xticks([1, 2, 3, 4]); ax.set_xticklabels(["Q1\n(early)", "Q2", "Q3", "Q4\n(late)"])
    ax.set_ylabel("FP surprisal at the K-generated token")
    ax.set_title("Divergence is FRONTAL, not cumulative (Qwen2.5-7B):\nK2 bifurcates early; surprise does not grow along the chain")
    ax.legend(fontsize=9)
    save(fig, "fig6_frontal")

def f8():

    

    intj = json.load(open(os.path.join(ROOT, "reports", "data", "paperB_int_kquant_fig8.json")))
    INT = {m: [tuple(p) for p in pts] for m, pts in intj["int_kquant_gsm8k_flexible"].items()}

    

    
    G = 256
    BPW_PACKED = {K: 2 * K + 16 * K / G for K in (2, 3)}
    cs = json.load(open(os.path.join(DATA, "overnight", "clean_sweep_2026-06-30.json")))
    def flex(model, K):
        return cs[f"{model}|K{K}|gsm8k"]["exact_match,flexible-extract"]
    models = list(INT)
    fig, axes = plt.subplots(3, 1, figsize=(6.8, 11.4), sharey=True)  
    for ax, m in zip(axes, models):
        xi, yi = zip(*INT[m])
        ax.plot(xi, yi, "o-", color="#d9544d", label="integer K-quant (llama.cpp, mixed-precision)", ms=8, lw=2)
        xt = [BPW_PACKED[2], BPW_PACKED[3]]
        yt = [flex(m, 2), flex(m, 3)]
        ax.plot(xt, yt, "s-", color="#3a64d9", label="uniform ternary (K2/K3, packed)", ms=8, lw=2)
        fp = flex(m, 0)
        ax.axhline(fp, color="gray", ls="--", lw=1, label="FP baseline")
        ax.annotate("uniform K2\ncliff", xy=(xt[0], yt[0]), xytext=(xt[0] + 0.45, max(yt[0] - 0.22, 0.05)),
                    fontsize=8, color="#3a64d9", arrowprops=dict(arrowstyle="->", color="#3a64d9"))
        ax.set_title(m, fontsize=10)
        ax.set_xlabel("bits per weight (packed storage)")
        ax.set_xlim(2.8, 6.6); ax.set_ylim(-0.03, 1.0); ax.grid(alpha=0.25)
        print(f"F8 {m}: int {INT[m]}  ternary K2 ({xt[0]:.3f},{yt[0]:.3f}) K3 ({xt[1]:.4f},{yt[1]:.3f}) FP {fp:.3f}")
    axes[0].set_ylabel("GSM8K flexible-extract")
    _h, _l = axes[0].get_legend_handles_labels()
    fig.legend(_h, _l, loc="upper center", ncol=3, fontsize=8.5, framealpha=0.95, bbox_to_anchor=(0.5, 0.9))  
    fig.suptitle("Mixed-precision integer K-quants survive at low bpw; uniform ternary K2 collapses.\n"
                 "Same-storage axis: ternary at packed 4.125/6.19 bpw (entropy bound 3.295/4.943 bpw).",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.845], h_pad=1.8)  
    save(fig, "fig8_int_vs_ternary")

for fn in (f1, f2, f3, f4, f5, f6, f8):
    print("=" * 60)
    fn()
print("=" * 60, "\nwrote 7 figures to", [os.path.relpath(d, ROOT) for d in OUTDIRS])
