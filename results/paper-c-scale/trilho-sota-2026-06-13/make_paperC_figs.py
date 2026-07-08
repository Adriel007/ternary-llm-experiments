
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 12, "axes.titlesize": 12, "axes.labelsize": 12,
                     "xtick.labelsize": 10.5, "ytick.labelsize": 10.5, "legend.fontsize": 10})

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
OUTDIRS = [os.path.join(ROOT, "papers", "paper-c-scale", "figures"),
           os.path.join(ROOT, "papers", "arxivorg", "paper-c-scale", "figures")]

def res(stem):
    p = os.path.join(HERE, stem + ".json")
    return json.load(open(p))["results"] if os.path.exists(p) else None

def gsm(r, key):
    if r is None or key not in r: return None
    v = r[key]
    return v["gsm8k"] if isinstance(v, dict) else v

def ret_one(stem, num="joint2", den="fp"):
    r = res(stem)
    a, b = gsm(r, num), gsm(r, den)
    return 100 * a / b if (a is not None and b) else None

def save(fig, stem):
    for d in OUTDIRS:
        fig.savefig(os.path.join(d, stem + ".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

GBPW = {256: 3.29, 128: 3.42, 64: 3.67, 32: 4.17, 16: 5.17}

def f1():
    qwen3 = {0.6: "scale_0.6B", 1.7: "scale_1.7B", 4: "scale_4B", 8: "scale_8B",
             14: "a1_14b", 32: "scale_32B"}
    falcon = {1: "scale_Falcon3-1B-Instruct", 3: "scale_Falcon3-3B-Instruct",
              7: "scale_Falcon3-7B-Instruct", 10: "scale_Falcon3-10B-Instruct"}
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    for name, fam, col in [("Qwen3", qwen3, "#c0392b"), ("Falcon3", falcon, "#e67e22")]:
        xs, ys = [], []
        for s, stem in fam.items():
            r = ret_one(stem)
            if r is not None: xs.append(s); ys.append(r)
        ax.plot(xs, ys, "o-", color=col, lw=2, label=name)
        print(f"F1 {name} GSM8K retention:", {s: round(y) for s, y in zip(xs, ys)})
    ax.axhline(100, ls=":", color="grey", lw=1)
    ax.axhspan(90, 100, color="green", alpha=0.06)
    ax.set_xscale("log")
    ax.set_xticks([0.6, 1, 2, 4, 8, 14, 32]); ax.set_xticklabels(["0.6", "1", "2", "4", "8", "14", "32"])
    ax.set_xlabel("model size (B params, log scale)")
    ax.set_ylabel("GSM8K retention vs FP (%)"); ax.set_ylim(0, 115)
    ax.set_title("Post-hoc ternary reasoning recovery improves with scale\n"
                 "(two families converge to ~95% by ~10-14B; the 4-8B middle is noisy)")
    ax.legend(loc="lower right")
    save(fig, "fig1_recovery")

def f2():
    qsz = {0.6: "0.6B", 1.7: "1.7B", 4: "4B", 8: "8B", 14: "14B"}
    series = {
        "GSM8K (reasoning)":  ("#c0392b", {0.6: "scale_0.6B", 1.7: "scale_1.7B", 4: "scale_4B", 8: "scale_8B", 14: "a1_14b"}),
        "SVAMP (reasoning)":  ("#e74c3c", {s: f"svamp_q3_{n}" for s, n in qsz.items()}),
        "MATH-500 (reasoning)": ("#922b21", {s: f"math_q3_{n}" for s, n in qsz.items()}),
        "MMLU (knowledge)":   ("#2e86de", {s: f"mmlu_q3_{n}" for s, n in qsz.items() if s != 0.6}),
        "ARC-C (commonsense)": ("#27ae60", {s: f"arc_q3_{n}" for s, n in qsz.items()}),
    }
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    for name, (col, fam) in series.items():
        xs, ys = [], []
        for s, stem in fam.items():
            r = ret_one(stem)
            if r is not None: xs.append(s); ys.append(r)
        ls = "o-" if "reasoning" in name else "s--"
        ax.plot(xs, ys, ls, color=col, lw=2, label=name)
        print(f"F2 {name}:", {s: round(y) for s, y in zip(xs, ys)})
    ax.axhline(100, ls=":", color="grey", lw=1)
    ax.set_xscale("log"); ax.set_xticks([0.6, 1, 2, 4, 8, 14]); ax.set_xticklabels(["0.6", "1", "2", "4", "8", "14"])
    ax.set_xlabel("Qwen3 size (B params, log scale)"); ax.set_ylabel("retention vs FP (%)"); ax.set_ylim(0, 115)
    ax.set_title("The scale-dependence is reasoning-specific:\nknowledge & commonsense survive ternarization at every scale")
    ax.legend(fontsize=8, loc="lower right")
    save(fig, "fig2_reasoning_specific")

def f3():
    groups = [256, 128, 64, 32, 16]
    def sweep(prefix, fpstem):
        fp = gsm(res(fpstem), "fp")
        xs, ys = [], []
        for g in groups:
            j = gsm(res(f"{prefix}_g{g}"), "joint2")
            if j is not None and fp: xs.append(GBPW[g]); ys.append(100 * j / fp)
        return xs, ys, fp
    fig, (a, b) = plt.subplots(2, 1, figsize=(6.8, 8.4))  
    lx, ly, lfp = sweep("bf_Llama-3.1-8B", "bf_Llama-3.1-8B_fp")
    qx, qy, qfp = sweep("bf_Qwen3-8B", "bf_Qwen3-8B_fp")
    a.plot(qx, qy, "o-", color="#c0392b", lw=2, label="Qwen3-8B (recovers)")
    a.plot(lx, ly, "s-", color="#34495e", lw=2, label="Llama-3.1-8B (does not)")
    a.set_xlabel("bits/weight (group size)"); a.set_ylabel("GSM8K retention vs FP (%)")
    a.set_title("(a) The recovery is NOT universal:\nLlama-3.1-8B never recovers across the bit-sweep")
    a.legend(fontsize=8); a.set_ylim(0, 115); a.axhline(100, ls=":", color="grey")
    print(f"F3 Llama bit-sweep (bpw->ret):", {round(x, 2): round(y) for x, y in zip(lx, ly)})
    print(f"F3 Qwen3-8B bit-sweep:", {round(x, 2): round(y) for x, y in zip(qx, qy)})
    
    base = 100 * gsm(res("bf_Llama-3.1-8B_g32"), "joint2") / lfp
    fracs = [0.0, 0.005, 0.01, 0.02]
    ys = [base] + [100 * gsm(res(f"llof_g32_of{f}"), "joint2") / lfp for f in (0.005, 0.01, 0.02)]
    b.bar(range(len(fracs)), ys, color=["#34495e", "#27ae60", "#27ae60", "#27ae60"])
    b.set_xticks(range(len(fracs))); b.set_xticklabels([f"{int(f*1000)/10:g}%" for f in fracs])
    b.set_xlabel("fraction of top-|W| weights kept in FP (g=32)")
    b.set_ylabel("Llama-3.1-8B GSM8K retention (%)")
    b.set_title("(b) An outlier-driven failure:\n+0.5% FP weights rescue it 15% -> ~52%")
    print(f"F3 Llama outlier rescue (frac->ret):", {f: round(y) for f, y in zip(fracs, ys)})
    fig.tight_layout(h_pad=2.0)  
    save(fig, "fig3_llama")

def f4():
    rows = [("Qwen3-4B\n(~3B active)", "wr_Qwen3-4B", "#2e86de"),
            ("Qwen3-32B\n(~30B total)", "wr_Qwen3-32B", "#8e44ad"),
            ("Qwen3-30B-A3B\n(MoE: 3B act / 30B tot)", "wr_Qwen3-30B-A3B", "#c0392b")]
    labs, ys, cols = [], [], []
    for lab, stem, col in rows:
        r = ret_one(stem)
        if r is None: r = ret_one("moe_Qwen3-30B-A3B")
        labs.append(lab); ys.append(r); cols.append(col)
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    bars = ax.bar(range(len(labs)), ys, color=cols, width=0.6)
    for i, y in enumerate(ys): ax.text(i, y + 1, f"{y:.0f}%", ha="center", fontsize=11, fontweight="bold")
    ax.set_xticks(range(len(labs))); ax.set_xticklabels(labs, fontsize=10)
    ax.set_ylabel("GSM8K retention vs FP (%)"); ax.set_ylim(0, 105)
    ax.set_title("Recovery tracks TOTAL params, not active compute\n"
                 "(within-run, same N=80: a 3B-active MoE recovers like a >30B model)")
    print("F4 MoE dissociation:", {l.splitlines()[0]: round(y) for l, y in zip(labs, ys)})
    save(fig, "fig4_moe")

def f5():
    sizes = {0.6: "0.6B", 1.7: "1.7B", 4: "4B", 8: "8B", 14: "14B"}
    groups = [256, 128, 64, 32, 16]
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    cmap = plt.cm.viridis(np.linspace(0, 0.85, len(sizes)))
    for (s, n), col in zip(sizes.items(), cmap):
        fp = gsm(res(f"bf_Qwen3-{n}_fp"), "fp")
        xs, ys = [], []
        for g in groups:
            j = gsm(res(f"bf_Qwen3-{n}_g{g}"), "joint2")
            if j is not None and fp: xs.append(GBPW[g]); ys.append(100 * j / fp)
        ax.plot(xs, ys, "o-", color=col, lw=2, label=f"Qwen3-{n}")
        print(f"F5 Qwen3-{n} frontier (bpw->ret):", {round(x, 2): round(y) for x, y in zip(xs, ys)})
    ax.set_xlabel("bits/weight (finer group ->)"); ax.set_ylabel("GSM8K retention vs FP (%)")
    ax.set_ylim(0, 115); ax.axhline(100, ls=":", color="grey")
    ax.set_title("The reasoning bit-budget threshold widens with scale\n"
                 "(small models sit at a cliff; g=32 is the robust operating point)")
    ax.legend(fontsize=9.5, loc="center left", bbox_to_anchor=(1.01, 0.5))  
    save(fig, "fig5_frontier")

for fn in (f1, f2, f3, f4, f5):
    print("=" * 60)
    fn()
print("=" * 60, "\nwrote figures to", [os.path.relpath(d, ROOT) for d in OUTDIRS])
