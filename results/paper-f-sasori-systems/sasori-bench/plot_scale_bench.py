
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
d17 = json.load(open(os.path.join(HERE, "cpu_bench_qwen3-1.7b.json")))
d7 = json.load(open(os.path.join(HERE, "cpu_bench_qwen2.5-7b.json")))
def col(q): return "#333" if q == "F16" else ("#c0392b" if q.startswith("sasori") else "#2e86de")
def mk(q): return "D" if q.startswith("sasori") else ("*" if q == "F16" else "o")

fig, (a, b) = plt.subplots(1, 2, figsize=(11.4, 4.5))

for r in d7["results"]:
    gb = r["size_bytes"] / 1e9
    a.scatter(gb, r["ppl_wikitext"], s=140, c=col(r["quant"]), marker=mk(r["quant"]), edgecolor="k", zorder=3)
    a.annotate(f"{r['quant']}\n{r['decode_tok_s']:.1f} tok/s", (gb, r["ppl_wikitext"]),
               fontsize=7.5, xytext=(6, 0), textcoords="offset points", va="center")
a.set_yscale("log")
a.annotate("Q4_K_M dominates sasori-TQ3P\n(smaller + lower PPL + faster)", (4.68, 7.12),
           fontsize=7.5, color="#2e86de", xytext=(10, -38), textcoords="offset points",
           arrowprops=dict(arrowstyle="->", color="#2e86de"))
a.set_xlabel("size on disk / RAM (GB)  ← smaller better")
a.set_ylabel("wikitext PPL (log)  ↓ lower better")
a.set_title("(a) Qwen2.5-7B, CPU — the FAIR case, still honest:\non perplexity-per-bit the K-quants dominate sasori")

def ratio(d, q):
    R = {r["quant"]: r["ppl_wikitext"] for r in d["results"]}
    return R[q] / R["F16"]
sizes = [1.7, 7]
for q, c in [("sasori-TQ2P", "#c0392b"), ("sasori-TQ3P", "#e67e22")]:
    ys = [ratio(d17, q), ratio(d7, q)]
    b.plot(sizes, ys, "o-", color=c, lw=2, markersize=9, label=q)
    for x, y in zip(sizes, ys):
        b.annotate(f"{y:.1f}x", (x, y), fontsize=9, xytext=(6, 4), textcoords="offset points")
b.axhline(1.0, ls=":", color="grey"); b.text(4, 1.15, "F16 (1.0x)", fontsize=8, color="grey")
b.set_xticks([1.7, 7]); b.set_xticklabels(["1.7B", "7B"])
b.set_xlabel("model size"); b.set_ylabel("PPL relative to F16  ↓")
b.set_yscale("log")
b.set_title("(b) The genuine sasori finding: quality RECOVERS with scale\n"
            "TQ2P 10.5x→3.3x (cross-validates Paper C); TQ3P stays near-FP")
b.legend(fontsize=8)

fig.suptitle("sasori CPU deployment positioning — honest: NO efficiency/quality-per-bit win vs llama.cpp K-quants\n"
             "(value is research + a quality ceiling; PPL ≠ task accuracy, where sasori-K3 retains ~98% GSM8K per Papers B/C)",
             fontsize=9.5)
plt.tight_layout(rect=[0, 0, 1, 0.92])
out = os.path.join(HERE, "cpu_bench_7b_and_scale.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print("wrote", out)
for d, n in [(d17, "1.7B"), (d7, "7B")]:
    print(f"  [{n}] TQ2P {ratio(d,'sasori-TQ2P'):.1f}x  TQ3P {ratio(d,'sasori-TQ3P'):.2f}x  F16-PPL {next(r['ppl_wikitext'] for r in d['results'] if r['quant']=='F16')}")
