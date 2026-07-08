
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
d = json.load(open(os.path.join(HERE, "cpu_bench_qwen3-1.7b.json")))
R = d["results"]
def col(q):
    if q == "F16": return "#333333"
    return "#c0392b" if q.startswith("sasori") else "#2e86de"

fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.4))

for r in R:
    gb = r["size_bytes"] / 1e9
    mk = "D" if r["quant"].startswith("sasori") else ("*" if r["quant"] == "F16" else "o")
    a.scatter(gb, r["ppl_wikitext"], s=130, c=col(r["quant"]), marker=mk, edgecolor="k", zorder=3)
    a.annotate(f"{r['quant']}\n{r['decode_tok_s']:.0f} tok/s",
               (gb, r["ppl_wikitext"]), fontsize=7.5, xytext=(6, 0), textcoords="offset points", va="center")
a.set_yscale("log")
a.axhspan(15, 30, color="green", alpha=0.06)
a.text(0.5, 27, "near-FP quality", fontsize=8, color="green")
a.set_xlabel("model size on disk / in RAM (GB)  ← smaller is better")
a.set_ylabel("wikitext perplexity (log)  ↓ lower is better")
a.set_xlim(0.5, 4.4)
a.set_title("(a) Quality × memory (Qwen3-1.7B, CPU)\nsasori = ◆  | llama.cpp quants = ● | F16 = ★")

for r in R:
    gb = r["size_bytes"] / 1e9
    mk = "D" if r["quant"].startswith("sasori") else ("*" if r["quant"] == "F16" else "o")
    b.scatter(gb, r["decode_tok_s"], s=130, c=col(r["quant"]), marker=mk, edgecolor="k", zorder=3)
    b.annotate(r["quant"], (gb, r["decode_tok_s"]), fontsize=7.5, xytext=(6, 0), textcoords="offset points", va="center")
b.set_xlabel("model size (GB)")
b.set_ylabel("decode tok/s (↑ faster)")
b.set_title("(b) Speed is memory-bound (decode ∝ 1/size)\nsasori tracks the size–speed line, no magic")
b.set_xlim(0.5, 4.4)

fig.suptitle("sasori is a QUALITY-first ternary CPU kernel — honest positioning vs llama.cpp quants\n"
             "(1.7B = sasori's worst case: FP embeddings inflate the file; PPL ≠ task accuracy)",
             fontsize=10)
plt.tight_layout(rect=[0, 0, 1, 0.93])
out = os.path.join(HERE, "cpu_bench_qwen3-1.7b.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print("wrote", out)
for r in R:
    print(f"  {r['quant']:13s} {r['size_bytes']/1e9:.2f}GB  decode {r['decode_tok_s']:5.1f} tok/s  PPL {r['ppl_wikitext']:.1f}")
