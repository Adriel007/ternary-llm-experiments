
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))

gsm = {}
for line in open(os.path.join(HERE, "gsm_q25.jsonl")):
    r = json.loads(line)
    if "flexible_acc" in r:
        gsm[r["quant"]] = r
def acc(q):
    base = gsm.get(q + "-full") or gsm.get(q)
    return base["flexible_acc"], base.get("flexible_ci", [None, None]), base["n"]

cb = json.load(open(os.path.join(HERE, "cpu_bench_qwen2.5-7b.json")))
size = {r["quant"]: r["size_bytes"] / 1e9 for r in cb["results"]}
ppl = {r["quant"]: r["ppl_wikitext"] for r in cb["results"]}

ORDER = ["F16", "Q8_0", "Q4_K_M", "Q2_K", "sasori-TQ3P", "sasori-TQ2P"]
def col(q): return "#333" if q == "F16" else ("#c0392b" if q.startswith("sasori") else "#2e86de")
def mk(q):  return "D" if q.startswith("sasori") else ("*" if q == "F16" else "o")

OFF_A = {"F16": (8, 6), "Q8_0": (6, 12), "Q4_K_M": (-2, 14), "Q2_K": (8, -4),
         "sasori-TQ3P": (4, -16), "sasori-TQ2P": (10, 2)}

fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.8))

for q in ORDER:
    ac, ci, n = acc(q)
    gb = size[q]
    yerr = [[ac - ci[0]], [ci[1] - ac]] if ci[0] is not None else None
    a.errorbar(gb, ac, yerr=yerr, fmt=mk(q), ms=13, c=col(q), ecolor=col(q),
               capsize=3, mec="k", zorder=3)
    dx, dy = OFF_A[q]
    a.annotate(f"{q}\n{ac:.3f}", (gb, ac), fontsize=8, fontweight="bold",
               xytext=(dx, dy), textcoords="offset points", va="center",
               color=col(q))
fF = acc("F16")[0]
a.axhline(fF, ls=":", color="grey", lw=1)
a.text(13.4, fF + 0.015, "F16", fontsize=8, color="grey")
a.set_ylim(0.0, 1.0)
a.set_xlabel("size on disk / RAM (GB)   $\\leftarrow$ smaller better")
a.set_ylabel("GSM8K accuracy (flexible-extract)   $\\uparrow$")
a.set_title("(a) The TASK vs size, Qwen2.5-7B-Instruct\n(greedy, 5-shot; full-set n=1319 for F16/Q4\\_K\\_M)")

for q in ORDER:
    ac, _, _ = acc(q)
    b.scatter(ppl[q], ac, s=170, c=col(q), marker=mk(q), edgecolor="k",
              zorder=3, label=f"{q} (PPL {ppl[q]:.1f}, acc {ac:.3f})")

for q in ["sasori-TQ2P", "Q2_K"]:
    ac, _, _ = acc(q)
    b.annotate(q, (ppl[q], ac), fontsize=8, fontweight="bold", color=col(q),
               xytext=(-6, 12), textcoords="offset points", ha="right")
b.annotate("cluster: F16 / Q8\\_0 / Q4\\_K\\_M / sasori-TQ3P\n"
           "all $\\approx$0.87--0.90 across PPL 6.7--7.8\n(PPL order $\\neq$ task order)",
           (7.0, 0.885), fontsize=7.5, xytext=(0, -64), textcoords="offset points",
           ha="center", arrowprops=dict(arrowstyle="->", color="grey", lw=0.8))
b.set_xscale("log"); b.set_ylim(0.0, 1.0)
b.set_xlabel("wikitext PPL (log)   $\\leftarrow$ lower 'better' (but PPL $\\neq$ task)")
b.set_ylabel("GSM8K accuracy (flexible-extract)   $\\uparrow$")
b.set_title("(b) PPL $\\neq$ task: K=2 (TQ2P) PPL only 3.3$\\times$ F16,\nyet GSM8K collapses 0.87$\\to$0.13; K=3 (TQ3P) holds")
b.legend(fontsize=6.6, loc="lower left", framealpha=0.9)

t3, q4 = acc("sasori-TQ3P")[0], acc("Q4_K_M")[0]
t2 = acc("sasori-TQ2P")[0]
fig.suptitle(
    "Deployed K-trit-plane ternary (sasori) on GSM8K --- the fair task metric.  "
    f"K=3: {t3:.3f} (ties Q4\\_K\\_M {q4:.3f}, near F16) despite higher PPL;  "
    f"K=2: {t2:.3f} (collapse).  Reasoning needs K$\\geq$3; K-quants shown as an efficiency reference.",
    fontsize=8.2)
plt.tight_layout(rect=[0, 0, 1, 0.94])
out = os.path.join(HERE, "gsm8k_qwen2.5-7b.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print("wrote", out)
for q in ORDER:
    ac, ci, n = acc(q)
    print(f"  {q:14s} acc {ac:.3f}  ci {ci}  n={n}  ppl {ppl[q]}")
