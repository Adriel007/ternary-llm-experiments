
import json, os, csv, collections

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "results", "retention_official.csv")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

def load(fn):
    p = os.path.join(DATA, fn)
    return [json.loads(l) for l in open(p) if l.strip()] if os.path.exists(p) else []

rows = []  

for fn in ("lmeval_stage3.jsonl", "lmeval_official.jsonl"):
    for r in load(fn):
        metric = r.get("metric", "exact_match,flexible-extract" if r["task"] == "gsm8k" else "acc_norm")
        rows.append((fn[:-6], r["model"], r["task"], metric, r["K"], r["value"]))

for r in load("mmlu_official.jsonl"):
    rows.append(("mmlu_official", r["model"], "mmlu", "acc,none", r["K"], r["metrics"]["acc,none"]))

for fn in ("math_official.jsonl", "math500_fulln.jsonl"):
    for r in load(fn):
        m = r["metrics"]
        if "math_verify,none" in m:
            rows.append((fn[:-6], r["model"], r["task"], "math_verify", r["K"], m["math_verify,none"]))
        elif "exact_match,flexible-extract" in m:
            rows.append((fn[:-6], r["model"], r["task"], "flexible-extract", r["K"], m["exact_match,flexible-extract"]))
        elif "exact_match,none" in m:
            rows.append((fn[:-6], r["model"], r["task"], "exact_match", r["K"], m["exact_match,none"]))

for r in load("humaneval.jsonl"):
    rows.append(("humaneval", r["model"], "humaneval", "pass@1", r["K"], r["passat1"]))

for r in load("ifeval_subset.jsonl"):
    rows.append(("ifeval_subset", r["model"], "ifeval", "prompt_strict", r["K"], r["prompt_strict"]))

for r in load("overnight/moe_qwen3_30b_k2.jsonl"):
    rows.append(("moe_qwen3_30b", r["model"], "gsm8k", "exact_match,flexible-extract", r["K"], r["flex"]))

base = {}
for exp, model, task, metric, K, score in rows:
    if K == 0:
        base[(exp, model, task, metric)] = score
out = []
for exp, model, task, metric, K, score in rows:
    b = base.get((exp, model, task, metric))
    ret = round(100.0 * score / b, 1) if b not in (None, 0) else ""
    out.append([exp, model, task, metric, K, round(score, 4), round(b, 4) if b is not None else "", ret])

out.sort(key=lambda x: (x[0], x[1], x[2], x[4]))
with open(OUT, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["experiment", "model", "task", "metric", "K", "score", "baseline_K0", "retention_pct"])
    w.writerows(out)
print(f"wrote {OUT}: {len(out)} rows from {len({r[0] for r in out})} experiments")
