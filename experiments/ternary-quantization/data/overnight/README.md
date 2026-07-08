# data/overnight — raw logs from the 2026-07 overnight/RunPod rounds (additive)

Files copied verbatim from `results/runpod-runs/` and `sasori/bench/` (the raw sources of
record). Nothing in the parent `data/` directory was modified.

| file | source | what it backs in Paper B |
|---|---|---|
| `moe_qwen3_30b_k2.jsonl` | `results/runpod-runs/phd-r2/` | **CANONICAL** Qwen3-30B-A3B MoE run (single machinery, same GSM8K slice, N=150): FP 0.8933, K2-experts 0.68, K3-experts 0.86, +router-K2 0.5933. §RQ1 MoE paragraph. |
| `moe_qwen3_30b_routerk2.jsonl` | `results/runpod-runs/RK2/` | router-ternarized control (all 48 routers): 0.5933 (subset of the row above, kept as its own log). |
| `moe_olmoe_k2.jsonl` | `results/runpod-runs/MOE2/` | OLMoE-1B-7B-Instruct: FP 0.36, K2 0.02, K3 0.36 (N=150). |
| `moe_dsv2lite_k2.jsonl` | `results/runpod-runs/MOE3/` | DeepSeek-V2-Lite-Chat: FP 0.4867, K2 0.02, K3 0.34 (N=150). |
| `rq93_capability_vs_robustness.json` | `results/runpod-runs/RQ93/` | #93 capability-vs-robustness analysis (14 models dense+MoE): capability floor ≈0.69 is necessary-not-sufficient; counterexamples inline. §RQ1. |
| `hh_datafree.result.json` | `results/runpod-runs/phd-hhdf/` | data-free per-bit head-to-head (Qwen2.5-1.5B, N=200, g=128): RTN-int3 0.005 / int4 0.395 vs ternary K2 0.02 / K3 0.505 / K4 0.595. Both bpw conventions recorded in `_meta`. §RQ1. |
| `hh_gptq.result.json` | `results/runpod-runs/phd-hhgptq/` | data-aware GPTQ head-to-head (Qwen2.5-1.5B, N=150, c4 calib): int4 0.5133 @ 4.156 bpw, int8 0.62 @ 8.19 bpw. Per-bit honesty sentence (#65). |
| `mixed_k_distill_lo1.json` | `results/runpod-runs/MK/` | mixed-K (14×K3 early + 14×K1 late, 4.125 bpw) + LoRA-KD: PTQ 0.005 → distill 0.05 vs uniform K3-PTQ 0.515. §RQ4. |
| `i76_{gen,coder,q3_4b,q3_8b}.result.json` | `results/runpod-runs/R76/` | #76 anatomy of K2-robust models: SQNR-late (dB) Qwen3-4B 5.65 vs Qwen3-8B 2.67 (separates), Coder-7B 8.86 ≈ general-7B 8.41 (does NOT separate). Triage-counterexample sentence, §RQ3. |
| `clean_sweep_2026-06-30.json` | `sasori/bench/` | clean lm-eval sweep (official harness, fake-quant g=256): FP/K2/K3 GSM8K-flex, ARC, HellaSwag for Qwen2.5-7B-Inst, Coder-7B-Inst, Qwen3-4B, Qwen3-8B, Qwen3-30B-A3B. Ternary points of Fig. 8. |
| `ifeval_official_2026-06-30.json` | `sasori/bench/` | official lm-eval `ifeval` (limit=100): prompt/instruction-level strict+loose for Qwen2.5-7B-Inst and Qwen3-8B at K0/K2/K3. IFEval footnote, §RQ2. |
| `distill_moe_fiel_2026-07-01.json` | `sasori/bench/` | Qwen1.5-MoE-A2.7B-Chat faithful run: FP 0.64, K2-PTQ 0.007. §RQ1 MoE paragraph. |

**Which Qwen3-30B run is canonical?** The parent `data/moe_qwen3_30b.jsonl` is an EARLIER,
partial run (FP 0.8467, K3 0.8867, **no K2 arm**, different quantization coverage counters).
It is kept untouched for provenance, but the numbers quoted in the paper (0.8933 / 0.68 /
0.86 / 0.5933) come from `overnight/moe_qwen3_30b_k2.jsonl` — the single-machinery run that
holds the GSM8K slice fixed across K ∈ {0,2,3} and the router control. Use the overnight run.

Integer llama.cpp K-quant points in Fig. 8 (Q2_K etc., GSM8K flexible, lim300) were measured
on pods and are recorded inline (with provenance comment) in
`analysis/make_paperB_figs.py` and `sasori/bench/plot_pareto_int_vs_ternary.py`.
