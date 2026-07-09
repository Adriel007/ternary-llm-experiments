# Ternary (K-trit-plane) PTQ — benchmark study (reproducibility package)

Post-training quantization of LLM weights to **ternary values** {-1,0,+1} via **K stacked
trit-planes** (`W ≈ Σ_{k=1..K} diag(α^k) T^k`, each plane a ternary residual; K2 ≈ 3.3 bits/weight,
K3 ≈ 4.9 bpw, K4 ≈ 6.6 bpw). The study measures how much task accuracy survives quantization,
across capabilities, model families, and scales. Ternary weights are dequantized to bf16 for
evaluation (fake-quant), so the numbers isolate the *representational* cost of ternary weights,
not a custom kernel's speed.

All accuracy numbers here are produced with the **official lm-eval-harness** (see `docs/HARNESS.md`)
unless a file is explicitly a custom-harness probe (noted below). Quantization is **deterministic**
(bit-identical across repeats), so "seeds" are N/A for the quantization step.

## Layout
- `data/` — raw per-experiment result logs: **33 JSONL** files (one JSON object per (model, K, task[, ...])) + `CAPABILITY_MAP.md`.
- `results/retention_official.csv` — consolidated long-format table (experiment, model, task, metric,
  K, score, baseline_K0, retention_pct) built from the official lm-eval files by `build_results_table.py`.
- `docs/HARNESS.md`, `docs/MODELS.md`, `docs/CONFIGS.md` — evaluation harness, model list, quantization config.
- `build_results_table.py` — regenerates the consolidated CSV from `data/`.

## Headline results (retention vs each model's own FP16 baseline)
Built from `results/retention_official.csv`. K3 ≈ 4.9 bpw is the universal "lossless-ish" setting; K2 collapses on reasoning.

| Capability (benchmark, harness)            | K3 retention | K2 retention | source file |
|--------------------------------------------|--------------|--------------|-------------|
| Math reasoning (GSM8K flex, 5 families)    | 81–99 %      | 1–18 %       | `lmeval_stage3.jsonl` (+`lmeval_gsm8k`) |
| Math, MoE 30B (GSM8K flex, experts ternarized) | 96 %         | 76 %         | `overnight/moe_qwen3_30b_k2.jsonl` |
| Competition math (MATH-500, minerva)       | 78–105 %; K5→95 % | ~0–3 % | `math500_fulln.jsonl`, `math_official.jsonl` |
| Knowledge (MMLU, 5 families / 6 models)    | 96–100 %     | 38–72 %      | `mmlu_official.jsonl` |
| Knowledge (ARC-Challenge acc_norm)         | 96–101 %     | 44–79 %      | `lmeval_stage3.jsonl` |
| Commonsense (HellaSwag acc_norm)           | 97–99 %      | 50–77 %      | `lmeval_capmap.jsonl`, `hellaswag.jsonl` |
| Code (HumanEval pass@1, Coder 7B/14B)      | 98–100 %     | 82–94 %      | `humaneval.jsonl` |
| Instruction-following (IFEval prompt-strict)| 96 %        | 54–85 %      | `ifeval_subset.jsonl` |
| Long-context retrieval (NIAH, to 32k)      | 100 %        | degrades (32k→33 %) | `niah_hardened.jsonl`, `niah_longctx.jsonl` |

**Central finding:** the K2→K3 jump rescues *reasoning* specifically; knowledge/commonsense/code
degrade far less under K2. K3 (~4.9 bpw) is the universal recommendation. Competition math is the
hardest to recover and benefits from K4/K5.

## data/ file manifest
**Official lm-eval (scrutiny-grade):**
- `lmeval_stage3.jsonl` — GSM8K(flex,5sh) + ARC-C(acc_norm) K0/K2/K3, 4 families (Qwen3-8B/Granite-8B/Phi-3.5/Mistral-7B); the Qwen2.5 GSM8K rows live in `lmeval_gsm8k`/`f5_qwen25inst`.
- `lmeval_official.jsonl` — ARC-C + HellaSwag acc_norm, Qwen2.5-1.5B/7B (reconstructed-from-print).
- `lmeval_gsm8k.jsonl` — GSM8K strict+flex, Qwen2.5-1.5B/7B.
- `mmlu_official.jsonl` — MMLU acc, 5 families / 6 models, K0/K2/K3 (limit=40 subjects).
- `math_official.jsonl` — minerva_math500 + gsm_plus_mini + hendrycks_math500, Qwen2.5-7B/Qwen3-8B, K0/K2/K3.
- `math500_fulln.jsonl` — MATH-500 full N=500, Qwen2.5-7B/Qwen3-8B, K0/K3/K4/K5 (the K-ladder for hard math).
- `k3_vs_int4.jsonl` — GSM8K K0/K3 vs NF4-4bit, three models (head-to-head with 4-bit).
- `int4_baseline.jsonl` — NF4-4bit GSM8K baselines (bpw 4.13), three models.
- `hellaswag.jsonl` / `svamp.jsonl` — custom-harness HellaSwag / SVAMP K-sweeps (N=200; ordering-valid, abs not lm-eval-calibrated).
- `lmeval_capmap.jsonl` — HellaSwag acc K0/K2/K3, 4 families (official, limit=2000).
- `humaneval.jsonl` — HumanEval pass@1 (executed), Qwen2.5-Coder 7B/14B, K0–K4.
- `ifeval_subset.jsonl` — IFEval-subset prompt/instr strict, Qwen2.5-7B + Coder-7B (own deterministic checkers).
- `niah_longctx.jsonl` / `niah_hardened.jsonl` — needle-in-a-haystack retrieval to 16k / 32k (Wilson CIs).

**MoE:**
- `moe_qwen3_30b.jsonl` — Qwen3-30B-A3B, K0_fp vs K3 with all 12288 experts genuinely ternarized (router FP). GSM8K flex.
- `moe_router2.jsonl` — Qwen1.5-MoE-A2.7B, experts ternarized, router FP vs ternary (the corrected MoE run).

**Scale / training-controlled:**
- `f5_pythia.jsonl` — Pythia 160m→6.9b (same corpus, size varies), lambada/hella/arc/piqa K0/K2/K3 — does recovery scale with size under controlled training? (Spearman(size, K3-retention)=+1.0 on lambada.)
- `f5_qwen25inst.jsonl` — Qwen2.5-Instruct 0.5B–14B GSM8K K0/K2/K3 (instruct-format caveat documented).
- `k5_above_baseline.jsonl` — McNemar paired test, Qwen2.5-1.5B ARC-C N=1172, K3–K6 vs FP (quant never beats FP).
- `bits_per_capability_q3_8b.jsonl` — Qwen3-8B K1/K4 across gsm8k/arc/hella (K1 collapses; K4 ≈ baseline).

**Allocation / mechanism (predictor study, secondary):**
- `attn_vs_mlp_v3.jsonl` — selective K2 on attention-only vs MLP-only vs attnK3+mlpK2, 4 families (attention = bottleneck).
- `mixedk_dual.jsonl` — mixed-K fraction sweep f∈{0,.25,.5,.75,1}, GSM8K+ARC, 1.5B/7B/8B (SQNR-guided vs random alloc).
- `mixedk_arc.jsonl` / `mixedk_sqnr.jsonl` — half-K3 allocation by SQNR vs random, ARC (positive) / GSM8K (≈random).
- `sqnr_kladder.jsonl` — late-activation SQNR(K=2..5), Qwen2.5-7B + Qwen3-8B (the R(K)-law concavity ladder).
- `mech_sqnr_validation.jsonl` — per-layer SQNR curves (math/knowledge) K2/K3, 12 models (predictor-validation, n=24).
- `mech_pilot_kl.jsonl` / `mech_pilot_freerun.jsonl` / `mech_pilot_sqnr.jsonl` — eliminated predictor candidates (logit-KL teacher-forced / free-run; SQNR pilot).
- `from_quant.jsonl` — compose ternary on top of FP8/INT8 quantized weights (error composition; #35).
- `paperE_amp.jsonl` — surprise-by-CoT-quartile, divergence is frontal not cumulative (Paper E).
- `e3_rank.jsonl` — 2nd-plane reconstruction gain vs activation energy (Spearman ≈ 0 → joint is not implicitly activation-aware).
- `CAPABILITY_MAP.md` — capability×K reasoning-specificity summary (custom MC evals, ordering-valid only).

## Caveats (read before quoting a number)
- **Retention is within-model** (score(K) / score(K=0) for the *same* model); it is not an absolute SOTA claim.
- Files tagged "custom harness" (hellaswag/svamp/ifeval/niah/capability-map/bits_per_capability) use own evals;
  trust the **ratio/ordering**, not the absolute, where the baseline differs from lm-eval.
- MATH-500 retention at low baseline is noisy; the K-ladder (`math500_fulln`) is the reliable read.
- `lmeval_official`/`lmeval_gsm8k` rows marked `reconstructed_from_print` were rebuilt from console logs after a
  runtime reset (numbers re-verified against re-runs); the stage3 file is the primary cross-family record.
- The earlier interpretability/PoC experiment data (BitNet circuits, SAEs, H2) lives in the repo at `reports/data/`,
  not here — this package is the ternary-quantization benchmark study only.
