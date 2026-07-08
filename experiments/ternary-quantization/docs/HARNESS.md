# Evaluation harness

## Official: lm-evaluation-harness (EleutherAI)
The scrutiny-grade accuracy numbers use the official harness via `lm_eval.simple_evaluate`, model
served as a live `HFLM(pretrained=<quantized bf16 model>)`. `bootstrap_iters=0` (point estimates).

Tasks and metrics (as reported):
- **gsm8k** — 5-shot, `exact_match,flexible-extract` (and `strict-match`). 8-shot CoT used in the early
  fase1 full-N runs; lm-eval default 5-shot for the official tables.
- **gsm_plus_mini** — 5-shot, flexible-extract (adversarial GSM variant).
- **arc_challenge** — 0-shot, `acc_norm` (length-normalized).
- **hellaswag** — 0-shot, `acc_norm`.
- **mmlu** — `acc,none` (subject sweep; `limit=40` per subject in the official table).
- **minerva_math500 / hendrycks_math500** — MATH-500, `math_verify` (sympy-checked) and `exact_match`.
- **humaneval** — pass@1, executed (code run in a sandbox).
- Coexistence: lm-eval 0.4.12 + transformers 5.10.2 run in one venv (no multi-venv needed).

### MATH-500 dependency isolation
`minerva_math500` imports `math_verify` → `latex2sympy2_extended`, generated for **antlr 4.11.0**,
which conflicts with the antlr 4.13 pulled by hydra/omegaconf (`TypeError: ord() ...` at task build).
Fix (no venv; Colab `python -m venv` lacks ensurepip): install antlr-4.11 into a side directory and
prepend it to `PYTHONPATH` for the eval subprocess:
```
pip install --target /content/mathlibs --upgrade antlr4-python3-runtime==4.11.0 math-verify latex2sympy2-extended sympy
PYTHONPATH=/content/mathlibs:$PYTHONPATH   # 4.11 wins
```
`hendrycks_math500` "runs" but its `\boxed{}` parser does not extract Instruct-format answers
(baseline ≈ 0) — use `minerva_math500` for MATH-500.

## Custom-harness probes (ratio/ordering valid, absolute not lm-eval-calibrated)
Some capability probes use own evaluators (deterministic):
- **HellaSwag/SVAMP** (`hellaswag.jsonl`, `svamp.jsonl`) — own multiple-choice logprob scorer, N=200.
- **IFEval-subset** (`ifeval_subset.jsonl`) — 260 real IFEval prompts + own deterministic instruction checkers.
- **NIAH** (`niah_*.jsonl`) — needle-in-a-haystack retrieval, Wilson confidence intervals.
- **Capability-map / bits_per_capability** — own MC evals (`CAPABILITY_MAP.md` documents the caveat).
The official IFEval run (`fase2__ifeval_oficial`, numbers in the project diario) validated these checkers:
they capture direction, not absolute (own checker overestimates vs lm-eval).

## Compute
Runs on Colab (A100 / RTX-PRO-6000-Blackwell-96GB) and RunPod (A40/A6000/H100). Quantization is
deterministic; the GPU only affects wall-clock, not the numbers.
