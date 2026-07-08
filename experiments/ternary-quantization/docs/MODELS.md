# Models evaluated

All models are public open-weight checkpoints, loaded in bf16 (the native dtype) as the FP baseline.

**Paper B population: 22 distinct checkpoints — 18 dense (0.5–14B, seven families, base and
instruct variants) + 4 mixtures-of-experts (7–30B total parameters).**

| Family | Checkpoints | Where used |
|---|---|---|
| Qwen2.5 (dense, Instruct) | 0.5B, 1.5B, 3B, 7B, 14B† | core retention grid (`f5_qwen25inst`, `mmlu_official`, `math_official`, `niah_*`, `mech_*`, `attn_vs_mlp_v3`, …) |
| Qwen2.5 (dense, base) | 1.5B, 3B | per-bit head-to-head + mixed-K+distill (`overnight/hh_*`, `overnight/mixed_k_distill_lo1`) and the capability-threshold grid (`overnight/rq93_*`) |
| Qwen2.5-Coder (Instruct) | 7B, 14B | `humaneval`, `ifeval_subset`, `overnight/clean_sweep_*` (Fig. 8) |
| Qwen3 (dense) | 1.7B, 4B, 8B | `lmeval_stage3` (8B), `mech_sqnr_validation` (all three), `overnight/rq93_*`, `overnight/i76_*` |
| Mistral | Mistral-7B-Instruct-v0.2 | core retention grid |
| Phi | Phi-3.5-mini-instruct (3.8B), Phi-4 (14B) | core grid; Phi-4 in `mech_sqnr_validation` |
| Granite 3.1 | 2B, 8B (Instruct) | 8B in core grid; 2B in `mech_sqnr_validation` + threshold grid |
| Falcon3 (dense) | Falcon3-3B | capability-threshold grid (`overnight/rq93_*`; the K2-surviving dense counterexample) |
| MoE | **Qwen3-30B-A3B**, Qwen1.5-MoE-A2.7B-Chat, OLMoE-1B-7B-0924-Instruct, DeepSeek-V2-Lite-Chat | `overnight/moe_*`, `overnight/distill_moe_fiel_*`, `moe_router2` |

† Qwen2.5-14B-Instruct is **excluded from GSM8K flexible-extract retention aggregates**: its FP
flexible-extract baseline shows a documented extractor inversion (flex 0.477 < strict 0.807 in
`f5_qwen25inst.jsonl`), which corrupts the retention denominator (K3 flex "retention" reads 124%;
strict-match gives 105%). Same exclusion as `overnight/rq93_capability_vs_robustness.json`.

Also present in `data/` but **not part of the Paper B population/claims**:

- **Pythia** (160m, 410m, 1.4b, 2.8b, 6.9b deduped; `f5_pythia.jsonl`) — the controlled-training
  ladder (same corpus, only size varies) used by the companion scale study to test whether
  "recovery scales with model size" is a confound of independent training.
- **GLM-4-9B-chat** (`hellaswag.jsonl` only) — a HellaSwag probe run not used in any Paper B figure
  or claim.

Notes:
- **MoE experts** in transformers 5.x are stored as a fused 3D tensor (`Qwen2/3MoeExperts.gate_up_proj`),
  not `nn.Linear`; the MoE runs here (`overnight/moe_*`, `moe_router2`) ternarize each expert slice
  explicitly. The parent-level `moe_qwen3_30b.jsonl` is an earlier partial run (no K2 arm) kept for
  provenance — the canonical Qwen3-30B numbers are in `overnight/moe_qwen3_30b_k2.jsonl`
  (see `data/overnight/README.md`).
- GLM-4 and Phi-3.5 load **without** `trust_remote_code` (the remote modeling files break on recent transformers).
