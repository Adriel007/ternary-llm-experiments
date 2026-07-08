# Quantization configuration

## K-trit-plane joint-ridge PTQ
Each weight matrix `W` is approximated by `K` stacked ternary planes:
`W ≈ Ŵ_K = Σ_{k=1..K} diag(α^k) T^k`, with `T^k ∈ {-1,0,+1}` and per-(row,group) scales `α^k`,
fit by alternating least squares / joint ridge over the planes (the reference implementation is
`sasori.reconstruct.quantize_matrix_k`).

Default knobs (used unless a file says otherwise):
- `group = 256` (per-row group size for the scales).
- `niter = 25` ALS iterations (15 for the very large MoE runs, for time).
- `lam = 1e-2` ridge.
- `row_chunk = 2048` — caps the temporary tensor in the `3^K`-candidate argmin (prevents VRAM blow-up at K5).

Effective bits/weight (with g=256 ternary planes): **K2 ≈ 3.29 bpw, K3 ≈ 4.94 bpw, K4 ≈ 6.6, K5 ≈ 8.2**
(see the `bpw` field in `attn_vs_mlp_v3.jsonl` / `mixedk_*.jsonl`).

## What gets quantized
- Dense models: all block linears (q/k/v/o_proj, gate/up/down_proj). Embeddings/lm_head/norms kept FP.
- MoE: the **fused 3D expert tensors** are ternarized per-expert slice (not skipped — `fakequant_inplace`
  on `nn.Linear` alone misses them); attention + shared-expert linears also ternarized; router kept FP by
  default (`moe_router2.jsonl` shows router FP-vs-ternary is within noise).
- Evaluation is via **dequantize → bf16** (fake-quant): isolates the representational cost, independent of kernel.

## Source-format control (fase 0, byte-test)
Ternarizing from bf16 vs from f16(←bf16) produces the **byte-identical** ternary artifact (ham=0, Δscale=0,
deterministic) across 48/48 tensor cells — so source dtype is not a confound. bf16 is used as the FP
baseline because it is the models' native dtype.

## Allocation variants (in the mechanism files)
- **mixed-K**: a fraction `f` of blocks at K3, the rest at K2 (`mixedk_dual/arc/sqnr.jsonl`).
- **by-type**: attention at K3 + MLP at K2 (and the inverse) — `attn_vs_mlp_v3.jsonl`.
- **SQNR-guided**: protect the blocks with lowest late-activation SQNR (vs random allocation).
The robust takeaway: K is the primary lever; group-size is a secondary refinement; cheap allocation
signals ≈ random for math (rugged optimum), with attention being the structural bottleneck.

## K-predictor tool
A companion CLI (`sasori suggest-k`) estimates the minimal K for a target retention from a cheap
late-activation SQNR measurement; its calibration and the R(K)-law derivation live with the sasori
package (`k_predictor_calib.json`, `k-predictor-math.md`). The SQNR ladders here (`sqnr_kladder.jsonl`,
`mech_sqnr_validation.jsonl`) are its empirical basis.
