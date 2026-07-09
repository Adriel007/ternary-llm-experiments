# Scale-invariant curvature control (Paper D, "Sharper, Not Flatter")

Protocol to decide whether the raw-curvature headline is geometry or a
reparameterization artifact. Pre-registered: decide the reading BEFORE looking at
the result.

Files:
- `demo_scale_confound.py` — pure-numpy demonstration that the control is needed (~2 s).
- `scale_invariant_metrics.py` — torch module: the three lenses + a verdict, wired to
  `metrics.dequantized_float_copy` so ternary and FP are probed identically.

## 1. The problem, in one sentence
The Hessian trace / lambda_max in RAW parameter space is not invariant to
function-preserving reparameterizations (Dinh et al. 2017, cited as [3] in the
paper). A realized ternary net (weights in {-b, 0, +b}) operates at a smaller weight
norm than its FP twin; that alone moves the trace. So "ternary trace = 2.15x FP" is
consistent with two hypotheses the raw data cannot separate:
- H1 (headline): the ternary minimum is geometrically sharper.
- H0 (confound): the same kind of minimum, read in a smaller-norm parameterization at
  a worse loss (the ternary point is less converged: 2.844 vs FP 2.726 nats).

## 2. Why the control is necessary (run the demo)
`demo_scale_confound.py` shows (a) a function-preserving rescaling (W1 -> c*W1,
W2 -> W2/c) changes the raw trace by ~9x while the loss is bit-identical, and (b) the
ternary-vs-FP weight-norm offset is of the same order that Part (a) shows moves the
trace. The raw 2.15x fits inside a pure confound. This does NOT prove the headline is
wrong; it proves it is undecidable as reported.

## 3. The three controls (run all; #1 is decisive and cheap)
1. **Same-loss (cheapest, do first).** Save FP checkpoints along training; pick the
   FP checkpoint whose train loss equals the ternary final loss (2.844), then compare
   the trace at the same loss. Removes the "less converged -> sharper" confound.
   `early_stop_to_loss(...)`.
2. **Filter-normalization (Li et al. 2018).** Probe with a Rademacher vector scaled
   per weight-filter by the filter norm; immune to positive-scale symmetry.
   `filter_scales(...)` + `hutchinson_trace(..., scaling=...)`.
3. **Fisher / Adam-preconditioned curvature.** The diagonal Fisher trace is the
   metric the optimizer actually sees. `fisher_diag_trace(...)`.

Hygiene fixes to land with it: measure on the real W1.58A8 model (with A8), not only
the float surrogate; >=5 seeds (currently 3); unify the t*=1.0 definition between toy
and scale; release the raw data (`reports/data/`).

## 4. Decision rule (pre-registered)
Let r = trace_ternary / trace_FP under filter-norm (control 2), at the same loss
(control 1):

| r | reading | editorial action |
|---|---|---|
| r > 1.15 | genuinely sharper | headline SUSTAINED; report the three lenses; keep the title. |
| 0.87 <= r <= 1.15 | comparable | demote to "curvature comparable under scale-invariant metrics; the raw difference is parameterization". |
| r < 0.87 | inverts | rewrite: "raw sharpness grows but vanishes/inverts under reparameterization" -- still publishable and honest. |

`print_decision(...)` implements this tree.

## 5. Why this is win-win
If r > 1.15 the headline stops being attackable and gains three supporting lenses; if
r <= 1.15 the paper reports a correct methodological fact (the raw effect is
parameterization), useful to anyone measuring sharpness in quantized nets, and avoids
a future retraction. Controls 2 and 3 are post-processing of already-trained models;
control 1 needs only saving FP checkpoints in a short toy re-run (TinyStories, 2000
steps) -- a pod job, not local.

## Result (2026-07-08, RTX 3090, 5 seeds, run_scale_invariant_control.py)
Artifact: `reports/data/scale_invariant_control_d.json`. Raw trace ratio reproduces
the published headline (2.07x vs 2.15x); losses match the paper (ternary 2.86 / FP
2.73). Filter-normalized ratio (ratio-of-means over 5 seeds):

| lens | all-params, final FP | all-params, same-loss FP | BitLinear only |
|---|---|---|---|
| raw | 2.07 | 2.54 | 2.42 |
| filter-norm | 0.81 | **0.99** | 0.70-0.91 |
| Fisher | 1.96 | 2.34 | 2.43-3.08 |

Verdict (pre-registered): applying BOTH controls (filter-norm at matched loss) gives
**0.99x -> the sharpness signal does NOT survive.** The raw 2.15x is a
reparameterization/convergence artifact (confirms H0 / Dinh 2017). The Fisher
(optimizer-preconditioned) curvature IS ~2x larger -- a genuine but distinct effect
(optimization surface, not invariant loss geometry). Paper D title/abstract/body
rewritten accordingly (uncommitted, pending Adriel's review).
