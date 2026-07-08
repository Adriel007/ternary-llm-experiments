# ternary-llm-experiments

Reproducibility package for a series of preprints on the **interpretability and
post-training ternary (1.58-bit / K-trit-plane) quantization of large language models**.
Each headline number in the papers has a committed artifact here.

- `experiments/` — analysis and harness code (e.g. `experiments/ternary-quantization/`,
  `experiments/interp/`, `experiments/dynamics/`), plus the fidelity gate and env stamp.
- `results/` — raw artifacts organized per paper (`results/paper-a-circuits/` …
  `results/paper-h-distill/`) with a `MANIFEST.md`.

Evaluation numbers use the official
[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness); the ternary
inference tooling (kernel + pipeline) lives in the companion repo **sasori**.

## Papers (Zenodo, CC-BY-4.0)

| | Title | DOI |
|---|---|---|
| A | The Same Circuits, in {-1,0,+1}: A Mechanistic Circuit Anatomy of a Deployed 1.58-bit LM | [10.5281/zenodo.21245579](https://doi.org/10.5281/zenodo.21245579) |
| B | How Many Ternary Planes Does Reasoning Need? | [10.5281/zenodo.21246002](https://doi.org/10.5281/zenodo.21246002) |
| C | Scale-Dependence of Post-Hoc Ternary Reasoning Recovery | [10.5281/zenodo.21246354](https://doi.org/10.5281/zenodo.21246354) |
| D | Sharper, Not Flatter: The Training Dynamics of From-Scratch Ternary LMs | [10.5281/zenodo.21247324](https://doi.org/10.5281/zenodo.21247324) |
| E | Where to Spend the Bits | [10.5281/zenodo.21247473](https://doi.org/10.5281/zenodo.21247473) |
| F | sasori: No-Retrain Multi-Plane Ternarization for GPU-Free LLM Inference | [10.5281/zenodo.21247534](https://doi.org/10.5281/zenodo.21247534) |
| G | How Faithfully Do Sparse Autoencoders Read a Deployed 1.58-bit LM? | [10.5281/zenodo.21248079](https://doi.org/10.5281/zenodo.21248079) |
| H | Distillation Recovers the Post-Training Ternary Reasoning Collapse | [10.5281/zenodo.21248308](https://doi.org/10.5281/zenodo.21248308) |

## License

MIT (see `LICENSE`). The papers themselves are CC-BY-4.0 on Zenodo.
