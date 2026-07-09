# MANIFEST — índice mestre do dado bruto

**Autoridade:** os *verdicts* (números, conclusões) são consolidados no ledger de desenvolvimento
`docs/backlog-done.md` (monorepo privado). Este pacote público embarca os **artefatos de dado** que
lastreiam cada número (`results/`, `reports/data/`, `reports/figures/`); este MANIFEST é o mapa de
localização deles: codinome-de-pod → item → paper → run canônico. Onde a atribuição a paper é inferida
(não lida direto do backlog), marco `≈`; onde é ambígua, aponto `→ backlog #NN`.

> **View navegável:** `results/paper-<id>/` reúne por **symlink** os stores de cada paper (dado real +
> stores in-place), sem mover nada. Ver `results/README.md`. Este MANIFEST é o índice textual; a view é
> o espelho navegável dele.

## 1) `runpod-runs/` — decodificador por item

Chave = número de item (`iNN`, presente nos nomes de arquivo e 1:1 com o backlog). Vários codinomes
cobrem o mesmo item (pods/retries/smokes distintos) — a coluna **canônico** diz qual run vale.

| item | codinome(s) em `runpod-runs/` | tópico | paper | run canônico / nota |
|---|---|---|---|---|
| #64 | `phd-ov-01`, `B1`(log) | multi-seed K2+distill | H | `phd-ov-01` (fechado, 0.333±0.015) |
| #65 | `phd-hhgptq`, `phd-w4gptq`, `phd-w7gptq2`, `C1`/`R76`(log) | GPTQ-int4 × K3 head-to-head | C | **`phd-hhgptq`** (`phd-w4gptq` 0.12 = bug CPU-offload, SUPERSEDED) |
| #66 | `A2`, `i66`, `phd-w1` | k-lever: últimos knobs (attn/β/rank/seq512) | B | `A2`/`phd-w1` |
| #67 | `A3`(teacher) | teacher/distill | H | `A3` |
| #68 | `RKN` | — | ≈B → backlog #68 | `RKN` |
| #69 | `RSAE2`, `RSAE`(log) | sobrevivência de feature em SAE | G | **`RSAE2`** (tem `.result.json`) |
| #70 | `phd-c70`, `phd-w2ct`, `RCT`(log) | circuit-twin IOI (TriLM/FloatLM) | A | `phd-c70`/`phd-w2ct` |
| #71 | `A3`(mixedk), `phd-w1`(mixedkd), `MK` | mixed-K + distill | B | `phd-w1`/`MK` |
| #72 | `RLC` | — | → backlog #72 | `RLC` |
| #73 | `A3`(0p5b) | escala 0.5B | ≈B | `A3` |
| #74 | `phd-w5abs`, `phd-w6absext` | absorption ladder 99M–3.9B | G | `phd-w6absext` (ext) |
| #76 | `R76`, `C2`(log) | K3 código × geração | ≈B/C | **`R76`** (`C2` = só logs) |
| #77 | `RMOE`, `B1`(log) | fidelidade MoE | ≈B | `RMOE` |
| #78 | `E` | — | ≈E → backlog #78 | `E` |
| #79 | `RSAFE2`, `RKN`/`RSAFE`(log) | — | → backlog #79 | **`RSAFE2`** |
| #80 | `C1` | — | → backlog #80 | `C1` |
| #81 | `RLC` | — | → backlog #81 | `RLC` |
| #82 | `E` | — | ≈E → backlog #82 | `E` |
| #83 | `RMOE` | correlação MoE | ≈B | `RMOE` |
| #84 | `phd-w884`, `phd-w2ct`(minilm), `MINILM`(log) | MiniLM QKV-KD + A/B ApiQ | H | `phd-w884` (b) |
| #85 | `C1`(r1_1p5b) | escala 1.5B | ≈C | `C1` |
| #86 | `REDIT`, `phd-w3ed`, `EDIT`(variantes), `phd-w9edit`(vazio) | edição de modelo sob ternário | backlog (sem paper próprio) | `REDIT` |
| #93 | `RQ93` | K2-cliff = limiar de capacidade | B | `RQ93` (+`rq93_fig.png`) |
| #94 | `BENCH94` | microbench SIMD fused TQ2P/TQ3P | F | `BENCH94` |
| #95 | `PLANE95` | compressibilidade de planos | ≈B/F | `PLANE95` |
| MoE | `MOE2`(olmoe), `MOE3`(dsv2lite), `MOE90`(chunked), `RK2`(qwen3-30b), `phd-r2`(qwen3-30b k2) | fidelidade ternária MoE | ≈B (REDO R2) | ver backlog #58/#61 |
| REDO | `phd-r3`(hawq), `phd-r4`(14b) | REDOs da auditoria Tier-1 | E (r3), C (r4) | ver backlog |
| df | `phd-hhdf`(hh_datafree) | data-free / crosscoder | ≈G/H | → backlog |

> `phd-w9edit/` está **vazio** (run que nunca produziu saída) — candidato a `trash/` numa próxima passada.

> **Órfãos load-bearing agora navegáveis por-paper (2026-07-09):** `phd-r2` (MoE-30B canônico, o lado
> "96%") e `phd-hhdf` (head-to-head INT4) deixaram de estar soltos — foram linkados nas views:
> `paper-b-kplane/{runpod-moe30b-phd-r2, runpod-int4hh-phd-hhdf}` e `paper-c-scale/runpod-moe30b-phd-r2`,
> de modo que o empacotamento `tar -czh` de cada view já os inclui.

## 2) Stores in-place (ficam no pacote self-contained; ponteiro + proveniência)

Movê-los quebraria scripts que os leem por caminho relativo — por isso **permanecem** onde estão:

| store | paper | é a fonte-única de |
|---|---|---|
| `sasori/bench/*.json` | F | benchmarks CPU/throughput/retention/pareto do kernel sasori |
| `trilho/artifacts/sota-2026-06-13/` | C + sistemas | rodada SOTA 2026-06-13 (BitDraft/CTC/fair-fight/e2e), lida por `make_paperC_figs.py` |
| `trilho/artifacts/gpu-session-2026-06-12/` | sistemas | bench GPU (BitBLAS/GEMV, A40/H100) |
| `ternary-boost/results/` | H | evals Artigo-2 (fp16/ternary/+LoRA, wikitext/task) |
| `experiments/interp/` | A/E/G | dados+código de interp (EAP/IOI/SAE); inclui `sae_absorption_*_2026-07-01.json` (Colab) |
| `experiments/dynamics/` | D | dados+código da dinâmica de treino |
| `experiments/ternary-quantization/{data,results}/` | B/C | dados+código do estudo K-plano |
| `reports/data/` + `reports/figures/` | A–E | snapshot de dado+figuras dos relatórios longos pré-paper (`reports/*.md`) |

## 3) Figuras — cadeia de geração (não é duplicata acidental)

`scripts *.py → reports/figures/figNN.png` (canônico gerado) → cópia renomeada em
`papers/<id>/figures/` (build self-contained do paper) → `papers/arxivorg/<id>/` é **build-artifact**
regenerado por `papers/build_arxiv.sh` (não versionado). Cada nível serve um público; sob a política
"moderada" escolhida, `reports/figures` ↔ `papers/figures` coexistem por design.
