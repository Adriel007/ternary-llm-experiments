# results/ — dado bruto de experimento (home único + índice)

Este diretório é o **home único do dado bruto de rodadas** (Colab/RunPod) e o **índice mestre**
(`MANIFEST.md`) de *onde vive cada pedaço de dado experimental do projeto* — inclusive o que, por
boa engenharia, **permanece no seu pacote self-contained** (ver abaixo).

## O que está aqui

```
results/
├── README.md        (este arquivo)
├── MANIFEST.md       (ÍNDICE MESTRE: codinome→item→paper→status, + ponteiros aos stores in-place)
├── runpod-runs/      (DADO REAL — era "overnight-results/": saídas brutas RunPod/Colab,
│                       um subdir por codinome-de-pod; .result.json = resultado, .log = execução)
└── paper-<id>/       (VIEW por-paper: SYMLINKS pros dados reais onde eles vivem — navegável)
```

### `paper-<id>/` = view por-paper (symlinks, não cópias)

Cada `results/paper-<id>/` reúne, via **symlink**, os stores relevantes àquele paper — tanto o dado
real em `runpod-runs/` quanto os stores in-place nos pacotes self-contained. **Nada é copiado nem
movido**: o link aponta pra source, que segue funcionando pros scripts. Assim você tem organização
por-paper E a source intacta ao mesmo tempo. Ex.: `paper-c-scale/trilho-sota-2026-06-13 →
../../trilho/artifacts/sota-2026-06-13`. Para empacotar tudo de um paper resolvendo os links:
`tar -czhf paper-c.tgz -C results paper-c-scale` (o `-h` deref os symlinks).

**Honestidade:** só linko onde a atribuição é confiante. Itens runpod ambíguos (#68/72/78/79/80/81/82/
83/85/86) **não** são forçados a um paper — ficam acessíveis em `runpod-runs/` e mapeados no MANIFEST.
`reports/data`+`reports/figures` servem A–E de forma compartilhada → não linko per-paper (implicaria
exclusividade falsa); ficam indexados no MANIFEST.

## Princípio: fonte-única, não-monolítica

- **Verdicts/ciência** são fonte-única em `docs/backlog-done.md` (versionado). O `MANIFEST.md`
  aqui **aponta** pra lá — não copia números (evita divergência).
- **Dado bruto de rodada** (RunPod/Colab) fica **aqui** em `runpod-runs/`.
- **Dado co-localizado com o código que o produziu** (pacotes self-contained) **fica no pacote** —
  movê-lo quebraria os scripts que o leem por caminho relativo (`os.path.join(HERE, …)`). O
  `MANIFEST.md` indexa esses stores com ponteiro + proveniência. São eles:
  `sasori/bench/` (Paper F), `trilho/artifacts/` (Paper C + sistemas), `ternary-boost/results/`
  (Paper H), `experiments/*/` (dado + código dos estudos), `reports/data/` + `reports/figures/`
  (snapshot dos relatórios longos pré-paper).

## Nota

Runs redundantes/superseded (smoke-tests, retries, execuções com bug já diagnosticado) **não são
apagados** — são o registro do processo. O `MANIFEST.md` marca qual run é o **canônico** por item.
