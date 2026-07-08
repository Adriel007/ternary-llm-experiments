# MAPA CAPACIDADE x K — reasoning-specific (gerado 2026-06-27T15:11:43)
K2-retencao (acc_K2/acc_K0, N=200, eval logprob proprio NAO-calibrado -> usar so razao/ordenacao).

| modelo | GSM8K math K2-ret | ARC sci K2-ret | HellaSwag commonsense K2-ret |
|---|---|---|---|
| Qwen2.5-7B-Instruct | 12% | 81% | 76% |
| Qwen3-4B | 40% | 66% | 74% |
| Qwen3-8B | 4% | 51% | 72% |
| glm-4-9b-chat-hf | 39% | 89% | 79% |

**FATO:** K2-retencao sobe c/ MENOS raciocinio multi-passo: math 4%-40% (severo+ERRATICO) < ciencia 51%-89% < commonsense 72%-79% (UNIFORME). Erraticidade e ESPECIFICA do math (HellaSwag ~uniforme em todos). K3 resgata tudo.
**CAVEAT:** evals MC proprios nao-calibrados (baseline<lm-eval); razao K2/K0 e ordenacao cross-tarefa sao validas; absolutos nao.
