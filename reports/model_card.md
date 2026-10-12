# Model card — previsão de passageiros por rota

Gerado por `anac-demanda treinar`.

## Uso pretendido
Estimar o total de passageiros pagos de cada rota regular (empresa + origem + destino) no mês seguinte ao
último mês publicado pela ANAC, como apoio ao planejamento de oferta. Não substitui informação interna das
empresas (vendas antecipadas, malha programada).

## Modelo
- Algoritmo: **Gradient Boosting (variação, perda absoluta)** (scikit-learn `HistGradientBoostingRegressor`).
- Formulação: prevê a variação `pax[t+1] / pax[t]` com peso `pax[t]`; a previsão é `pax[t] × variação`.
- Features: 31 numéricas e 7 categóricas, todas com informação até o mês *t*.
- Treinado com meses-alvo até **2025-12**; mar/2020 a dez/2021 removidos (decisão por validação).
- Artefato: `artifacts/modelo.joblib` (metadados em `artifacts/modelo.json`).

## Desempenho
| Conjunto | WAPE | MAE (passageiros/rota/mês) | Viés |
|---|---|---|---|
| Teste 2024-01 a 2025-12 | 11.1% | 701 | -0.6% |
| Monitoramento | 12.5% | 813 | +1.9% |

Baseline de referência no teste (Ingênuo (repete o mês atual)): WAPE 13.3%.

## Limitações
- Só prevê rotas com passageiros no mês atual; rotas que reabrem depois de uma pausa ficam de fora.
- Não antecipa choques externos (desastres, falências, mudanças de malha anunciadas).
- Rotas com menos de mil passageiros/mês têm erro alto (WAPE acima de 40%).
- Previsão pontual, sem intervalo de confiança.

## Próxima previsão
mai/2026: 10,461,212 passageiros previstos no total das rotas ativas.
