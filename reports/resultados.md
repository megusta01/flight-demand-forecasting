# Resultados — previsão de passageiros um mês à frente

Gerado por `anac-demanda treinar`.

## Configuração
- Série: rota operada por uma empresa (empresa + origem + destino), voos regular.
- Alvo: passageiros pagos no mês seguinte.
- Amostras: treino 527,054 · validação 18,917 · teste 38,117 · monitoramento 6,728.
- Métrica principal: WAPE = soma dos erros absolutos / soma dos passageiros reais.

## Validação (2023-01 a 2023-12) — escolha do modelo
| modelo                                         | tipo     | WAPE   |   MAE |   RMSE | vies   |
|:-----------------------------------------------|:---------|:-------|------:|-------:|:-------|
| Gradient Boosting (variação, perda absoluta)   | modelo   | 12.3%  |   711 |  1,389 | -0.9%  |
| Gradient Boosting (variação, perda quadrática) | modelo   | 12.5%  |   720 |  1,390 | -1.2%  |
| Gradient Boosting (Poisson)                    | modelo   | 12.8%  |   741 |  1,495 | -1.0%  |
| Gradient Boosting (log)                        | modelo   | 14.6%  |   842 |  1,579 | -6.2%  |
| Ingênuo (repete o mês atual)                   | baseline | 14.8%  |   852 |  1,630 | -0.1%  |
| Ingênuo com ajuste sazonal                     | baseline | 17.8%  | 1,027 |  2,000 | 2.2%   |
| Sazonal x crescimento recente                  | baseline | 20.6%  | 1,190 |  2,196 | 2.2%   |
| Sazonal (repete o mesmo mês do ano anterior)   | baseline | 24.9%  | 1,438 |  2,634 | -10.2% |
| Ridge (log)                                    | modelo   | 27.8%  | 1,604 |  3,574 | 6.3%   |

Modelo escolhido: **Gradient Boosting (variação, perda absoluta)** · baseline de referência: **Ingênuo (repete o mês atual)**.

## Meses da COVID no treino
| Treino | WAPE validação |
|---|---|
| Com mar/2020–dez/2021 (+ flag `covid`) | 12.32% |
| Sem mar/2020–dez/2021 | 12.27% |

Decisão (validação): **remover** esses meses no treino final.

## Teste (2024-01 a 2025-12)
| modelo                                       | tipo     | WAPE   |   MAE |   RMSE | vies   |
|:---------------------------------------------|:---------|:-------|------:|-------:|:-------|
| Gradient Boosting (variação, perda absoluta) | modelo   | 11.1%  |   701 |  1,467 | -0.6%  |
| Ingênuo (repete o mês atual)                 | baseline | 13.3%  |   841 |  1,691 | -0.1%  |
| Ingênuo com ajuste sazonal                   | baseline | 16.3%  | 1,028 |  2,187 | 1.8%   |
| Sazonal x crescimento recente                | baseline | 18.6%  | 1,179 |  2,425 | 1.3%   |
| Sazonal (repete o mesmo mês do ano anterior) | baseline | 22.3%  | 1,408 |  2,720 | -6.0%  |

Redução de erro frente ao baseline de referência: **16.6%**.

## Monitoramento
Modelo de produção (treinado até 2025-12) avaliado em jan/2026 a abr/2026, meses publicados depois do congelamento (6,728 previsões):

| | WAPE | MAE |
|---|---|---|
| Modelo | 12.5% | 813 |
| Ingênuo (repete o mês atual) | 17.3% | 1,120 |

## Erro por porte da rota (teste)
| porte          |   amostras | participacao   | WAPE_modelo   | WAPE_baseline   |
|:---------------|-----------:|:---------------|:--------------|:----------------|
| até 1 mil      |       9255 | 1.6%           | 43.6%         | 45.3%           |
| 1 a 5 mil      |      13972 | 17.2%          | 17.0%         | 19.0%           |
| 5 a 20 mil     |      12551 | 52.7%          | 10.2%         | 12.5%           |
| mais de 20 mil |       2339 | 28.5%          | 7.4%          | 9.7%            |

## Erro por mês (teste e monitoramento)
|   ano |   mes |   passageiros |   previsto_modelo |   previsto_baseline | WAPE_modelo   | WAPE_baseline   |
|------:|------:|--------------:|------------------:|--------------------:|:--------------|:----------------|
|  2024 |     1 |     9,801,061 |         9,905,512 |           9,709,986 | 14.6%         | 18.5%           |
|  2024 |     2 |     8,538,197 |         8,696,426 |           9,928,084 | 11.5%         | 20.7%           |
|  2024 |     3 |     9,340,508 |         9,160,628 |           8,567,226 | 12.1%         | 16.3%           |
|  2024 |     4 |     8,920,541 |         9,229,547 |           9,353,203 | 16.6%         | 17.3%           |
|  2024 |     5 |     8,928,529 |         9,176,593 |           8,971,075 | 14.6%         | 15.1%           |
|  2024 |     6 |     9,076,214 |         8,801,880 |           8,972,159 | 9.0%          | 9.6%            |
|  2024 |     7 |    10,117,454 |         9,973,032 |           9,115,164 | 12.5%         | 15.7%           |
|  2024 |     8 |     9,841,616 |         9,949,954 |          10,244,065 | 11.1%         | 13.2%           |
|  2024 |     9 |     9,627,772 |         9,660,950 |           9,898,679 | 9.1%          | 9.2%            |
|  2024 |    10 |     9,756,136 |         9,853,649 |           9,637,742 | 8.1%          | 9.0%            |
|  2024 |    11 |     9,737,259 |         9,826,329 |           9,927,710 | 16.9%         | 17.7%           |
|  2024 |    12 |    10,120,628 |        10,146,270 |           9,767,943 | 10.1%         | 12.1%           |
|  2025 |     1 |    10,556,573 |        10,569,738 |          10,229,816 | 10.9%         | 15.0%           |
|  2025 |     2 |     9,369,660 |         9,344,895 |          10,637,675 | 11.9%         | 20.0%           |
|  2025 |     3 |    10,106,837 |         9,986,163 |           9,423,761 | 9.5%          | 12.8%           |
|  2025 |     4 |     9,962,001 |         9,946,952 |          10,142,905 | 12.9%         | 13.9%           |
|  2025 |     5 |    10,283,619 |        10,121,129 |          10,005,739 | 7.2%          | 7.7%            |
|  2025 |     6 |    10,246,325 |         9,756,477 |          10,298,317 | 8.0%          | 7.6%            |
|  2025 |     7 |    11,301,303 |        10,959,963 |          10,326,538 | 12.2%         | 14.7%           |
|  2025 |     8 |    11,052,625 |        10,998,439 |          11,358,579 | 10.6%         | 12.7%           |
|  2025 |     9 |    10,706,866 |        10,713,087 |          11,100,793 | 8.0%          | 8.6%            |
|  2025 |    10 |    11,192,551 |        10,828,333 |          10,756,252 | 7.7%          | 8.6%            |
|  2025 |    11 |    10,926,707 |        10,726,427 |          11,247,202 | 12.5%         | 13.6%           |
|  2025 |    12 |    11,376,643 |        11,144,164 |          10,947,473 | 10.6%         | 12.7%           |
|  2026 |     1 |    12,000,370 |        11,971,283 |          11,529,778 | 10.7%         | 14.7%           |
|  2026 |     2 |    10,359,052 |        10,509,886 |          12,064,675 | 9.7%          | 19.1%           |
|  2026 |     3 |    11,074,323 |        10,974,698 |          10,371,372 | 12.7%         | 16.9%           |
|  2026 |     4 |    10,153,067 |        10,942,533 |          11,095,150 | 17.5%         | 18.9%           |

## Importância das variáveis (validação)
| feature                    |   aumento_mae | aumento_mae_pct   |
|:---------------------------|--------------:|:------------------|
| pax_lag0                   |         6,107 | 859.1%            |
| mes_alvo                   |            74 | 10.4%             |
| decolagens_lag0            |            26 | 3.6%              |
| variacao_sazonal_1a        |            21 | 3.0%              |
| rel_pax_media_12m          |            15 | 2.1%              |
| aeroporto_de_origem_sigla  |            14 | 2.0%              |
| aeroporto_de_destino_sigla |            12 | 1.7%              |
| rel_pax_lag11              |            12 | 1.7%              |
| rel_pax_lag2               |            10 | 1.4%              |
| ocupacao_lag0              |             9 | 1.3%              |
| empresa_sigla              |             9 | 1.3%              |
| rel_pax_media_3m           |             7 | 1.0%              |
| mercado_crescimento_aa     |             3 | 0.4%              |
| mercado_lag0               |             3 | 0.4%              |
| rel_assentos_lag0          |             3 | 0.4%              |
