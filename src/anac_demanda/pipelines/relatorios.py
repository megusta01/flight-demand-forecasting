"""Relatórios em Markdown/JSON gerados a partir dos resultados do treino."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from ..config import Config
from ..features import FEATURES_CATEGORICAS, FEATURES_NUMERICAS
from ..periodos import Mes

if TYPE_CHECKING:
    from .treinar import ResultadoTreino

_PCT = {"WAPE", "vies", "WAPE_modelo", "WAPE_baseline", "participacao", "aumento_mae_pct"}
_INT = {"MAE", "RMSE", "aumento_mae", "passageiros", "previsto_modelo", "previsto_baseline"}


def tabela_md(df: pd.DataFrame) -> str:
    out = df.copy()
    for col in out.columns:
        if col in _PCT:
            out[col] = out[col].map(lambda v: f"{v:.1%}")
        elif col in _INT:
            out[col] = out[col].map(lambda v: f"{v:,.0f}")
    return out.to_markdown(index=False)


def _monitoramento_md(r: ResultadoTreino) -> str:
    m = r.monitoramento
    if not m:
        return "Não há meses posteriores ao teste na base."
    ini, fim = Mes.parse(m["inicio"]).rotulo, Mes.parse(m["fim"]).rotulo
    return (
        f"Modelo de produção (treinado até {r.extras['treinado_ate']}) avaliado em {ini} a {fim}, "
        f"meses publicados depois do congelamento ({m['amostras']:,} previsões):\n\n"
        f"| | WAPE | MAE |\n|---|---|---|\n"
        f"| Modelo | {m['modelo']['WAPE']:.1%} | {m['modelo']['MAE']:,.0f} |\n"
        f"| {r.baseline_referencia} | {m['baseline']['WAPE']:.1%} | {m['baseline']['MAE']:,.0f} |"
    )


def escrever_resultados(r: ResultadoTreino, cfg: Config) -> None:
    covid = r.covid
    n = r.linhas
    amostras = (
        f"treino {n['treino']:,} · validação {n['validacao']:,} · teste {n['teste']:,} · "
        f"monitoramento {n['monitoramento']:,}"
    )
    texto = f"""# Resultados — previsão de passageiros um mês à frente

Gerado por `anac-demanda treinar`.

## Configuração
- Série: rota operada por uma empresa (empresa + origem + destino), voos {cfg.grupo_de_voo.lower()}.
- Alvo: passageiros pagos no mês seguinte.
- Amostras: {amostras}.
- Métrica principal: WAPE = soma dos erros absolutos / soma dos passageiros reais.

## Validação ({cfg.validacao[0]} a {cfg.validacao[1]}) — escolha do modelo
{tabela_md(r.validacao)}

Modelo escolhido: **{r.modelo_escolhido}** · baseline de referência: **{r.baseline_referencia}**.

## Meses da COVID no treino
| Treino | WAPE validação |
|---|---|
| Com mar/2020–dez/2021 (+ flag `covid`) | {covid["WAPE_com_covid"]:.2%} |
| Sem mar/2020–dez/2021 | {covid["WAPE_sem_covid"]:.2%} |

Decisão ({covid["decidido_por"]}): **{"remover" if covid["remover"] else "manter"}** esses meses no treino final.

## Teste ({cfg.teste[0]} a {cfg.teste[1]})
{tabela_md(r.teste)}

Redução de erro frente ao baseline de referência: **{r.reducao_erro:.1%}**.

## Monitoramento
{_monitoramento_md(r)}

## Erro por porte da rota (teste)
{tabela_md(r.porte)}

## Erro por mês (teste e monitoramento)
{tabela_md(r.mensal)}

## Importância das variáveis (validação)
{tabela_md(r.importancia.head(15))}
"""
    (cfg.dir_relatorios / "resultados.md").write_text(texto, encoding="utf-8")


def escrever_model_card(r: ResultadoTreino, cfg: Config) -> None:
    mt = r.teste.set_index("modelo").loc[r.modelo_escolhido]
    proxima = r.extras.get("proxima_previsao")
    mon = r.monitoramento["modelo"] if r.monitoramento else None
    linha_mon = f"| {mon['WAPE']:.1%} | {mon['MAE']:,.0f} | {mon['vies']:+.1%} |" if mon else "| — | — | — |"
    n_feats = f"{len(FEATURES_NUMERICAS)} numéricas e {len(FEATURES_CATEGORICAS)} categóricas"
    covid_txt = "removidos" if r.covid["remover"] else "mantidos"
    texto = f"""# Model card — previsão de passageiros por rota

Gerado por `anac-demanda treinar`.

## Uso pretendido
Estimar o total de passageiros pagos de cada rota regular (empresa + origem + destino) no mês seguinte ao
último mês publicado pela ANAC, como apoio ao planejamento de oferta. Não substitui informação interna das
empresas (vendas antecipadas, malha programada).

## Modelo
- Algoritmo: **{r.modelo_escolhido}** (scikit-learn `HistGradientBoostingRegressor`).
- Formulação: prevê a variação `pax[t+1] / pax[t]` com peso `pax[t]`; a previsão é `pax[t] × variação`.
- Features: {n_feats}, todas com informação até o mês *t*.
- Treinado com meses-alvo até **{r.extras["treinado_ate"]}**; mar/2020 a dez/2021 {covid_txt} (decisão por validação).
- Artefato: `artifacts/modelo.joblib` (metadados em `artifacts/modelo.json`).

## Desempenho
| Conjunto | WAPE | MAE (passageiros/rota/mês) | Viés |
|---|---|---|---|
| Teste {cfg.teste[0]} a {cfg.teste[1]} | {mt["WAPE"]:.1%} | {mt["MAE"]:,.0f} | {mt["vies"]:+.1%} |
| Monitoramento {linha_mon}

Baseline de referência no teste ({r.baseline_referencia}): WAPE {r.wape_baseline_teste:.1%}.

## Limitações
- Só prevê rotas com passageiros no mês atual; rotas que reabrem depois de uma pausa ficam de fora.
- Não antecipa choques externos (desastres, falências, mudanças de malha anunciadas).
- Rotas com menos de mil passageiros/mês têm erro alto (WAPE acima de 40%).
- Previsão pontual, sem intervalo de confiança.

## Próxima previsão
{f"{proxima[0].rotulo}: {proxima[1]:,} passageiros previstos no total das rotas ativas." if proxima else "—"}
"""
    (cfg.dir_relatorios / "model_card.md").write_text(texto, encoding="utf-8")


def resumo_json(r: ResultadoTreino) -> dict:
    proxima = r.extras.get("proxima_previsao")
    return {
        "modelo_escolhido": r.modelo_escolhido,
        "baseline_referencia": r.baseline_referencia,
        "amostras": r.linhas,
        "validacao": r.validacao.to_dict(orient="records"),
        "covid": r.covid,
        "teste": r.teste.to_dict(orient="records"),
        "reducao_erro_teste": r.reducao_erro,
        "monitoramento": r.monitoramento,
        "proxima_previsao": {"mes": str(proxima[0]), "total": proxima[1]} if proxima else None,
    }
