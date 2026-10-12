"""Pipeline de análise exploratória: figuras + resumo em Markdown."""

from __future__ import annotations

import pandas as pd

from ..config import Config
from ..data import carregar_dados
from ..logs import cronometro, get_logger
from ..viz import eda

log = get_logger(__name__)


def ultimo_ano_completo(df: pd.DataFrame) -> int:
    meses = df.groupby("ano")["mes"].nunique()
    return int(meses[meses == 12].index.max())


def executar(cfg: Config) -> dict:
    cfg.garantir_diretorios()
    df = carregar_dados(cfg)
    ano = ultimo_ano_completo(df)
    with cronometro(log, "Análise exploratória"):
        eda.passageiros_por_ano(df, cfg.dir_figuras, ano)
        eda.sazonalidade(df, cfg.dir_figuras, ano)
        eda.principais_rotas(df, cfg.dir_figuras, ano, cfg.grupo_de_voo)
        eda.participacao_empresas(df, cfg.dir_figuras, ano, cfg.grupo_de_voo)
        resumo = _escrever_relatorio(df, cfg, ano)
    return resumo


def _escrever_relatorio(df: pd.DataFrame, cfg: Config, ano: int) -> dict:
    anual = df.groupby("ano")["passageiros_pagos"].sum()
    grupos = df.groupby("grupo_de_voo", dropna=False)["passageiros_pagos"].sum()
    grupos = (grupos / grupos.sum()).sort_values(ascending=False)
    ausentes = df.isna().mean().sort_values(ascending=False)
    ultimo = df[df["ano"] == df["ano"].max()]["mes"].max()
    resumo = {
        "registros": len(df),
        "empresas": int(df["empresa_sigla"].nunique()),
        "aeroportos": int(pd.concat([df["aeroporto_de_origem_sigla"], df["aeroporto_de_destino_sigla"]]).nunique()),
        "periodo": f"{df['ano'].min()}-01 a {df['ano'].max()}-{ultimo:02d}",
        "queda_2020": float(1 - anual.loc[2020] / anual.loc[2019]) if {2019, 2020} <= set(anual.index) else None,
        "nivel_vs_2019": float(anual.loc[ano] / anual.loc[2019]) if 2019 in anual.index else None,
        "participacao_regulares": float(grupos.get(cfg.grupo_de_voo, 0.0)),
    }
    texto = f"""# Análise exploratória

Gerado por `anac-demanda explorar`.

| Indicador | Valor |
|---|---|
| Período | {resumo["periodo"]} |
| Registros | {resumo["registros"]:,} |
| Empresas | {resumo["empresas"]:,} |
| Aeroportos | {resumo["aeroportos"]:,} |
| Passageiros pagos no período | {anual.sum():,.0f} |
| Queda de 2019 para 2020 (COVID) | {resumo["queda_2020"]:.1%} |
| Nível de {ano} em relação a 2019 | {resumo["nivel_vs_2019"]:.1%} |
| Participação dos voos regulares | {resumo["participacao_regulares"]:.1%} |

## Passageiros por grupo de voo
{grupos.rename("participacao").map(lambda v: f"{v:.2%}").to_frame().to_markdown()}

## Valores ausentes por coluna
{ausentes.rename("ausentes").map(lambda v: f"{v:.1%}").to_frame().to_markdown()}
"""
    (cfg.dir_relatorios / "eda.md").write_text(texto, encoding="utf-8")
    return resumo
