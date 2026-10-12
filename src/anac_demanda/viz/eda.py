"""Gráficos da análise exploratória."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..periodos import NOMES_MESES
from .style import CATEGORICAS, CORES, MILHAR, PERCENTUAL, aplicar_estilo, salvar, subtitulo

NOMES_EMPRESAS = {"TAM": "LATAM", "GLO": "GOL", "AZU": "Azul"}


def passageiros_por_ano(df: pd.DataFrame, destino: Path, ultimo_ano_completo: int) -> Path:
    aplicar_estilo()
    anual = df[df["ano"] <= ultimo_ano_completo].groupby("ano")["passageiros_pagos"].sum()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(anual.index, anual.values, color=CORES["azul"], marker="o", markersize=4)
    ax.axvspan(2019.5, 2021.5, color=CORES["faixa"], linewidth=0)
    ax.text(2020.5, anual.max() * 0.98, "COVID-19", ha="center", va="top", fontsize=9, color=CORES["texto2"])
    ax.yaxis.set_major_formatter(MILHAR)
    ax.set_ylabel("Passageiros pagos")
    ax.set_title("Passageiros pagos por ano")
    subtitulo(ax, "Voos domésticos e internacionais com origem ou destino no Brasil")
    return salvar(fig, destino, "eda_passageiros_por_ano.png")


def sazonalidade(df: pd.DataFrame, destino: Path, ultimo_ano_completo: int) -> Path:
    aplicar_estilo()
    m = df[df["ano"] <= ultimo_ano_completo].groupby(["ano", "mes"])["passageiros_pagos"].sum().unstack()
    indice = m.div(m.mean(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(indice.values, cmap="RdBu_r", vmin=0.6, vmax=1.4, aspect="auto")
    ax.set_xticks(range(12), NOMES_MESES)
    ax.set_yticks(range(len(indice)), indice.index)
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.8).set_label("Passageiros do mês / média do ano")
    ax.set_title("Sazonalidade: picos em janeiro, julho e dezembro")
    subtitulo(ax, "Azul = abaixo da média do ano; vermelho = acima. A quebra de 2020 é a COVID")
    return salvar(fig, destino, "eda_sazonalidade.png")


def principais_rotas(df: pd.DataFrame, destino: Path, ano: int, grupo: str) -> Path:
    aplicar_estilo()
    d = df[(df["ano"] == ano) & (df["grupo_de_voo"] == grupo)].dropna(
        subset=["aeroporto_de_origem_sigla", "aeroporto_de_destino_sigla"]
    )
    o, de = d["aeroporto_de_origem_sigla"], d["aeroporto_de_destino_sigla"]
    par = np.where(o < de, o + " ↔ " + de, de + " ↔ " + o)
    top = d.assign(par=par).groupby("par")["passageiros_pagos"].sum().nlargest(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.barh(top.index, top.values, color=CORES["azul"], height=0.6)
    ax.xaxis.set_major_formatter(MILHAR)
    ax.grid(axis="y", visible=False)
    ax.set_title(f"Pares de aeroportos mais movimentados em {ano}")
    subtitulo(ax, "Passageiros pagos em voos regulares, somando os dois sentidos")
    return salvar(fig, destino, "eda_top_rotas.png")


def participacao_empresas(df: pd.DataFrame, destino: Path, ultimo_ano_completo: int, grupo: str) -> Path:
    aplicar_estilo()
    d = df[(df["natureza"] == "DOMÉSTICA") & (df["grupo_de_voo"] == grupo) & (df["ano"] <= ultimo_ano_completo)]
    fatia = d.pivot_table(index="ano", columns="empresa_sigla", values="passageiros_pagos", aggfunc="sum")
    fatia = fatia.div(fatia.sum(axis=1), axis=0)
    top = fatia.iloc[-1].nlargest(3).index.tolist()

    # rótulos diretos no fim das linhas, afastados para não colidir
    finais = sorted(((fatia[e].iloc[-1], e) for e in top), reverse=True)
    y_rotulo, anterior = {}, None
    for valor, e in finais:
        y = valor if anterior is None else min(valor, anterior - 0.035)
        y_rotulo[e], anterior = y, y

    fig, ax = plt.subplots(figsize=(10, 4.2))
    x_fim = fatia.index[-1]
    for cor, e in zip(CATEGORICAS, top, strict=False):
        ax.plot(fatia.index, fatia[e], color=cor)
        ax.text(x_fim + 0.3, y_rotulo[e], NOMES_EMPRESAS.get(e, e), va="center", fontsize=9.5)
    demais = 1 - fatia[top].sum(axis=1)
    ax.plot(fatia.index, demais, color=CORES["cinza"], linestyle="--")
    ax.text(x_fim + 0.3, demais.iloc[-1], "Demais", color=CORES["texto2"], va="center", fontsize=9.5)
    ax.yaxis.set_major_formatter(PERCENTUAL)
    ax.set_xlim(fatia.index[0], x_fim + 3)
    ax.set_title("Participação no mercado doméstico regular")
    subtitulo(ax, "Fatia dos passageiros pagos por empresa")
    return salvar(fig, destino, "eda_participacao_empresas.png")
