"""Gráficos de resultados do modelo."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .style import CORES, MILHAR, PERCENTUAL, aplicar_estilo, eixo_mensal, pct, salvar, subtitulo


def _datas(ano, mes) -> pd.Series:
    return pd.to_datetime(pd.DataFrame({"year": ano, "month": mes, "day": 1}))


def comparacao_teste(tabela: pd.DataFrame, destino: Path) -> Path:
    aplicar_estilo()
    t = tabela.sort_values("WAPE")
    fig, ax = plt.subplots(figsize=(9, 4))
    cores = [CORES["azul"] if tipo == "modelo" else CORES["cinza"] for tipo in t["tipo"]]
    ax.barh(t["modelo"], t["WAPE"], color=cores, height=0.6)
    ax.invert_yaxis()
    for i, v in enumerate(t["WAPE"]):
        ax.text(v + 0.003, i, pct(v), va="center", fontsize=10)
    ax.xaxis.set_major_formatter(PERCENTUAL)
    ax.set_xlabel("WAPE (erro absoluto / total de passageiros) — menor é melhor")
    ax.grid(axis="y", visible=False)
    ax.set_title("Erro de previsão no teste (2024–2025)")
    subtitulo(ax, "Modelo em azul; baselines em cinza")
    return salvar(fig, destino, "resultado_teste.png")


def serie_total(
    mensal: pd.DataFrame,
    inicio_monitoramento: pd.Timestamp | None,
    previsao_producao: tuple[pd.Timestamp, float] | None,
    nome_base: str,
    destino: Path,
) -> Path:
    """Total nacional real vs. soma das previsões (teste + monitoramento) e a previsão do próximo mês."""
    aplicar_estilo()
    x = _datas(mensal["ano"], mensal["mes"])
    fig, ax = plt.subplots(figsize=(10.5, 4.4))
    if inicio_monitoramento is not None:
        ax.axvspan(
            inicio_monitoramento - pd.Timedelta(days=15),
            x.max() + pd.Timedelta(days=45),
            color=CORES["faixa"],
            linewidth=0,
        )
        ax.text(
            inicio_monitoramento,
            0.04,
            " monitoramento\n (dados novos)",
            transform=ax.get_xaxis_transform(),
            fontsize=9,
            color=CORES["texto2"],
            va="bottom",
        )
    ax.plot(x, mensal["passageiros"], color=CORES["texto"], label="Real", linewidth=2.2)
    ax.plot(x, mensal["previsto_modelo"], color=CORES["azul"], label="Modelo")
    ax.plot(x, mensal["previsto_baseline"], color=CORES["laranja"], label=nome_base, linewidth=1.5, linestyle="--")
    if previsao_producao is not None:
        data, valor = previsao_producao
        ax.plot(
            [x.iloc[-1], data],
            [mensal["previsto_modelo"].iloc[-1], valor],
            color=CORES["azul"],
            linestyle=":",
            linewidth=1.5,
        )
        ax.scatter([data], [valor], color=CORES["azul"], s=60, zorder=5, edgecolor="white", linewidth=1.5)
        ax.annotate(
            "previsão",
            (data, valor),
            xytext=(0, -18),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            color=CORES["azul"],
        )
    eixo_mensal(ax, 3)
    ax.yaxis.set_major_formatter(MILHAR)
    ax.set_ylabel("Passageiros pagos (voos regulares)")
    ax.legend(loc="upper left", ncols=3)
    ax.set_title("Soma das previsões por rota vs. total real")
    subtitulo(ax, "Previsão um mês à frente; teste em 2024–2025, depois dados publicados após o congelamento do modelo")
    return salvar(fig, destino, "serie_total.png")


def rotas_exemplo(
    df: pd.DataFrame, pred_modelo: np.ndarray, pred_base: np.ndarray, nome_base: str, destino: Path, n: int = 4
) -> Path:
    aplicar_estilo()
    d = df.assign(modelo=pred_modelo, baseline=pred_base)
    d["rota"] = d["empresa_sigla"] + "  " + d["aeroporto_de_origem_sigla"] + " → " + d["aeroporto_de_destino_sigla"]
    top = d.groupby("rota")["alvo"].sum().nlargest(n).index
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5), sharex=True)
    for ax, rota in zip(axes.flat, top, strict=False):
        r = d[d["rota"] == rota].sort_values(["ano_alvo", "mes_alvo"])
        x = _datas(r["ano_alvo"], r["mes_alvo"])
        ax.plot(x, r["alvo"], color=CORES["texto"], label="Real")
        ax.plot(x, r["modelo"], color=CORES["azul"], label="Modelo")
        ax.plot(x, r["baseline"], color=CORES["laranja"], label=nome_base, linestyle="--", linewidth=1.4)
        ax.set_title(rota, fontsize=11, pad=8)
        ax.yaxis.set_major_formatter(MILHAR)
        eixo_mensal(ax, 4)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.suptitle("Rotas com mais passageiros no teste", x=0.01, ha="left", fontweight="bold", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.legend(handles, labels, loc="upper right", ncols=3, bbox_to_anchor=(0.99, 1.0))
    return salvar(fig, destino, "rotas_exemplo.png")


def importancia(imp: pd.DataFrame, destino: Path, top: int = 12) -> Path:
    aplicar_estilo()
    # pax_lag0 é a base multiplicada pela variação prevista: embaralhá-la destrói a previsão inteira.
    d = imp[imp["feature"] != "pax_lag0"].head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.barh(d["feature"], d["aumento_mae_pct"], color=CORES["azul"], height=0.6)
    ax.xaxis.set_major_formatter(PERCENTUAL)
    ax.set_xlabel("Aumento do erro (MAE) ao embaralhar a variável")
    ax.grid(axis="y", visible=False)
    ax.set_title("O que explica a variação de um mês para o outro")
    subtitulo(ax, "Importância por permutação na validação, sem o nível atual da rota (pax_lag0)")
    return salvar(fig, destino, "importancia_features.png")


def erro_por_porte(tabela: pd.DataFrame, nome_base: str, destino: Path) -> Path:
    aplicar_estilo()
    fig, ax = plt.subplots(figsize=(8.5, 4))
    x, w = np.arange(len(tabela)), 0.38
    ax.bar(x - w / 2 - 0.01, tabela["WAPE_modelo"], w, color=CORES["azul"], label="Modelo")
    ax.bar(x + w / 2 + 0.01, tabela["WAPE_baseline"], w, color=CORES["cinza"], label=nome_base)
    ax.set_xticks(x, tabela["porte"])
    ax.yaxis.set_major_formatter(PERCENTUAL)
    ax.set_xlabel("Passageiros da rota no mês atual")
    ax.set_ylabel("WAPE")
    ax.grid(axis="x", visible=False)
    ax.legend()
    ax.set_title("Erro por porte da rota (teste)")
    subtitulo(ax, "Rotas pequenas são mais voláteis; o modelo ganha do baseline em todos os portes")
    return salvar(fig, destino, "erro_por_porte.png")
