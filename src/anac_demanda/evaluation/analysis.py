"""Análises de erro: por porte da rota, por mês e importância das variáveis."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..features import FEATURES
from ..models import ModeloPrevisao
from .metrics import calcular_metricas

FAIXAS_PORTE = [0, 1_000, 5_000, 20_000, np.inf]
ROTULOS_PORTE = ["até 1 mil", "1 a 5 mil", "5 a 20 mil", "mais de 20 mil"]


def erro_por_porte(df: pd.DataFrame, pred_modelo: np.ndarray, pred_base: np.ndarray) -> pd.DataFrame:
    """WAPE do modelo e do baseline por faixa de passageiros da rota no mês atual."""
    faixa = pd.cut(df["pax_lag0"], bins=FAIXAS_PORTE, labels=ROTULOS_PORTE, right=False)
    y = df["alvo"].to_numpy()
    linhas = []
    for rotulo in ROTULOS_PORTE:
        m = (faixa == rotulo).to_numpy()
        if not m.any():
            continue
        linhas.append(
            {
                "porte": rotulo,
                "amostras": int(m.sum()),
                "participacao": float(y[m].sum() / y.sum()),
                "WAPE_modelo": calcular_metricas(y[m], pred_modelo[m])["WAPE"],
                "WAPE_baseline": calcular_metricas(y[m], pred_base[m])["WAPE"],
            }
        )
    return pd.DataFrame(linhas)


def erro_por_mes(df: pd.DataFrame, pred_modelo: np.ndarray, pred_base: np.ndarray) -> pd.DataFrame:
    d = df[["ano_alvo", "mes_alvo", "alvo"]].assign(modelo=pred_modelo, baseline=pred_base)
    linhas = []
    for (ano, mes), g in d.groupby(["ano_alvo", "mes_alvo"]):
        linhas.append(
            {
                "ano": int(ano),
                "mes": int(mes),
                "passageiros": float(g["alvo"].sum()),
                "previsto_modelo": float(g["modelo"].sum()),
                "previsto_baseline": float(g["baseline"].sum()),
                "WAPE_modelo": calcular_metricas(g["alvo"], g["modelo"])["WAPE"],
                "WAPE_baseline": calcular_metricas(g["alvo"], g["baseline"])["WAPE"],
            }
        )
    return pd.DataFrame(linhas)


def importancia_permutacao(
    modelo: ModeloPrevisao, df: pd.DataFrame, semente: int, n_amostras: int = 20_000, repeticoes: int = 3
) -> pd.DataFrame:
    """Quanto o MAE da previsão final piora ao embaralhar cada feature."""
    rng = np.random.default_rng(semente)
    amostra = df.sample(min(n_amostras, len(df)), random_state=semente).reset_index(drop=True)
    X, y = amostra[FEATURES], amostra["alvo"].to_numpy()
    mae_base = calcular_metricas(y, modelo.predict(X))["MAE"]
    linhas = []
    for feat in FEATURES:
        aumentos = []
        for _ in range(repeticoes):
            Xp = X.copy()
            Xp[feat] = Xp[feat].to_numpy()[rng.permutation(len(Xp))]
            aumentos.append(calcular_metricas(y, modelo.predict(Xp))["MAE"] - mae_base)
        linhas.append({"feature": feat, "aumento_mae": float(np.mean(aumentos))})
    return (
        pd.DataFrame(linhas)
        .assign(aumento_mae_pct=lambda d: d["aumento_mae"] / mae_base)
        .sort_values("aumento_mae", ascending=False)
        .reset_index(drop=True)
    )
