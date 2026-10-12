"""Métricas de erro de previsão."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def wape(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Weighted Absolute Percentage Error: soma dos erros absolutos / soma dos valores reais."""
    y_true = np.asarray(y_true, dtype=float)
    total = y_true.sum()
    if total <= 0:
        raise ValueError("WAPE indefinida: a soma dos valores reais é zero.")
    return float(np.abs(np.asarray(y_pred, dtype=float) - y_true).sum() / total)


def calcular_metricas(y_true: ArrayLike, y_pred: ArrayLike) -> dict[str, float]:
    """WAPE (principal), MAE, RMSE e viés (positivo = superestima)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 0, None)
    erro = y_pred - y_true
    return {
        "WAPE": wape(y_true, y_pred),
        "MAE": float(np.abs(erro).mean()),
        "RMSE": float(np.sqrt((erro**2).mean())),
        "vies": float(erro.sum() / y_true.sum()),
    }
