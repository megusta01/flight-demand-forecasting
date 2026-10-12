"""Baselines: regras simples que qualquer modelo precisa superar."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

Baseline = Callable[[pd.DataFrame], np.ndarray]


def ingenuo(X: pd.DataFrame) -> np.ndarray:
    """Mês que vem = este mês."""
    return X["pax_lag0"].to_numpy(dtype=float)


def sazonal(X: pd.DataFrame) -> np.ndarray:
    """Mês que vem = mesmo mês do ano anterior (cai para o ingênuo se a rota não existia)."""
    s = X["pax_lag11"].to_numpy(dtype=float)
    return np.where(s > 0, s, ingenuo(X))


def sazonal_com_crescimento(X: pd.DataFrame) -> np.ndarray:
    """Mesmo mês do ano anterior x crescimento recente (3 meses vs. mesmos 3 meses do ano anterior)."""
    g = np.clip(np.nan_to_num(X["crescimento_aa"].to_numpy(dtype=float), nan=1.0), 0.5, 2.0)
    s = X["pax_lag11"].to_numpy(dtype=float)
    return np.where(s > 0, s * g, ingenuo(X))


def ingenuo_com_ajuste_sazonal(X: pd.DataFrame) -> np.ndarray:
    """Este mês x variação que a rota teve entre esses mesmos dois meses no ano anterior."""
    r = np.clip(np.nan_to_num(X["variacao_sazonal_1a"].to_numpy(dtype=float), nan=1.0), 0.5, 2.0)
    return ingenuo(X) * r


BASELINES: dict[str, Baseline] = {
    "Ingênuo (repete o mês atual)": ingenuo,
    "Ingênuo com ajuste sazonal": ingenuo_com_ajuste_sazonal,
    "Sazonal x crescimento recente": sazonal_com_crescimento,
    "Sazonal (repete o mesmo mês do ano anterior)": sazonal,
}
