"""Modelo de previsão: pré-processamento + estimador + transformação do alvo, com persistência."""

from __future__ import annotations

import json
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler

from .. import __version__
from ..config import Config, ModeloSpec
from ..features import FEATURES, FEATURES_CATEGORICAS, FEATURES_NUMERICAS

LIMITE_VARIACAO = 5.0  # variação máxima prevista: 5x o mês atual

# Features de contagem (assimétricas) que recebem log1p no modelo linear.
_FEATURES_LOG = [
    "pax_lag0",
    "pax_lag1",
    "pax_lag2",
    "pax_lag11",
    "pax_lag12",
    "pax_lag23",
    "pax_media_3m",
    "pax_media_12m",
    "assentos_lag0",
    "decolagens_lag0",
    "assentos_lag11",
    "mercado_lag0",
    "crescimento_aa",
    "distancia_por_voo_km",
]


class AgrupadorRaras(BaseEstimator, TransformerMixin):
    """Troca categorias pouco frequentes no treino por 'OUTROS' (inclusive categorias novas)."""

    def __init__(self, colunas: list[str], min_ocorrencias: int = 500, max_categorias: int = 200):
        self.colunas = colunas
        self.min_ocorrencias = min_ocorrencias
        self.max_categorias = max_categorias

    def fit(self, X: pd.DataFrame, y=None):
        self.manter_ = {}
        for col in self.colunas:
            cont = X[col].value_counts()
            cont = cont[cont >= self.min_ocorrencias].head(self.max_categorias - 1)
            self.manter_[col] = set(cont.index)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col, manter in self.manter_.items():
            X[col] = X[col].where(X[col].isin(manter), "OUTROS").astype(str)
        return X


def _log1p_positivo(X):
    return np.log1p(np.clip(X, 0, None))


def _estimador(spec: ModeloSpec, semente: int):
    if spec.tipo == "ridge":
        outras = [c for c in FEATURES_NUMERICAS if c not in _FEATURES_LOG]
        pre = ColumnTransformer(
            [
                (
                    "log",
                    Pipeline(
                        [
                            ("imp", SimpleImputer(strategy="median")),
                            ("log", FunctionTransformer(_log1p_positivo)),
                            ("esc", StandardScaler()),
                        ]
                    ),
                    _FEATURES_LOG,
                ),
                ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("esc", StandardScaler())]), outras),
                ("cat", OneHotEncoder(handle_unknown="ignore"), FEATURES_CATEGORICAS),
            ]
        )
        return pre, Ridge(**spec.params)

    pre = ColumnTransformer(
        [
            ("num", "passthrough", FEATURES_NUMERICAS),
            (
                "cat",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-1),
                FEATURES_CATEGORICAS,
            ),
        ],
        verbose_feature_names_out=False,
    )
    mascara_cat = [False] * len(FEATURES_NUMERICAS) + [True] * len(FEATURES_CATEGORICAS)
    params = {"l2_regularization": 1.0, "early_stopping": False, **spec.params}
    return pre, HistGradientBoostingRegressor(categorical_features=mascara_cat, random_state=semente, **params)


class ModeloPrevisao:
    """Prevê passageiros do mês seguinte por rota.

    Alvos suportados:
      - variacao: treina em r = pax[t+1] / pax[t] com peso pax[t] e prevê pax[t] * r.
                  Com perda absoluta, o treino minimiza exatamente a WAPE.
      - log:      treina em log1p(pax[t+1]) e prevê expm1(...).
      - contagem: treina direto em pax[t+1] (ex.: perda Poisson).
    """

    def __init__(self, spec: ModeloSpec, cfg: Config):
        self.spec = spec
        pre, est = _estimador(spec, cfg.semente)
        self.pipeline = Pipeline(
            [
                ("raras", AgrupadorRaras(FEATURES_CATEGORICAS, cfg.min_ocorrencias_categoria, cfg.max_categorias)),
                ("pre", pre),
                ("modelo", est),
            ]
        )
        self.metadados: dict[str, Any] = {}

    @property
    def nome(self) -> str:
        return self.spec.nome

    def fit(self, X: pd.DataFrame, y) -> ModeloPrevisao:
        X = X[FEATURES]
        y = np.asarray(y, dtype=float)
        if self.spec.alvo == "variacao":
            base = X["pax_lag0"].to_numpy(dtype=float)
            self.pipeline.fit(X, np.clip(y / base, 0, LIMITE_VARIACAO), modelo__sample_weight=base)
        elif self.spec.alvo == "log":
            self.pipeline.fit(X, np.log1p(y))
        else:
            self.pipeline.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X = X[FEATURES]
        p = self.pipeline.predict(X)
        if self.spec.alvo == "variacao":
            p = X["pax_lag0"].to_numpy(dtype=float) * np.clip(p, 0, None)
        elif self.spec.alvo == "log":
            p = np.expm1(p)
        return np.clip(p, 0, None)

    def clonar(self) -> ModeloPrevisao:
        return deepcopy(self)

    # ------------------------------------------------------------ persistência
    def salvar(self, diretorio: Path, nome_arquivo: str = "modelo") -> Path:
        diretorio.mkdir(parents=True, exist_ok=True)
        caminho = diretorio / f"{nome_arquivo}.joblib"
        self.metadados.update(
            {
                "modelo": self.spec.nome,
                "spec": {"tipo": self.spec.tipo, "alvo": self.spec.alvo, "params": self.spec.params},
                "features": FEATURES,
                "versao_pacote": __version__,
                "salvo_em": datetime.now(UTC).isoformat(timespec="seconds"),
            }
        )
        joblib.dump(self, caminho, compress=3)
        (diretorio / f"{nome_arquivo}.json").write_text(
            json.dumps(self.metadados, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )
        return caminho

    @staticmethod
    def carregar(caminho: Path) -> ModeloPrevisao:
        if not caminho.exists():
            raise FileNotFoundError(f"Modelo não encontrado em {caminho}. Rode 'anac-demanda treinar' antes.")
        modelo = joblib.load(caminho)
        if not isinstance(modelo, ModeloPrevisao):
            raise TypeError(f"{caminho} não contém um ModeloPrevisao.")
        return modelo


def construir_modelos(cfg: Config) -> list[ModeloPrevisao]:
    """Instancia todos os modelos candidatos definidos na configuração."""
    return [ModeloPrevisao(spec, cfg) for spec in cfg.modelos]
