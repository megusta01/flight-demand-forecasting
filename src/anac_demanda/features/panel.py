"""Painel rota-mês e engenharia de features.

Cada série é uma rota operada por uma empresa (empresa + origem + destino). Para cada mês t em que
a rota teve passageiros, a linha do painel descreve a rota até o fim de t e o alvo é o total de
passageiros pagos em t+1.

Regra contra vazamento: toda feature usa apenas informação disponível até o fim do mês t; do mês
previsto entra só o calendário (mês do ano, flag de COVID). `tests/test_features.py` verifica isso.

Implementação: as séries são pivotadas em matrizes [rota x mês] (NumPy), o que torna defasagens e
janelas móveis operações vetorizadas: o painel completo (~600 mil linhas) é montado em segundos.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import Config
from ..logs import get_logger
from ..periodos import Mes

log = get_logger(__name__)

CHAVES_SERIE = ["empresa_sigla", "aeroporto_de_origem_sigla", "aeroporto_de_destino_sigla"]

FEATURES_CATEGORICAS = [
    "empresa_sigla",
    "empresa_nacionalidade",
    "natureza",
    "aeroporto_de_origem_sigla",
    "aeroporto_de_destino_sigla",
    "aeroporto_de_origem_regiao",
    "aeroporto_de_destino_regiao",
]

FEATURES_NUMERICAS = [
    # calendário do mês previsto
    "mes_alvo",
    "covid",
    # histórico de demanda da rota
    "pax_lag0",
    "pax_lag1",
    "pax_lag2",
    "pax_lag11",
    "pax_lag12",
    "pax_lag23",
    "pax_media_3m",
    "pax_media_12m",
    "crescimento_aa",
    "meses_ativos_12m",
    "idade_meses",
    # histórico relativo ao mês atual (ajuda a prever a variação t -> t+1)
    "rel_pax_lag1",
    "rel_pax_lag2",
    "rel_pax_lag11",
    "rel_pax_lag12",
    "rel_pax_lag23",
    "rel_pax_media_3m",
    "rel_pax_media_12m",
    "rel_assentos_lag0",
    "rel_assentos_lag11",
    "variacao_sazonal_1a",
    "variacao_sazonal_2a",
    # oferta já realizada
    "assentos_lag0",
    "decolagens_lag0",
    "assentos_lag11",
    "ocupacao_lag0",
    "distancia_por_voo_km",
    # mercado
    "mercado_lag0",
    "mercado_crescimento_aa",
]

FEATURES = FEATURES_NUMERICAS + FEATURES_CATEGORICAS


# ---------------------------------------------------------------- utilitários vetorizados
def _lag(mat: np.ndarray, rows: np.ndarray, cols: np.ndarray, k: int) -> np.ndarray:
    """mat[s, t-k]; zero antes do início da base."""
    c = cols - k
    out = np.zeros(len(rows), dtype=float)
    ok = c >= 0
    out[ok] = mat[rows[ok], c[ok]]
    return out


def _media_janela(acum: np.ndarray, rows: np.ndarray, cols: np.ndarray, atraso: int, tamanho: int) -> np.ndarray:
    """Média de mat[s, t-atraso-tamanho+1 .. t-atraso] via soma acumulada."""
    hi = cols - atraso
    lo = hi - tamanho
    v_hi = np.where(hi >= 0, acum[rows, np.clip(hi, 0, None)], 0.0)
    v_lo = np.where(lo >= 0, acum[rows, np.clip(lo, 0, None)], 0.0)
    return (v_hi - v_lo) / tamanho


def _razao(num, den) -> np.ndarray:
    num, den = np.asarray(num, dtype=float), np.asarray(den, dtype=float)
    return np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)


def _ffill(mat: np.ndarray) -> np.ndarray:
    """Propaga para frente o último valor não-NaN ao longo do tempo (nunca para trás)."""
    idx = np.where(~np.isnan(mat), np.arange(mat.shape[1]), 0)
    np.maximum.accumulate(idx, axis=1, out=idx)
    return mat[np.arange(mat.shape[0])[:, None], idx]


# ---------------------------------------------------------------- estrutura do painel
@dataclass
class Matrizes:
    """Séries mensais [rota x mês] usadas para calcular as features."""

    pax: np.ndarray
    assentos: np.ndarray
    decolagens: np.ndarray
    dist_por_voo: np.ndarray
    primeiro_mes: np.ndarray
    estaticos: pd.DataFrame
    ano_base: int
    ultimo_mes: Mes


def montar_matrizes(df: pd.DataFrame, cfg: Config) -> Matrizes:
    reg = df[df["grupo_de_voo"] == cfg.grupo_de_voo].dropna(subset=CHAVES_SERIE).copy()
    ano_base = cfg.ano_inicial
    reg["t"] = (reg["ano"] - ano_base) * 12 + (reg["mes"] - 1)
    ultimo = Mes(int(reg["ano"].max()), int(reg.loc[reg["ano"] == reg["ano"].max(), "mes"].max()))
    n_meses = ultimo.indice(ano_base) + 1

    estaticos_cols = [c for c in FEATURES_CATEGORICAS if c not in CHAVES_SERIE]
    reg = reg.sort_values("t")
    estaticos = reg.groupby(CHAVES_SERIE, sort=True)[estaticos_cols].first().reset_index()
    estaticos["serie_id"] = np.arange(len(estaticos))
    reg = reg.merge(estaticos[[*CHAVES_SERIE, "serie_id"]], on=CHAVES_SERIE, how="left")

    s, t = reg["serie_id"].to_numpy(), reg["t"].to_numpy()

    def larga(coluna: str) -> np.ndarray:
        mat = np.zeros((len(estaticos), n_meses))
        np.add.at(mat, (s, t), reg[coluna].fillna(0).to_numpy(dtype=float))
        return mat

    pax, assentos, decolagens, dist = (
        larga("passageiros_pagos"),
        larga("assentos"),
        larga("decolagens"),
        larga("distancia_voada_km"),
    )
    dist_por_voo = _ffill(np.where((decolagens > 0) & (dist > 0), _razao(dist, decolagens), np.nan))

    registro = np.zeros_like(pax, dtype=bool)
    registro[s, t] = True
    return Matrizes(pax, assentos, decolagens, dist_por_voo, registro.argmax(axis=1), estaticos, ano_base, ultimo)


def _features(m: Matrizes, rows: np.ndarray, cols: np.ndarray, cfg: Config) -> pd.DataFrame:
    pax = m.pax
    out = pd.DataFrame({"serie_id": rows, "t_origem": cols})
    t_alvo = cols + 1
    out["ano_alvo"] = m.ano_base + t_alvo // 12
    out["mes_alvo"] = t_alvo % 12 + 1

    covid_ini, covid_fim = (x.indice(m.ano_base) for x in cfg.covid)
    out["covid"] = ((t_alvo >= covid_ini) & (t_alvo <= covid_fim)).astype(int)

    # histórico de demanda
    for k in (0, 1, 2, 11, 12, 23):
        out[f"pax_lag{k}"] = _lag(pax, rows, cols, k)
    acum = pax.cumsum(axis=1)
    out["pax_media_3m"] = _media_janela(acum, rows, cols, 0, 3)
    out["pax_media_12m"] = _media_janela(acum, rows, cols, 0, 12)
    out["crescimento_aa"] = _razao(out["pax_media_3m"], _media_janela(acum, rows, cols, 12, 3))
    ativos = np.cumsum(pax > 0, axis=1)
    out["meses_ativos_12m"] = ativos[rows, cols] - np.where(cols >= 12, ativos[rows, np.clip(cols - 12, 0, None)], 0)
    out["idade_meses"] = cols - m.primeiro_mes[rows]

    # sazonalidade da própria rota: variação entre os mesmos dois meses há 1 e 2 anos
    out["variacao_sazonal_1a"] = _razao(out["pax_lag11"], out["pax_lag12"])
    out["variacao_sazonal_2a"] = _razao(out["pax_lag23"], _lag(pax, rows, cols, 24))

    # oferta já realizada
    out["assentos_lag0"] = m.assentos[rows, cols]
    out["assentos_lag11"] = _lag(m.assentos, rows, cols, 11)
    out["decolagens_lag0"] = m.decolagens[rows, cols]
    out["ocupacao_lag0"] = np.clip(_razao(out["pax_lag0"], out["assentos_lag0"]), 0, 2)
    out["distancia_por_voo_km"] = m.dist_por_voo[rows, cols]

    # relativo ao mês atual
    base = out["pax_lag0"].to_numpy()
    for col in (
        "pax_lag1",
        "pax_lag2",
        "pax_lag11",
        "pax_lag12",
        "pax_lag23",
        "pax_media_3m",
        "pax_media_12m",
        "assentos_lag0",
        "assentos_lag11",
    ):
        out[f"rel_{col}"] = _razao(out[col], base)

    # mercado (todas as rotas regulares)
    mercado = pax.sum(axis=0)
    out["mercado_lag0"] = mercado[cols]
    out["mercado_crescimento_aa"] = _razao(
        mercado[cols], np.where(cols >= 12, mercado[np.clip(cols - 12, 0, None)], np.nan)
    )

    out = out.merge(m.estaticos, on="serie_id", how="left")
    for col in FEATURES_CATEGORICAS:
        out[col] = out[col].astype("string").fillna("DESCONHECIDO")
    return out


def construir_painel(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Painel com uma linha por (rota, mês t) com passageiros em t.

    - Linhas com t+1 dentro da base têm `alvo` = passageiros em t+1 (zero se a rota parou).
    - Linhas do último mês da base têm `alvo` = NaN: são as previsões "de produção" para o mês seguinte.
    """
    m = montar_matrizes(df, cfg)
    rows, cols = np.nonzero(m.pax > 0)
    painel = _features(m, rows, cols, cfg)

    t_alvo = cols + 1
    alvo = np.full(len(rows), np.nan)
    dentro = t_alvo < m.pax.shape[1]
    alvo[dentro] = m.pax[rows[dentro], t_alvo[dentro]]
    painel["alvo"] = alvo

    painel.attrs["ultimo_mes"] = str(m.ultimo_mes)
    log.info(
        "Painel: %s linhas, %s rotas, dados até %s",
        f"{len(painel):,}",
        f"{painel['serie_id'].nunique():,}",
        m.ultimo_mes.rotulo,
    )
    return painel.sort_values(["t_origem", "serie_id"]).reset_index(drop=True)


def carregar_painel(cfg: Config, df: pd.DataFrame | None = None, recriar: bool = False) -> pd.DataFrame:
    if cfg.cache_painel.exists() and not recriar:
        return pd.read_parquet(cfg.cache_painel)
    if df is None:
        from ..data import carregar_dados

        df = carregar_dados(cfg)
    painel = construir_painel(df, cfg)
    cfg.dir_cache.mkdir(parents=True, exist_ok=True)
    painel.to_parquet(cfg.cache_painel, index=False)
    return painel


def ultimo_mes_do_painel(painel: pd.DataFrame, ano_base: int) -> Mes:
    """Mês mais recente com dados (origem das previsões de produção)."""
    return Mes.de_indice(int(painel["t_origem"].max()), ano_base)


def historico_mensal(df: pd.DataFrame, cfg: Config, desde_ano: int = 2019) -> pd.DataFrame:
    """Passageiros por rota e mês (formato longo), usado pelo painel interativo."""
    reg = df[(df["grupo_de_voo"] == cfg.grupo_de_voo) & (df["ano"] >= desde_ano)].dropna(subset=CHAVES_SERIE)
    return (
        reg.groupby([*CHAVES_SERIE, "ano", "mes"], as_index=False)["passageiros_pagos"]
        .sum()
        .rename(columns={"passageiros_pagos": "passageiros"})
    )
