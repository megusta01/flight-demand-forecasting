"""Leitura do CSV da ANAC e cache limpo em Parquet."""

from __future__ import annotations

import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Config
from ..logs import cronometro, get_logger
from .schema import COLUNAS_NUMERICAS, COLUNAS_OBRIGATORIAS, COLUNAS_TEXTO, validar_colunas, validar_dados

log = get_logger(__name__)

URL_DADOS = (
    "https://www.anac.gov.br/acesso-a-informacao/dados-abertos/areas-de-atuacao/"
    "voos-e-operacoes-aereas/dados-estatisticos-do-transporte-aereo"
)


def encontrar_csv(diretorio: Path) -> Path:
    arquivos = sorted(diretorio.glob("*.csv"))
    if not arquivos:
        raise FileNotFoundError(
            f"Nenhum CSV encontrado em {diretorio}. Baixe 'Dados_Estatisticos.csv' em {URL_DADOS} "
            "e salve nessa pasta (detalhes em data/README.md)."
        )
    if len(arquivos) > 1:
        log.warning("Mais de um CSV em %s; usando %s", diretorio, arquivos[0].name)
    return arquivos[0]


def normalizar_nome_coluna(nome: str) -> str:
    ascii_ = unicodedata.normalize("NFKD", str(nome)).encode("ascii", "ignore").decode("ascii")
    return ascii_.strip().lower().replace(" ", "_")


def _detectar_formato(caminho: Path) -> dict:
    """O arquivo da ANAC traz uma linha 'Atualizado em: ...' antes do cabeçalho."""
    for encoding in ("utf-8-sig", "latin1"):
        try:
            with caminho.open("r", encoding=encoding) as fh:
                for i, linha in enumerate(fh):
                    if "EMPRESA_SIGLA" in linha.upper():
                        sep = max([";", ",", "\t"], key=linha.count)
                        return {"encoding": encoding, "sep": sep, "skiprows": i}
                    if i > 20:
                        break
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Cabeçalho com EMPRESA_SIGLA não encontrado em {caminho}.")


def _para_numero(serie: pd.Series) -> pd.Series:
    """Converte '1.234,5' -> 1234.5; vazio -> NaN."""
    texto = serie.astype("string").str.strip().replace("", pd.NA)
    texto = texto.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    return pd.to_numeric(texto, errors="coerce").astype("float64")


def ler_csv_anac(caminho: Path) -> pd.DataFrame:
    fmt = _detectar_formato(caminho)
    cabecalho = pd.read_csv(caminho, sep=fmt["sep"], encoding=fmt["encoding"], skiprows=fmt["skiprows"], nrows=0)
    mapa = {c: normalizar_nome_coluna(c) for c in cabecalho.columns}
    validar_colunas(list(mapa.values()))
    usar = [c for c, n in mapa.items() if n in COLUNAS_OBRIGATORIAS]

    df = pd.read_csv(
        caminho,
        sep=fmt["sep"],
        encoding=fmt["encoding"],
        skiprows=fmt["skiprows"],
        usecols=usar,
        dtype=str,
        keep_default_na=False,
    )
    df = df.rename(columns=mapa)
    for col in COLUNAS_NUMERICAS:
        df[col] = _para_numero(df[col])
    for col in COLUNAS_TEXTO:
        df[col] = df[col].replace("", np.nan).astype("string")
    df = df.dropna(subset=["ano", "mes"])
    df["ano"] = df["ano"].astype(int)
    df["mes"] = df["mes"].astype(int)
    return df[COLUNAS_OBRIGATORIAS]


def carregar_dados(cfg: Config, recriar: bool = False) -> pd.DataFrame:
    """Base limpa a partir de `ano_inicial`, com cache em Parquet.

    A primeira leitura do CSV (~340 MB) leva 1-2 minutos; as seguintes, poucos segundos.
    """
    if cfg.cache_limpo.exists() and not recriar:
        return pd.read_parquet(cfg.cache_limpo)

    caminho = encontrar_csv(cfg.dir_bruto)
    with cronometro(log, f"Lendo {caminho.name}"):
        df = ler_csv_anac(caminho)
    df = df[df["ano"] >= cfg.ano_inicial].reset_index(drop=True)
    validar_dados(df)

    cfg.dir_cache.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cfg.cache_limpo, index=False)
    log.info("Cache salvo em %s (%s registros)", cfg.cache_limpo, f"{len(df):,}")
    return df
