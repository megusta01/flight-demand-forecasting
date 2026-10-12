"""Esquema esperado dos Dados Estatísticos da ANAC e validações de qualidade."""

from __future__ import annotations

import pandas as pd

from ..logs import get_logger

log = get_logger(__name__)

# Colunas numéricas do CSV original (vírgula decimal e ponto de milhar).
COLUNAS_NUMERICAS = ["ano", "mes", "passageiros_pagos", "assentos", "decolagens", "distancia_voada_km"]

COLUNAS_TEXTO = [
    "empresa_sigla",
    "empresa_nacionalidade",
    "aeroporto_de_origem_sigla",
    "aeroporto_de_origem_uf",
    "aeroporto_de_origem_regiao",
    "aeroporto_de_origem_pais",
    "aeroporto_de_destino_sigla",
    "aeroporto_de_destino_uf",
    "aeroporto_de_destino_regiao",
    "aeroporto_de_destino_pais",
    "natureza",
    "grupo_de_voo",
]

COLUNAS_OBRIGATORIAS = COLUNAS_TEXTO + COLUNAS_NUMERICAS


class SchemaError(ValueError):
    """Arquivo de dados fora do formato esperado."""


def validar_colunas(colunas: list[str]) -> None:
    faltando = sorted(set(COLUNAS_OBRIGATORIAS) - set(colunas))
    if faltando:
        raise SchemaError(f"Colunas obrigatórias ausentes no CSV: {faltando}")


def validar_dados(df: pd.DataFrame) -> dict:
    """Checa consistência básica e devolve um resumo de qualidade (também registrado no log)."""
    if df.empty:
        raise SchemaError("A base está vazia depois da leitura.")
    if not df["mes"].between(1, 12).all():
        raise SchemaError("Há valores de mês fora do intervalo 1-12.")

    duplicadas = int(
        df.duplicated(
            ["empresa_sigla", "aeroporto_de_origem_sigla", "aeroporto_de_destino_sigla", "ano", "mes", "grupo_de_voo"]
        ).sum()
    )
    resumo = {
        "registros": len(df),
        "periodo": f"{int(df['ano'].min())}-{int(df['ano'].max())}",
        "passageiros_negativos": int((df["passageiros_pagos"] < 0).sum()),
        "passageiros_ausentes": int(df["passageiros_pagos"].isna().sum()),
        "sem_origem_ou_destino": int(
            df[["aeroporto_de_origem_sigla", "aeroporto_de_destino_sigla"]].isna().any(axis=1).sum()
        ),
        "chaves_duplicadas": duplicadas,
    }
    if resumo["passageiros_negativos"]:
        raise SchemaError(f"{resumo['passageiros_negativos']} registros com passageiros negativos.")
    if duplicadas:
        log.warning("%d registros duplicados (empresa, rota, mês, grupo de voo) serão somados.", duplicadas)
    log.info("Qualidade dos dados: %s", resumo)
    return resumo
