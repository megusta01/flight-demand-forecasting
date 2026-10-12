"""Fixtures: dados sintéticos no formato do CSV da ANAC e configuração isolada em diretório temporário."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from anac_demanda.config import Config, carregar_config

CABECALHO = [
    "EMPRESA_SIGLA",
    "EMPRESA_NOME",
    "EMPRESA_NACIONALIDADE",
    "ANO",
    "MES",
    "AEROPORTO_DE_ORIGEM_SIGLA",
    "AEROPORTO_DE_ORIGEM_NOME",
    "AEROPORTO_DE_ORIGEM_UF",
    "AEROPORTO_DE_ORIGEM_REGIAO",
    "AEROPORTO_DE_ORIGEM_PAIS",
    "AEROPORTO_DE_ORIGEM_CONTINENTE",
    "AEROPORTO_DE_DESTINO_SIGLA",
    "AEROPORTO_DE_DESTINO_NOME",
    "AEROPORTO_DE_DESTINO_UF",
    "AEROPORTO_DE_DESTINO_REGIAO",
    "AEROPORTO_DE_DESTINO_PAIS",
    "AEROPORTO_DE_DESTINO_CONTINENTE",
    "NATUREZA",
    "GRUPO_DE_VOO",
    "PASSAGEIROS_PAGOS",
    "PASSAGEIROS_GRATIS",
    "ASSENTOS",
    "DECOLAGENS",
    "DISTANCIA_VOADA_KM",
]


def linha_registro(
    empresa: str, ano: int, mes: int, origem: str, destino: str, pax: float, grupo: str = "REGULAR"
) -> dict:
    return {
        "EMPRESA_SIGLA": empresa,
        "EMPRESA_NOME": f"EMPRESA {empresa}",
        "EMPRESA_NACIONALIDADE": "BRASILEIRA",
        "ANO": ano,
        "MES": mes,
        "AEROPORTO_DE_ORIGEM_SIGLA": origem,
        "AEROPORTO_DE_ORIGEM_NOME": origem,
        "AEROPORTO_DE_ORIGEM_UF": "SP",
        "AEROPORTO_DE_ORIGEM_REGIAO": "SUDESTE",
        "AEROPORTO_DE_ORIGEM_PAIS": "BRASIL",
        "AEROPORTO_DE_ORIGEM_CONTINENTE": "AMÉRICA DO SUL",
        "AEROPORTO_DE_DESTINO_SIGLA": destino,
        "AEROPORTO_DE_DESTINO_NOME": destino,
        "AEROPORTO_DE_DESTINO_UF": "RJ",
        "AEROPORTO_DE_DESTINO_REGIAO": "SUDESTE",
        "AEROPORTO_DE_DESTINO_PAIS": "BRASIL",
        "AEROPORTO_DE_DESTINO_CONTINENTE": "AMÉRICA DO SUL",
        "NATUREZA": "DOMÉSTICA",
        "GRUPO_DE_VOO": grupo,
        "PASSAGEIROS_PAGOS": pax,
        "PASSAGEIROS_GRATIS": 0,
        "ASSENTOS": pax * 1.25,
        "DECOLAGENS": 30,
        "DISTANCIA_VOADA_KM": 30 * 400,
    }


def escrever_csv_anac(registros: list[dict], caminho: Path) -> Path:
    """Grava no formato oficial: linha 'Atualizado em', ';' como separador e vírgula decimal."""
    df = pd.DataFrame(registros)[CABECALHO]
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with caminho.open("w", encoding="utf-8-sig") as fh:
        fh.write("Atualizado em: 2026-03-31\n")
        df.astype(str).apply(lambda c: c.str.replace(".", ",", regex=False)).to_csv(fh, sep=";", index=False, quoting=1)
    return caminho


def registros_sinteticos(inicio=(2018, 1), fim=(2026, 2), n_rotas: int = 12, semente: int = 0) -> list[dict]:
    """Rotas com tendência, sazonalidade (picos em jan/jul/dez) e ruído."""
    rng = np.random.default_rng(semente)
    sazonal = np.array([1.2, 0.85, 0.95, 0.95, 0.95, 0.95, 1.15, 1.0, 0.95, 1.0, 1.0, 1.15])
    regs = []
    for r in range(n_rotas):
        empresa = ["AAA", "BBB", "CCC"][r % 3]
        base = rng.uniform(800, 30_000)
        ano, mes, i = *inicio, 0
        while (ano, mes) <= fim:
            pax = base * (1 + 0.01 * i) ** 0.5 * sazonal[mes - 1] * rng.normal(1, 0.05)
            regs.append(linha_registro(empresa, ano, mes, f"SB{r:02d}", f"SD{r:02d}", round(pax)))
            mes, i = (1, i + 1) if mes == 12 else (mes + 1, i + 1)
            if mes == 1:
                ano += 1
    regs.append(linha_registro("AAA", 2024, 5, "SB00", "SD00", 50, grupo="NÃO REGULAR"))
    return regs


@pytest.fixture
def config_teste(tmp_path: Path) -> Config:
    """Config apontando tudo para um diretório temporário, com modelos pequenos (testes rápidos)."""
    escrever_csv_anac(registros_sinteticos(), tmp_path / "raw" / "Dados_Estatisticos.csv")
    raw = yaml.safe_load((Path(__file__).parents[1] / "configs" / "default.yaml").read_text(encoding="utf-8"))
    raw["dados"].update(
        {"dir_bruto": str(tmp_path / "raw"), "dir_cache": str(tmp_path / "interim"), "ano_inicial": 2018}
    )
    raw["features"].update({"min_ocorrencias_categoria": 1})
    raw["saidas"] = {k: str(tmp_path / k) for k in ("figuras", "relatorios", "artefatos", "previsoes")}
    raw["modelos"] = [
        {
            "nome": "GB variação",
            "tipo": "hgb",
            "alvo": "variacao",
            "params": {"loss": "absolute_error", "max_iter": 30, "max_leaf_nodes": 7, "min_samples_leaf": 5},
        },
        {"nome": "Ridge", "tipo": "ridge", "alvo": "log", "params": {"alpha": 1.0}},
    ]
    caminho = tmp_path / "config.yaml"
    caminho.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    return carregar_config(caminho)
