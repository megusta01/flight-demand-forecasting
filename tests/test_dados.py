import pandas as pd
import pytest

from anac_demanda.data import carregar_dados, encontrar_csv, ler_csv_anac
from anac_demanda.data.schema import SchemaError, validar_dados

from .conftest import escrever_csv_anac, linha_registro


def test_le_formato_oficial_com_virgula_decimal(tmp_path):
    reg = linha_registro("AAA", 2024, 3, "SBGR", "SBRJ", 1234.0)
    caminho = escrever_csv_anac([reg], tmp_path / "dados.csv")
    df = ler_csv_anac(caminho)
    assert list(df["empresa_sigla"]) == ["AAA"]
    assert df.loc[0, "passageiros_pagos"] == 1234.0
    assert df.loc[0, "assentos"] == pytest.approx(1542.5)  # "1542,5" -> 1542.5
    assert df["ano"].dtype.kind == "i" and df["mes"].dtype.kind == "i"


def test_coluna_obrigatoria_ausente_gera_erro(tmp_path):
    caminho = tmp_path / "ruim.csv"
    caminho.write_text("Atualizado em: x\nEMPRESA_SIGLA;ANO;MES\nAAA;2024;1\n", encoding="utf-8")
    with pytest.raises(SchemaError, match="ausentes"):
        ler_csv_anac(caminho)


def test_sem_csv_mostra_onde_baixar(tmp_path):
    with pytest.raises(FileNotFoundError, match="anac.gov.br"):
        encontrar_csv(tmp_path)


def test_passageiros_negativos_sao_rejeitados():
    df = pd.DataFrame(
        {
            "mes": [1],
            "ano": [2024],
            "passageiros_pagos": [-5.0],
            "empresa_sigla": ["A"],
            "aeroporto_de_origem_sigla": ["X"],
            "aeroporto_de_destino_sigla": ["Y"],
            "grupo_de_voo": ["R"],
        }
    )
    with pytest.raises(SchemaError, match="negativos"):
        validar_dados(df)


def test_cache_parquet_e_reaproveitado(config_teste):
    df1 = carregar_dados(config_teste)
    assert config_teste.cache_limpo.exists()
    df2 = carregar_dados(config_teste)
    pd.testing.assert_frame_equal(df1, df2)
    assert df1["ano"].min() >= config_teste.ano_inicial
