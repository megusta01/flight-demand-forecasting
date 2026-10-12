"""Painel de features: alvo correto, defasagens corretas e ausência de vazamento do futuro."""

import numpy as np
import pandas as pd

from anac_demanda.data.ingest import ler_csv_anac
from anac_demanda.features import FEATURES, FEATURES_CATEGORICAS, construir_painel

from .conftest import escrever_csv_anac, linha_registro


def _serie(n_meses=30, inicio=(2018, 1), passo=10):
    out, (ano, mes) = {}, inicio
    for i in range(n_meses):
        out[(ano, mes)] = 1000 + passo * i
        ano, mes = (ano + 1, 1) if mes == 12 else (ano, mes + 1)
    return out


def _painel(serie: dict, cfg, tmp_path, nome="dados.csv"):
    regs = [linha_registro("AAA", a, m, "SBAA", "SBBB", p) for (a, m), p in serie.items()]
    df = ler_csv_anac(escrever_csv_anac(regs, tmp_path / nome))
    return construir_painel(df, cfg)


def test_alvo_e_defasagens(config_teste, tmp_path):
    p = _painel(_serie(), config_teste, tmp_path)
    linha = p[(p["ano_alvo"] == 2019) & (p["mes_alvo"] == 6)].iloc[0]
    assert linha["alvo"] == 1170  # jun/2019 é o 18º mês
    assert linha["pax_lag0"] == 1160  # mai/2019
    assert linha["pax_lag11"] == 1050  # jun/2018: mesmo mês do alvo, um ano antes
    assert linha["pax_media_3m"] == np.mean([1140, 1150, 1160])
    assert linha["rel_pax_lag11"] == 1050 / 1160


def test_mudar_o_mes_previsto_nao_muda_nenhuma_feature(config_teste, tmp_path):
    base = _serie()
    alterada = dict(base)
    alterada[(2019, 6)] = 999_999  # mexe só no mês-alvo
    p1 = _painel(base, config_teste, tmp_path, "a.csv")
    p2 = _painel(alterada, config_teste, tmp_path, "b.csv")
    chave = (p1["ano_alvo"] == 2019) & (p1["mes_alvo"] == 6)
    numericas = [f for f in FEATURES if f not in FEATURES_CATEGORICAS]
    pd.testing.assert_frame_equal(
        p1.loc[chave, numericas].reset_index(drop=True), p2.loc[chave, numericas].reset_index(drop=True)
    )
    assert p2.loc[chave, "alvo"].item() == 999_999


def test_rota_que_para_tem_alvo_zero(config_teste, tmp_path):
    serie = {(2020, 1): 500, (2020, 2): 600}
    outra = {(a, m): 100 for a, m in _serie(30, (2019, 1))}  # mantém a base viva até 2021
    regs = [linha_registro("AAA", a, m, "SBAA", "SBBB", p) for (a, m), p in serie.items()]
    regs += [linha_registro("BBB", a, m, "SBCC", "SBDD", p) for (a, m), p in outra.items()]
    p = construir_painel(ler_csv_anac(escrever_csv_anac(regs, tmp_path / "c.csv")), config_teste)
    linha = p[(p["empresa_sigla"] == "AAA") & (p["ano_alvo"] == 2020) & (p["mes_alvo"] == 3)].iloc[0]
    assert linha["alvo"] == 0


def test_ultimo_mes_vira_linha_de_producao(config_teste, tmp_path):
    p = _painel(_serie(), config_teste, tmp_path)
    ultimo = p["t_origem"].max()
    assert p.loc[p["t_origem"] == ultimo, "alvo"].isna().all()
    assert p.loc[p["t_origem"] < ultimo, "alvo"].notna().all()


def test_apenas_grupo_configurado_entra_no_painel(config_teste, tmp_path):
    regs = [linha_registro("AAA", 2020, m, "SBAA", "SBBB", 100) for m in range(1, 7)]
    regs += [linha_registro("ZZZ", 2020, m, "SBXX", "SBYY", 100, grupo="NÃO REGULAR") for m in range(1, 7)]
    p = construir_painel(ler_csv_anac(escrever_csv_anac(regs, tmp_path / "d.csv")), config_teste)
    assert set(p["empresa_sigla"]) == {"AAA"}
