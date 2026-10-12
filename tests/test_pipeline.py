"""Teste de ponta a ponta: CSV sintético -> CLI 'tudo' -> modelo, relatórios, gráficos e previsões."""

import json

import pandas as pd

from anac_demanda.cli import main


def test_pipeline_completo_via_cli(config_teste, tmp_path):
    cfg_path = tmp_path / "config.yaml"
    assert main(["--config", str(cfg_path), "tudo"]) == 0

    assert (config_teste.dir_artefatos / "modelo.joblib").exists()
    meta = json.loads((config_teste.dir_artefatos / "modelo.json").read_text(encoding="utf-8"))
    assert meta["treinado_ate"] == "2025-12"

    metricas = json.loads((config_teste.dir_relatorios / "metricas.json").read_text(encoding="utf-8"))
    assert metricas["monitoramento"]["inicio"] == "2026-01"
    assert metricas["proxima_previsao"]["mes"] == "2026-03"
    for nome in ("resultados.md", "model_card.md", "eda.md"):
        assert (config_teste.dir_relatorios / nome).stat().st_size > 0
    for fig in ("resultado_teste.png", "serie_total.png", "importancia_features.png", "eda_sazonalidade.png"):
        assert (config_teste.dir_figuras / fig).exists()

    prev = pd.read_csv(config_teste.dir_previsoes / "previsoes_2026-03.csv")
    assert len(prev) == 12  # uma linha por rota regular ativa
    assert (prev["previsao"] >= 0).all() and set(prev["mes_base"]) == {"2026-02"}


def test_prever_sem_modelo_retorna_erro(config_teste, tmp_path):
    assert main(["--config", str(tmp_path / "config.yaml"), "prever"]) == 1


def test_config_invalida_retorna_codigo_2(tmp_path):
    assert main(["--config", str(tmp_path / "nao_existe.yaml"), "treinar"]) == 2
