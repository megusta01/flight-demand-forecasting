import numpy as np
import pandas as pd
import pytest

from anac_demanda.data import carregar_dados
from anac_demanda.evaluation import calcular_metricas, wape
from anac_demanda.features import construir_painel
from anac_demanda.models import BASELINES, ModeloPrevisao, construir_modelos
from anac_demanda.splits import dividir


@pytest.fixture
def painel(config_teste):
    return construir_painel(carregar_dados(config_teste), config_teste)


def test_wape():
    assert wape([100, 100], [90, 110]) == pytest.approx(0.1)
    with pytest.raises(ValueError):
        wape([0, 0], [1, 1])


def test_metricas_cortam_previsoes_negativas():
    m = calcular_metricas([10, 10], [-5, 10])
    assert m["MAE"] == pytest.approx(5.0)  # -5 vira 0


def test_divisao_temporal_sem_sobreposicao(painel, config_teste):
    d = dividir(painel, config_teste)
    t = {k: set(v["t_origem"]) for k, v in d.__dict__.items()}
    nomes = list(t)
    for i, a in enumerate(nomes):
        for b in nomes[i + 1 :]:
            assert not t[a] & t[b], f"{a} e {b} compartilham meses"
    assert max(t["treino"]) < min(t["validacao"]) < min(t["teste"]) < min(t["monitoramento"]) < min(t["producao"])


def test_baselines(painel):
    for fn in BASELINES.values():
        pred = fn(painel)
        assert pred.shape == (len(painel),) and np.all(pred >= 0)


@pytest.mark.parametrize("indice", [0, 1])
def test_modelos_treinam_preveem_e_persistem(painel, config_teste, tmp_path, indice):
    d = dividir(painel, config_teste)
    modelo = construir_modelos(config_teste)[indice].fit(d.treino, d.treino["alvo"])
    pred = modelo.predict(d.validacao)
    assert pred.shape == (len(d.validacao),) and np.all(pred >= 0)

    caminho = modelo.salvar(tmp_path / "artefatos")
    recarregado = ModeloPrevisao.carregar(caminho)
    np.testing.assert_allclose(recarregado.predict(d.validacao), pred)
    assert (tmp_path / "artefatos" / "modelo.json").exists()


def test_modelo_aceita_categoria_nunca_vista(painel, config_teste):
    d = dividir(painel, config_teste)
    modelo = construir_modelos(config_teste)[0].fit(d.treino, d.treino["alvo"])
    novo = d.validacao.head(3).copy()
    novo["empresa_sigla"] = pd.array(["NOVA"] * 3, dtype="string")
    assert np.all(np.isfinite(modelo.predict(novo)))


def test_carregar_modelo_inexistente(tmp_path):
    with pytest.raises(FileNotFoundError, match="treinar"):
        ModeloPrevisao.carregar(tmp_path / "nada.joblib")
