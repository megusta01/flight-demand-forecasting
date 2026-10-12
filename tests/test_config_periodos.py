import pytest
import yaml

from anac_demanda.config import DEFAULT_CONFIG, ConfigError, carregar_config
from anac_demanda.periodos import Mes


def test_mes_parse_indice_e_soma():
    m = Mes.parse("2024-12")
    assert m.indice(2000) == 24 * 12 + 11
    assert m.somar(1) == Mes(2025, 1)
    assert Mes.de_indice(m.indice(2000), 2000) == m
    assert Mes(2024, 1) < Mes(2024, 2) < Mes(2025, 1)
    assert m.rotulo == "dez/2024"


@pytest.mark.parametrize("texto", ["2024-13", "24-01", "2024/01", ""])
def test_mes_invalido(texto):
    with pytest.raises(ValueError):
        Mes.parse(texto)


def test_config_padrao_carrega():
    cfg = carregar_config()
    assert cfg.fim_treino < cfg.validacao[0] < cfg.teste[0]
    assert cfg.dir_bruto.is_absolute()
    assert len(cfg.modelos) >= 1


def _config_alterada(tmp_path, alterar):
    raw = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    alterar(raw)
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    return p


def test_periodos_fora_de_ordem(tmp_path):
    p = _config_alterada(tmp_path, lambda r: r["periodos"].update({"fim_treino": "2024-06"}))
    with pytest.raises(ConfigError, match="ordem"):
        carregar_config(p)


def test_tipo_de_modelo_invalido(tmp_path):
    p = _config_alterada(tmp_path, lambda r: r["modelos"][0].update({"tipo": "xgboost"}))
    with pytest.raises(ConfigError, match="tipo"):
        carregar_config(p)


def test_arquivo_inexistente(tmp_path):
    with pytest.raises(ConfigError):
        carregar_config(tmp_path / "nao_existe.yaml")
