"""Leitura e validação da configuração (YAML -> dataclasses tipadas)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

from .periodos import Mes

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"

TipoModelo = Literal["hgb", "ridge"]
AlvoModelo = Literal["variacao", "log", "contagem"]


@dataclass(frozen=True)
class ModeloSpec:
    nome: str
    tipo: TipoModelo
    alvo: AlvoModelo
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Config:
    raiz: Path
    semente: int
    dir_bruto: Path
    dir_cache: Path
    ano_inicial: int
    grupo_de_voo: str
    fim_treino: Mes
    validacao: tuple[Mes, Mes]
    teste: tuple[Mes, Mes]
    covid: tuple[Mes, Mes]
    remover_covid: bool | None  # None = decidido pela validação
    min_ocorrencias_categoria: int
    max_categorias: int
    modelos: tuple[ModeloSpec, ...]
    dir_figuras: Path
    dir_relatorios: Path
    dir_artefatos: Path
    dir_previsoes: Path

    @property
    def cache_limpo(self) -> Path:
        return self.dir_cache / "anac_limpo.parquet"

    @property
    def cache_painel(self) -> Path:
        return self.dir_cache / "painel_rotas.parquet"

    def garantir_diretorios(self) -> None:
        for d in (self.dir_cache, self.dir_figuras, self.dir_relatorios, self.dir_artefatos, self.dir_previsoes):
            d.mkdir(parents=True, exist_ok=True)


class ConfigError(ValueError):
    """Configuração inválida."""


def _caminho(raiz: Path, valor: str) -> Path:
    p = Path(valor)
    return p if p.is_absolute() else raiz / p


def carregar_config(caminho: str | Path | None = None, raiz: Path | None = None) -> Config:
    """Lê o YAML, valida e devolve um `Config` com caminhos absolutos."""
    caminho = Path(caminho) if caminho else DEFAULT_CONFIG
    if not caminho.exists():
        raise ConfigError(f"Arquivo de configuração não encontrado: {caminho}")
    with caminho.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    raiz = raiz or PROJECT_ROOT
    try:
        dados, periodos, covid = raw["dados"], raw["periodos"], raw["covid"]
        feats, saidas = raw["features"], raw["saidas"]
        modelos = tuple(ModeloSpec(**m) for m in raw["modelos"])
    except (KeyError, TypeError) as exc:
        raise ConfigError(f"Seção ou campo ausente em {caminho}: {exc}") from exc

    for m in modelos:
        if m.tipo not in ("hgb", "ridge"):
            raise ConfigError(f"Modelo '{m.nome}': tipo '{m.tipo}' inválido (use hgb ou ridge).")
        if m.alvo not in ("variacao", "log", "contagem"):
            raise ConfigError(f"Modelo '{m.nome}': alvo '{m.alvo}' inválido.")
    if not modelos:
        raise ConfigError("Defina ao menos um modelo em 'modelos'.")

    remover = covid.get("remover_do_treino", "auto")
    cfg = Config(
        raiz=raiz,
        semente=int(raw.get("semente", 42)),
        dir_bruto=_caminho(raiz, dados["dir_bruto"]),
        dir_cache=_caminho(raiz, dados["dir_cache"]),
        ano_inicial=int(dados["ano_inicial"]),
        grupo_de_voo=str(dados["grupo_de_voo"]),
        fim_treino=Mes.parse(periodos["fim_treino"]),
        validacao=(Mes.parse(periodos["validacao"][0]), Mes.parse(periodos["validacao"][1])),
        teste=(Mes.parse(periodos["teste"][0]), Mes.parse(periodos["teste"][1])),
        covid=(Mes.parse(covid["inicio"]), Mes.parse(covid["fim"])),
        remover_covid=None if remover == "auto" else bool(remover),
        min_ocorrencias_categoria=int(feats["min_ocorrencias_categoria"]),
        max_categorias=int(feats["max_categorias"]),
        modelos=modelos,
        dir_figuras=_caminho(raiz, saidas["figuras"]),
        dir_relatorios=_caminho(raiz, saidas["relatorios"]),
        dir_artefatos=_caminho(raiz, saidas["artefatos"]),
        dir_previsoes=_caminho(raiz, saidas["previsoes"]),
    )
    if not (cfg.fim_treino < cfg.validacao[0] <= cfg.validacao[1] < cfg.teste[0] <= cfg.teste[1]):
        raise ConfigError("Períodos fora de ordem: esperado treino < validação < teste.")
    if cfg.max_categorias > 254:
        raise ConfigError("max_categorias deve ser no máximo 254.")
    return cfg
