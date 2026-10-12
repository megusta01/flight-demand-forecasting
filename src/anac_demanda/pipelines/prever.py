"""Pipeline de inferência: previsão do mês seguinte ao último mês disponível nos dados."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..config import Config
from ..data import carregar_dados
from ..features import carregar_painel, historico_mensal
from ..logs import get_logger
from ..models import BASELINES, ModeloPrevisao
from ..periodos import Mes
from ..splits import dividir

log = get_logger(__name__)

COLUNAS_SAIDA = [
    "empresa_sigla",
    "aeroporto_de_origem_sigla",
    "aeroporto_de_destino_sigla",
    "aeroporto_de_origem_regiao",
    "aeroporto_de_destino_regiao",
    "natureza",
    "mes_base",
    "passageiros_mes_base",
    "mes_previsto",
    "previsao",
    "variacao_prevista",
    "previsao_ingenua",
]


def caminho_modelo(cfg: Config) -> Path:
    return cfg.dir_artefatos / "modelo.joblib"


def gerar_previsoes(modelo: ModeloPrevisao, linhas: pd.DataFrame, ano_base: int) -> pd.DataFrame:
    """Previsões para as linhas de produção (alvo desconhecido), em formato de tabela de negócio."""
    if linhas.empty:
        raise ValueError("Não há linhas para prever (o último mês da base não tem rotas ativas).")
    base = Mes.de_indice(int(linhas["t_origem"].iloc[0]), ano_base)
    out = linhas.copy()
    out["mes_base"] = str(base)
    out["mes_previsto"] = str(base.somar(1))
    out["passageiros_mes_base"] = out["pax_lag0"].astype(int)
    out["previsao"] = modelo.predict(linhas).round().astype(int)
    out["variacao_prevista"] = (out["previsao"] / out["passageiros_mes_base"] - 1).round(4)
    out["previsao_ingenua"] = BASELINES["Ingênuo (repete o mês atual)"](linhas).round().astype(int)
    return out[COLUNAS_SAIDA].sort_values("previsao", ascending=False).reset_index(drop=True)


def executar(cfg: Config, modelo_path: Path | None = None) -> Path:
    cfg.garantir_diretorios()
    modelo = ModeloPrevisao.carregar(modelo_path or caminho_modelo(cfg))
    df = carregar_dados(cfg)
    painel = carregar_painel(cfg, df)
    producao = dividir(painel, cfg).producao

    previsoes = gerar_previsoes(modelo, producao, cfg.ano_inicial)
    mes = previsoes["mes_previsto"].iloc[0]
    destino = cfg.dir_previsoes / f"previsoes_{mes}.csv"
    previsoes.to_csv(destino, index=False)

    # histórico compacto usado pelo painel interativo
    historico_mensal(df, cfg).to_parquet(cfg.dir_previsoes / "historico_rotas.parquet", index=False)

    log.info(
        "Previsão de %s: %s rotas, %s passageiros no total (modelo treinado até %s) -> %s",
        Mes.parse(mes).rotulo,
        f"{len(previsoes):,}",
        f"{previsoes['previsao'].sum():,}",
        modelo.metadados.get("treinado_ate", "?"),
        destino,
    )
    return destino
