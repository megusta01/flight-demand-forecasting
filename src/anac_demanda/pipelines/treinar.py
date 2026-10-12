"""Pipeline de treino e avaliação.

Etapas:
1. Compara baselines e modelos candidatos na validação (modelos ajustados só no treino).
2. Decide, também na validação, se os meses da COVID entram no treino.
3. Retreina o melhor modelo em treino + validação e avalia no teste (2024-2025).
4. Treina o modelo de produção com todos os meses até o fim do teste e o salva em `artifacts/`.
5. Monitoramento: avalia o modelo de produção nos meses publicados depois do teste.
6. Gera gráficos, relatórios e o model card.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import Config
from ..data import carregar_dados
from ..evaluation import calcular_metricas
from ..evaluation.analysis import erro_por_mes, erro_por_porte, importancia_permutacao
from ..features import CHAVES_SERIE, carregar_painel
from ..logs import cronometro, get_logger
from ..models import BASELINES, ModeloPrevisao, construir_modelos
from ..periodos import Mes
from ..splits import Divisao, dividir, sem_covid
from ..viz import resultados as graficos
from . import relatorios
from .prever import caminho_modelo, gerar_previsoes

log = get_logger(__name__)


@dataclass
class ResultadoTreino:
    validacao: pd.DataFrame
    modelo_escolhido: str
    baseline_referencia: str
    covid: dict
    teste: pd.DataFrame
    monitoramento: dict | None
    importancia: pd.DataFrame
    porte: pd.DataFrame
    mensal: pd.DataFrame
    linhas: dict[str, int]
    extras: dict = field(default_factory=dict)

    @property
    def wape_modelo_teste(self) -> float:
        return float(self.teste.loc[self.teste["modelo"] == self.modelo_escolhido, "WAPE"].item())

    @property
    def wape_baseline_teste(self) -> float:
        return float(self.teste.loc[self.teste["modelo"] == self.baseline_referencia, "WAPE"].item())

    @property
    def reducao_erro(self) -> float:
        return 1 - self.wape_modelo_teste / self.wape_baseline_teste


def _linha(nome: str, tipo: str, y, pred) -> dict:
    return {"modelo": nome, "tipo": tipo, **calcular_metricas(y, pred)}


def comparar_na_validacao(d: Divisao, cfg: Config) -> tuple[pd.DataFrame, dict[str, ModeloPrevisao]]:
    linhas = [_linha(n, "baseline", d.validacao["alvo"], f(d.validacao)) for n, f in BASELINES.items()]
    ajustados = {}
    for modelo in construir_modelos(cfg):
        with cronometro(log, f"Treinando '{modelo.nome}'"):
            modelo.fit(d.treino, d.treino["alvo"])
        linhas.append(_linha(modelo.nome, "modelo", d.validacao["alvo"], modelo.predict(d.validacao)))
        ajustados[modelo.nome] = modelo
        log.info("  WAPE validação: %.2f%%", 100 * linhas[-1]["WAPE"])
    tabela = pd.DataFrame(linhas).sort_values("WAPE").reset_index(drop=True)
    return tabela, ajustados


def decidir_covid(melhor: ModeloPrevisao, d: Divisao, cfg: Config) -> dict:
    """Compara, na validação, o treino com e sem os meses-alvo da COVID."""
    sem = sem_covid(d.treino)
    alternativo = melhor.clonar().fit(sem, sem["alvo"])
    com_wape = calcular_metricas(d.validacao["alvo"], melhor.predict(d.validacao))["WAPE"]
    sem_wape = calcular_metricas(d.validacao["alvo"], alternativo.predict(d.validacao))["WAPE"]
    remover = cfg.remover_covid if cfg.remover_covid is not None else sem_wape < com_wape
    return {
        "WAPE_com_covid": com_wape,
        "WAPE_sem_covid": sem_wape,
        "remover": bool(remover),
        "decidido_por": "configuração" if cfg.remover_covid is not None else "validação",
    }


def _treino_final(*partes: pd.DataFrame, remover_covid: bool) -> pd.DataFrame:
    df = pd.concat(partes)
    return sem_covid(df) if remover_covid else df


def _tabela_previsoes(df: pd.DataFrame, conjunto: str, pred_modelo, pred_base) -> pd.DataFrame:
    cols = [*CHAVES_SERIE, "ano_alvo", "mes_alvo", "pax_lag0", "alvo"]
    return df[cols].assign(conjunto=conjunto, previsto_modelo=pred_modelo, previsto_baseline=pred_base)


def executar(cfg: Config) -> ResultadoTreino:
    cfg.garantir_diretorios()
    df = carregar_dados(cfg)
    painel = carregar_painel(cfg, df)
    d = dividir(painel, cfg)
    log.info("Amostras: %s", {k: f"{v:,}" for k, v in d.resumo().items()})

    # 1. seleção na validação
    validacao, ajustados = comparar_na_validacao(d, cfg)
    escolhido = validacao[validacao["tipo"] == "modelo"].iloc[0]["modelo"]
    ref_base = validacao[validacao["tipo"] == "baseline"].iloc[0]["modelo"]
    melhor = ajustados[escolhido]
    log.info("Melhor modelo na validação: %s | baseline de referência: %s", escolhido, ref_base)

    # 2. COVID e importância das variáveis (ambos com o modelo ajustado só no treino)
    covid = decidir_covid(melhor, d, cfg)
    log.info(
        "COVID no treino: %s (WAPE validação com %.2f%% / sem %.2f%%)",
        "remover" if covid["remover"] else "manter",
        100 * covid["WAPE_com_covid"],
        100 * covid["WAPE_sem_covid"],
    )
    with cronometro(log, "Importância por permutação"):
        imp = importancia_permutacao(melhor, d.validacao, cfg.semente)

    # 3. teste
    with cronometro(log, "Treino final (treino + validação) e avaliação no teste"):
        base_teste = _treino_final(d.treino, d.validacao, remover_covid=covid["remover"])
        avaliador = melhor.clonar().fit(base_teste, base_teste["alvo"])
        pred_teste = avaliador.predict(d.teste)
    preds_base_teste = {n: f(d.teste) for n, f in BASELINES.items()}
    teste = (
        pd.DataFrame(
            [_linha(escolhido, "modelo", d.teste["alvo"], pred_teste)]
            + [_linha(n, "baseline", d.teste["alvo"], p) for n, p in preds_base_teste.items()]
        )
        .sort_values("WAPE")
        .reset_index(drop=True)
    )
    pred_base_teste = preds_base_teste[ref_base]

    # 4. modelo de produção (todos os meses até o fim do teste)
    with cronometro(log, "Treino do modelo de produção"):
        base_prod = _treino_final(d.treino, d.validacao, d.teste, remover_covid=covid["remover"])
        producao = melhor.clonar().fit(base_prod, base_prod["alvo"])
    m_teste = teste.set_index("modelo").loc[escolhido].drop("tipo").to_dict()
    producao.metadados = {
        "treinado_ate": str(cfg.teste[1]),
        "amostras_treino": len(base_prod),
        "covid_removida_do_treino": covid["remover"],
        "metricas_teste": m_teste,
        "baseline_referencia": ref_base,
    }

    # 5. monitoramento: meses publicados depois do teste
    tabelas = [_tabela_previsoes(d.teste, "teste", pred_teste, pred_base_teste)]
    monitoramento = None
    if not d.monitoramento.empty:
        pm = producao.predict(d.monitoramento)
        pb = BASELINES[ref_base](d.monitoramento)
        monitoramento = {
            "inicio": str(Mes.de_indice(int(d.monitoramento["t_origem"].min()) + 1, cfg.ano_inicial)),
            "fim": str(Mes.de_indice(int(d.monitoramento["t_origem"].max()) + 1, cfg.ano_inicial)),
            "amostras": len(d.monitoramento),
            "modelo": calcular_metricas(d.monitoramento["alvo"], pm),
            "baseline": calcular_metricas(d.monitoramento["alvo"], pb),
        }
        producao.metadados["metricas_monitoramento"] = monitoramento["modelo"]
        tabelas.append(_tabela_previsoes(d.monitoramento, "monitoramento", pm, pb))
        log.info(
            "Monitoramento %s a %s: WAPE modelo %.2f%% | baseline %.2f%%",
            monitoramento["inicio"],
            monitoramento["fim"],
            100 * monitoramento["modelo"]["WAPE"],
            100 * monitoramento["baseline"]["WAPE"],
        )
    producao.salvar(cfg.dir_artefatos)
    previsoes = pd.concat(tabelas, ignore_index=True)
    previsoes.to_parquet(cfg.dir_previsoes / "previsoes_avaliacao.parquet", index=False)

    # 6. análises, gráficos e relatórios
    porte = erro_por_porte(d.teste, pred_teste, pred_base_teste)
    mensal = erro_por_mes(
        pd.concat([d.teste, d.monitoramento]),
        np.asarray(previsoes["previsto_modelo"]),
        np.asarray(previsoes["previsto_baseline"]),
    )
    proxima = None
    if not d.producao.empty:
        prev = gerar_previsoes(producao, d.producao, cfg.ano_inicial)
        proxima = (Mes.parse(prev["mes_previsto"].iloc[0]), int(prev["previsao"].sum()))

    nome_base = ref_base.split(" (")[0]
    inicio_mon = pd.Timestamp(f"{monitoramento['inicio']}-01") if monitoramento else None
    graficos.comparacao_teste(teste, cfg.dir_figuras)
    graficos.serie_total(
        mensal,
        inicio_mon,
        (pd.Timestamp(f"{proxima[0]}-01"), proxima[1]) if proxima else None,
        nome_base,
        cfg.dir_figuras,
    )
    graficos.rotas_exemplo(d.teste, pred_teste, pred_base_teste, nome_base, cfg.dir_figuras)
    graficos.importancia(imp, cfg.dir_figuras)
    graficos.erro_por_porte(porte, nome_base, cfg.dir_figuras)

    resultado = ResultadoTreino(
        validacao=validacao,
        modelo_escolhido=escolhido,
        baseline_referencia=ref_base,
        covid=covid,
        teste=teste,
        monitoramento=monitoramento,
        importancia=imp,
        porte=porte,
        mensal=mensal,
        linhas=d.resumo(),
        extras={"proxima_previsao": proxima, "treinado_ate": str(cfg.teste[1])},
    )
    relatorios.escrever_resultados(resultado, cfg)
    relatorios.escrever_model_card(resultado, cfg)
    with (cfg.dir_relatorios / "metricas.json").open("w", encoding="utf-8") as fh:
        json.dump(relatorios.resumo_json(resultado), fh, ensure_ascii=False, indent=2)

    log.info(
        "Teste: WAPE modelo %.2f%% | %s %.2f%% | redução de erro %.1f%% | modelo salvo em %s",
        100 * resultado.wape_modelo_teste,
        ref_base,
        100 * resultado.wape_baseline_teste,
        100 * resultado.reducao_erro,
        caminho_modelo(cfg),
    )
    return resultado
