"""Divisão temporal do painel pelo mês previsto."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .config import Config


@dataclass
class Divisao:
    treino: pd.DataFrame
    validacao: pd.DataFrame
    teste: pd.DataFrame
    monitoramento: pd.DataFrame  # meses observados depois do teste (nunca usados no desenvolvimento)
    producao: pd.DataFrame  # último mês da base: alvo desconhecido, é o que se quer prever

    def resumo(self) -> dict[str, int]:
        return {k: len(v) for k, v in self.__dict__.items()}


def dividir(painel: pd.DataFrame, cfg: Config) -> Divisao:
    t_alvo = painel["t_origem"] + 1
    base = cfg.ano_inicial

    def entre(ini, fim):
        return t_alvo.between(ini.indice(base), fim.indice(base))

    conhecido = painel["alvo"].notna()
    return Divisao(
        treino=painel[conhecido & (t_alvo <= cfg.fim_treino.indice(base))],
        validacao=painel[conhecido & entre(*cfg.validacao)],
        teste=painel[conhecido & entre(*cfg.teste)],
        monitoramento=painel[conhecido & (t_alvo > cfg.teste[1].indice(base))],
        producao=painel[~conhecido],
    )


def sem_covid(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["covid"] == 0]
