"""Interface de linha de comando.

Exemplos:
    anac-demanda tudo                      # pipeline completo
    anac-demanda explorar                  # análise exploratória
    anac-demanda treinar                   # seleção, teste, modelo de produção e monitoramento
    anac-demanda prever                    # previsão do mês seguinte com o modelo salvo
    anac-demanda --config configs/x.yaml treinar
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from . import __version__
from .config import ConfigError, carregar_config
from .logs import configurar_logging, get_logger

log = get_logger(__name__)

ETAPAS = ("dados", "explorar", "features", "treinar", "prever", "tudo")


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="anac-demanda",
        description="Previsão de passageiros por rota aérea (dados abertos da ANAC).",
    )
    p.add_argument(
        "etapa",
        choices=ETAPAS,
        nargs="?",
        default="tudo",
        help="dados: lê o CSV e cria o cache | explorar: EDA | features: monta o painel | "
        "treinar: seleção, teste e modelo de produção | prever: próximo mês | tudo: todas",
    )
    p.add_argument("--config", type=Path, default=None, help="arquivo YAML (padrão: configs/default.yaml)")
    p.add_argument("--recriar-cache", action="store_true", help="relê o CSV bruto e refaz o painel")
    p.add_argument("--modelo", type=Path, default=None, help="caminho do modelo para 'prever'")
    p.add_argument("-v", "--verbose", action="store_true", help="logs detalhados")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    configurar_logging(logging.DEBUG if args.verbose else logging.INFO)
    try:
        cfg = carregar_config(args.config)
    except ConfigError as exc:
        log.error("%s", exc)
        return 2

    inicio = time.perf_counter()
    try:
        etapa = args.etapa
        if etapa in ("dados", "tudo") or args.recriar_cache:
            from .data import carregar_dados

            carregar_dados(cfg, recriar=args.recriar_cache or etapa == "dados")
        if etapa in ("features", "tudo") or args.recriar_cache:
            from .features import carregar_painel

            carregar_painel(cfg, recriar=True)
        if etapa in ("explorar", "tudo"):
            from .pipelines import explorar

            explorar.executar(cfg)
        if etapa in ("treinar", "tudo"):
            from .pipelines import treinar

            treinar.executar(cfg)
        if etapa in ("prever", "tudo"):
            from .pipelines import prever

            prever.executar(cfg, args.modelo)
    except FileNotFoundError as exc:
        log.error("%s", exc)
        return 1

    log.info("Concluído em %.0fs.", time.perf_counter() - inicio)
    return 0


if __name__ == "__main__":
    sys.exit(main())
