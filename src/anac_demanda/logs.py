"""Configuração de logging do projeto."""

from __future__ import annotations

import logging
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager

LOGGER_NAME = "anac_demanda"


def configurar_logging(nivel: int = logging.INFO) -> None:
    logger = logging.getLogger(LOGGER_NAME)
    if logger.handlers:
        logger.setLevel(nivel)
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S"))
    logger.addHandler(handler)
    logger.setLevel(nivel)
    logger.propagate = False


def get_logger(nome: str) -> logging.Logger:
    return logging.getLogger(f"{LOGGER_NAME}.{nome.split('.')[-1]}")


@contextmanager
def cronometro(logger: logging.Logger, etapa: str) -> Iterator[None]:
    inicio = time.perf_counter()
    logger.info("%s...", etapa)
    yield
    logger.info("%s concluído em %.1fs", etapa, time.perf_counter() - inicio)
