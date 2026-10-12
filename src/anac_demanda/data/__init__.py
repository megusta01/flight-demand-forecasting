"""Ingestão e validação dos dados da ANAC."""

from .ingest import carregar_dados, encontrar_csv, ler_csv_anac

__all__ = ["carregar_dados", "encontrar_csv", "ler_csv_anac"]
