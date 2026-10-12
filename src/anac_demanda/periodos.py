"""Representação de meses (ano + mês) e conversão para índice inteiro."""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import total_ordering

_PADRAO = re.compile(r"^(\d{4})-(\d{2})$")
NOMES_MESES = ["jan", "fev", "mar", "abr", "mai", "jun", "jul", "ago", "set", "out", "nov", "dez"]


@total_ordering
@dataclass(frozen=True)
class Mes:
    ano: int
    mes: int

    def __post_init__(self) -> None:
        if not 1 <= self.mes <= 12:
            raise ValueError(f"Mês inválido: {self.mes}")

    @classmethod
    def parse(cls, texto: str) -> Mes:
        m = _PADRAO.match(str(texto).strip())
        if not m:
            raise ValueError(f"Use o formato AAAA-MM (recebido: {texto!r})")
        return cls(int(m.group(1)), int(m.group(2)))

    @classmethod
    def de_indice(cls, indice: int, ano_base: int) -> Mes:
        return cls(ano_base + indice // 12, indice % 12 + 1)

    def indice(self, ano_base: int) -> int:
        """Número de meses desde janeiro do ano-base (jan/ano_base = 0)."""
        return (self.ano - ano_base) * 12 + (self.mes - 1)

    def somar(self, meses: int) -> Mes:
        total = self.ano * 12 + (self.mes - 1) + meses
        return Mes(total // 12, total % 12 + 1)

    def __lt__(self, other: Mes) -> bool:
        return (self.ano, self.mes) < (other.ano, other.mes)

    def __str__(self) -> str:
        return f"{self.ano:04d}-{self.mes:02d}"

    @property
    def rotulo(self) -> str:
        """Ex.: 'mai/2026'."""
        return f"{NOMES_MESES[self.mes - 1]}/{self.ano}"
