"""Estilo visual comum dos gráficos (paleta validada para daltonismo)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

from ..periodos import NOMES_MESES  # noqa: E402

CORES = {
    "azul": "#2a78d6",
    "laranja": "#eb6834",
    "verde": "#1baf7a",
    "amarelo": "#eda100",
    "cinza": "#9a9890",
    "texto": "#0b0b0b",
    "texto2": "#52514e",
    "grade": "#e6e5e0",
    "faixa": "#f3f2ee",
}
CATEGORICAS = [CORES["azul"], CORES["laranja"], CORES["verde"], CORES["amarelo"]]


def aplicar_estilo() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": CORES["grade"],
            "axes.labelcolor": CORES["texto2"],
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.titlepad": 22,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": CORES["grade"],
            "grid.linewidth": 0.8,
            "xtick.color": CORES["texto2"],
            "ytick.color": CORES["texto2"],
            "text.color": CORES["texto"],
            "font.size": 10.5,
            "legend.frameon": False,
            "lines.linewidth": 2,
        }
    )


def _milhar(x, _pos=None) -> str:
    if abs(x) >= 1e6:
        return f"{x / 1e6:.1f} mi".replace(".", ",")
    if abs(x) >= 1e3:
        return f"{x / 1e3:.0f} mil"
    return f"{x:.0f}"


MILHAR = FuncFormatter(_milhar)
PERCENTUAL = FuncFormatter(lambda x, _p: f"{x:.0%}")


def eixo_mensal(ax, intervalo: int = 3) -> None:
    """Datas no formato 'jan/24'."""
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=intervalo))
    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda x, _p: f"{NOMES_MESES[mdates.num2date(x).month - 1]}/{mdates.num2date(x).year % 100:02d}")
    )


def subtitulo(ax, texto: str) -> None:
    ax.text(0, 1.01, texto, transform=ax.transAxes, fontsize=9.5, color=CORES["texto2"], va="bottom")


def salvar(fig, diretorio: Path, nome: str) -> Path:
    diretorio.mkdir(parents=True, exist_ok=True)
    caminho = diretorio / nome
    fig.savefig(caminho, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return caminho


def pct(v: float, casas: int = 1) -> str:
    return f"{v:.{casas}%}".replace(".", ",")
