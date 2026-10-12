"""Smoke test do painel Streamlit com as saídas versionadas em outputs/ e reports/."""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

PAINEL = Path(__file__).resolve().parents[1] / "app" / "painel.py"
SAIDAS = PAINEL.parents[1] / "outputs"


@pytest.mark.skipif(not any(SAIDAS.glob("previsoes_20*.csv")), reason="sem previsões geradas")
def test_painel_carrega_sem_erros():
    app = AppTest.from_file(str(PAINEL), default_timeout=60).run()
    assert not app.exception
    assert len(app.metric) == 4
    assert app.tabs[0].label.startswith("Previsão de")
