"""Painel interativo: previsões por rota, histórico e desempenho do modelo.

Uso:  streamlit run app/painel.py      (depois de `anac-demanda tudo`)
"""

from __future__ import annotations

import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

RAIZ = Path(__file__).resolve().parents[1]
SAIDAS = RAIZ / "outputs"
RELATORIOS = RAIZ / "reports"
MESES = ["jan", "fev", "mar", "abr", "mai", "jun", "jul", "ago", "set", "out", "nov", "dez"]
AZUL, LARANJA, CINZA, TEXTO = "#2a78d6", "#eb6834", "#9a9890", "#0b0b0b"
# Rótulos de eixo em português (o Vega-Lite usa inglês por padrão)
EIXO_MES = alt.Axis(
    labelExpr="['jan','fev','mar','abr','mai','jun','jul','ago','set','out','nov','dez'][month(datum.value)]"
    " + '/' + timeFormat(datum.value, '%y')",
    title=None,
)
EIXO_NUM = alt.Axis(format="~s")

st.set_page_config(page_title="Demanda aérea por rota", page_icon="✈️", layout="wide")


def rotulo_mes(aaaa_mm: str) -> str:
    ano, mes = aaaa_mm.split("-")
    return f"{MESES[int(mes) - 1]}/{ano}"


def num(v: float) -> str:
    return f"{v:,.0f}".replace(",", ".")


def pct(v: float, casas: int = 1) -> str:
    return f"{v:.{casas}%}".replace(".", ",")


def pp(v: float) -> str:
    """Diferença em pontos percentuais."""
    return f"{100 * v:+.1f} p.p.".replace(".", ",", 1)


@st.cache_data
def carregar() -> dict:
    arquivos = sorted(SAIDAS.glob("previsoes_20*.csv"))
    faltando = [p for p in (SAIDAS / "previsoes_avaliacao.parquet", RELATORIOS / "metricas.json") if not p.exists()]
    if not arquivos or faltando:
        return {}
    dados = {
        "previsoes": pd.read_csv(arquivos[-1]),
        "avaliacao": pd.read_parquet(SAIDAS / "previsoes_avaliacao.parquet"),
        "metricas": json.loads((RELATORIOS / "metricas.json").read_text(encoding="utf-8")),
        "historico": None,
    }
    if (SAIDAS / "historico_rotas.parquet").exists():
        dados["historico"] = pd.read_parquet(SAIDAS / "historico_rotas.parquet")
    for nome in ("previsoes", "avaliacao", "historico"):
        df = dados[nome]
        if df is not None:
            df["rota"] = df["aeroporto_de_origem_sigla"] + " → " + df["aeroporto_de_destino_sigla"]
    return dados


dados = carregar()
if not dados:
    st.error("Arquivos de saída não encontrados. Rode `anac-demanda tudo` na raiz do projeto e recarregue a página.")
    st.stop()

prev, aval, met = dados["previsoes"], dados["avaliacao"], dados["metricas"]
mes_prev = prev["mes_previsto"].iloc[0]
teste = {r["modelo"]: r for r in met["teste"]}
m_teste, b_teste = teste[met["modelo_escolhido"]], teste[met["baseline_referencia"]]

# ---------------------------------------------------------------- cabeçalho
st.title("✈️ Demanda aérea por rota")
st.caption(
    "Previsão de passageiros pagos no mês seguinte para cada rota regular no Brasil, "
    "com dados abertos da ANAC. Modelo: " + met["modelo_escolhido"] + "."
)

c1, c2, c3, c4 = st.columns(4)
c1.metric(f"Passageiros previstos · {rotulo_mes(mes_prev)}", num(prev["previsao"].sum()))
c2.metric("Rotas previstas", num(len(prev)))
c3.metric(
    "Erro no teste (WAPE)",
    pct(m_teste["WAPE"]),
    delta=f"{pp(m_teste['WAPE'] - b_teste['WAPE'])} vs. baseline",
    delta_color="inverse",
)
if met.get("monitoramento"):
    mon = met["monitoramento"]
    c4.metric(
        f"Erro em dados novos ({rotulo_mes(mon['inicio'])}–{rotulo_mes(mon['fim'])})",
        pct(mon["modelo"]["WAPE"]),
        delta=f"{pp(mon['modelo']['WAPE'] - mon['baseline']['WAPE'])} vs. baseline",
        delta_color="inverse",
    )

aba_prev, aba_rota, aba_desemp = st.tabs([f"Previsão de {rotulo_mes(mes_prev)}", "Explorar uma rota", "Desempenho"])

# ---------------------------------------------------------------- previsão do próximo mês
with aba_prev:
    f1, f2, f3 = st.columns(3)
    empresas = f1.multiselect("Empresa", sorted(prev["empresa_sigla"].unique()), placeholder="Todas")
    regioes = f2.multiselect(
        "Região de origem", sorted(prev["aeroporto_de_origem_regiao"].dropna().unique()), placeholder="Todas"
    )
    natureza = f3.multiselect("Natureza", sorted(prev["natureza"].unique()), placeholder="Todas")
    filtrado = prev
    if empresas:
        filtrado = filtrado[filtrado["empresa_sigla"].isin(empresas)]
    if regioes:
        filtrado = filtrado[filtrado["aeroporto_de_origem_regiao"].isin(regioes)]
    if natureza:
        filtrado = filtrado[filtrado["natureza"].isin(natureza)]

    top = filtrado.head(15).assign(nome=lambda d: d["empresa_sigla"] + "  " + d["rota"])
    grafico = (
        alt.Chart(top)
        .mark_bar(color=AZUL, cornerRadiusEnd=3)
        .encode(
            x=alt.X("previsao:Q", title=f"Passageiros previstos em {rotulo_mes(mes_prev)}", axis=EIXO_NUM),
            y=alt.Y("nome:N", sort="-x", title=None),
            tooltip=[
                alt.Tooltip("nome:N", title="Rota"),
                alt.Tooltip("passageiros_mes_base:Q", title="Mês atual", format=",.0f"),
                alt.Tooltip("previsao:Q", title="Previsão", format=",.0f"),
                alt.Tooltip("variacao_prevista:Q", title="Variação", format="+.1%"),
            ],
        )
        .properties(height=420, title="15 rotas com mais passageiros previstos")
    )
    st.altair_chart(grafico, use_container_width=True)

    tabela = filtrado[
        ["empresa_sigla", "rota", "natureza", "passageiros_mes_base", "previsao", "variacao_prevista"]
    ].rename(
        columns={
            "empresa_sigla": "Empresa",
            "rota": "Rota",
            "natureza": "Natureza",
            "passageiros_mes_base": f"Real em {rotulo_mes(prev['mes_base'].iloc[0])}",
            "previsao": f"Previsão {rotulo_mes(mes_prev)}",
            "variacao_prevista": "Variação",
        }
    )
    st.dataframe(
        tabela,
        hide_index=True,
        use_container_width=True,
        column_config={"Variação": st.column_config.NumberColumn(format="percent")},
    )
    st.download_button(
        "Baixar previsões (CSV)",
        filtrado.to_csv(index=False).encode("utf-8"),
        file_name=f"previsoes_{mes_prev}.csv",
        mime="text/csv",
    )

# ---------------------------------------------------------------- uma rota
with aba_rota:
    e1, e2 = st.columns([1, 2])
    lista_empresas = sorted(prev["empresa_sigla"].unique())
    empresa = e1.selectbox("Empresa", lista_empresas, index=lista_empresas.index(prev["empresa_sigla"].iloc[0]))
    opcoes = prev[prev["empresa_sigla"] == empresa] if empresa else prev
    rota = e2.selectbox("Rota (ordenada pela previsão)", opcoes["rota"].unique(), index=0)
    if empresa is None or rota is None:
        st.info("Escolha uma empresa e uma rota para ver o histórico e as previsões.")
    else:
        alvo_prev = prev[(prev["empresa_sigla"] == empresa) & (prev["rota"] == rota)]
        series = []
        if dados["historico"] is not None:
            h = dados["historico"]
            h = h[(h["empresa_sigla"] == empresa) & (h["rota"] == rota)]
            series.append(
                pd.DataFrame(
                    {
                        "data": pd.to_datetime(
                            h[["ano", "mes"]].assign(day=1).rename(columns={"ano": "year", "mes": "month"})
                        ),
                        "valor": h["passageiros"],
                        "série": "Real",
                    }
                )
            )
        a = aval[(aval["empresa_sigla"] == empresa) & (aval["rota"] == rota)]
        datas_a = pd.to_datetime(
            a[["ano_alvo", "mes_alvo"]].assign(day=1).rename(columns={"ano_alvo": "year", "mes_alvo": "month"})
        )
        series.append(pd.DataFrame({"data": datas_a, "valor": a["previsto_modelo"], "série": "Previsão do modelo"}))
        if dados["historico"] is None:
            series.append(pd.DataFrame({"data": datas_a, "valor": a["alvo"], "série": "Real"}))
        if not alvo_prev.empty:
            series.append(
                pd.DataFrame(
                    {
                        "data": [pd.Timestamp(f"{mes_prev}-01")],
                        "valor": [alvo_prev["previsao"].iloc[0]],
                        "série": "Próximo mês",
                    }
                )
            )
        s = pd.concat(series, ignore_index=True)
        cores = alt.Scale(domain=["Real", "Previsão do modelo", "Próximo mês"], range=[TEXTO, AZUL, LARANJA])
        base = alt.Chart(s).encode(
            x=alt.X("data:T", axis=EIXO_MES),
            y=alt.Y("valor:Q", title="Passageiros pagos", axis=EIXO_NUM),
            color=alt.Color("série:N", scale=cores, legend=alt.Legend(orient="top", title=None)),
            tooltip=[alt.Tooltip("data:T", format="%b/%Y"), "série:N", alt.Tooltip("valor:Q", format=",.0f")],
        )
        linhas = base.transform_filter(alt.datum["série"] != "Próximo mês").mark_line(strokeWidth=2)
        ponto = base.transform_filter(alt.datum["série"] == "Próximo mês").mark_point(size=120, filled=True)
        st.altair_chart((linhas + ponto).properties(height=380), use_container_width=True)
        if not a.empty:
            erro_m = (a["previsto_modelo"] - a["alvo"]).abs().sum() / a["alvo"].sum()
            erro_b = (a["previsto_baseline"] - a["alvo"]).abs().sum() / a["alvo"].sum()
            st.caption(f"Erro nesta rota no teste + monitoramento: modelo {pct(erro_m)} · baseline {pct(erro_b)}")

# ---------------------------------------------------------------- desempenho
with aba_desemp:
    st.subheader("Erro mês a mês")
    st.caption(
        "Previsão um mês à frente. Teste: 2024–2025. Depois disso, meses publicados após o modelo ser congelado."
    )
    linhas = []
    for (ano, mes, conj), g in aval.groupby(["ano_alvo", "mes_alvo", "conjunto"]):
        total = g["alvo"].sum()
        for nome, col in (("Modelo", "previsto_modelo"), ("Baseline ingênuo", "previsto_baseline")):
            erro = (g[col] - g["alvo"]).abs().sum() / total
            linhas.append(
                {"data": pd.Timestamp(year=ano, month=mes, day=1), "WAPE": erro, "série": nome, "conjunto": conj}
            )
    erros = pd.DataFrame(linhas)
    st.altair_chart(
        alt.Chart(erros)
        .mark_line(point=True)
        .encode(
            x=alt.X("data:T", axis=EIXO_MES),
            y=alt.Y("WAPE:Q", axis=alt.Axis(format="%"), title="WAPE"),
            color=alt.Color(
                "série:N", scale=alt.Scale(range=[CINZA, AZUL]), legend=alt.Legend(orient="top", title=None)
            ),
            tooltip=[
                alt.Tooltip("data:T", format="%b/%Y"),
                "série:N",
                alt.Tooltip("WAPE:Q", format=".1%"),
                "conjunto:N",
            ],
        )
        .properties(height=320),
        use_container_width=True,
    )

    st.subheader("Comparação no teste")
    tab = pd.DataFrame(met["teste"])[["modelo", "tipo", "WAPE", "MAE"]]
    st.dataframe(
        tab,
        hide_index=True,
        use_container_width=True,
        column_config={
            "WAPE": st.column_config.NumberColumn(format="percent"),
            "MAE": st.column_config.NumberColumn("MAE (passageiros/rota/mês)", format="%.0f"),
        },
    )
    st.caption("WAPE = soma dos erros absolutos / soma dos passageiros reais. Menor é melhor.")
