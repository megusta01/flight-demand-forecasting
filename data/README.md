# Dados

O projeto usa os **Dados Estatísticos do Transporte Aéreo** da ANAC: dados abertos, atualizados mensalmente,
com uma linha por empresa × origem × destino × mês × grupo de voo desde 2000.

1. Acesse a página do conjunto de dados:
   https://www.anac.gov.br/acesso-a-informacao/dados-abertos/areas-de-atuacao/voos-e-operacoes-aereas/dados-estatisticos-do-transporte-aereo
2. Baixe o arquivo completo `Dados_Estatisticos.csv` (~340 MB, separado por `;`).
3. Salve em `data/raw/Dados_Estatisticos.csv`.

O pipeline encontra automaticamente o primeiro `.csv` em `data/raw/`, valida as colunas obrigatórias e cria
um cache Parquet em `data/interim/` (ignorado pelo Git). As execuções seguintes levam segundos para carregar.

| Pasta | Conteúdo | Versionado |
|---|---|---|
| `raw/` | CSV original da ANAC | não |
| `interim/` | `anac_limpo.parquet` (base tipada) e `painel_rotas.parquet` (features) | não |

Os resultados publicados no repositório usam o arquivo "Atualizado em: 2026-06-21", com dados até abril de 2026.
Ao baixar uma versão mais nova, o mesmo comando (`anac-demanda tudo`) incorpora os meses novos: eles entram
como monitoramento e a previsão passa a ser do mês seguinte ao último publicado.
