.PHONY: help install install-dev dados treinar prever tudo testes lint formatar app notebooks limpar

PY ?= python

help:  ## mostra os comandos disponíveis
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install:  ## instala o pacote
	$(PY) -m pip install -e .

install-dev:  ## instala pacote + ferramentas de desenvolvimento + painel
	$(PY) -m pip install -e ".[dev,app]"
	pre-commit install

dados:  ## lê o CSV da ANAC e cria o cache
	anac-demanda dados

treinar:  ## seleção de modelo, teste, modelo de produção e monitoramento
	anac-demanda treinar

prever:  ## previsão do mês seguinte com o modelo salvo
	anac-demanda prever

tudo:  ## pipeline completo
	anac-demanda tudo

testes:  ## roda os testes com cobertura
	pytest --cov --cov-report=term-missing

lint:  ## checa estilo e erros comuns
	ruff check src tests app
	ruff format --check src tests app

formatar:  ## formata o código
	ruff check --fix src tests app
	ruff format src tests app

app:  ## abre o painel interativo
	streamlit run app/painel.py

notebooks:  ## executa os notebooks e salva as saídas
	jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb

limpar:  ## remove caches e arquivos temporários; preserva modelos e saídas
	rm -rf data/interim .pytest_cache .ruff_cache htmlcov
	rm -f .coverage .coverage.* coverage.xml coverage.json
	find src tests app -type d -name __pycache__ -prune -exec rm -rf {} +
