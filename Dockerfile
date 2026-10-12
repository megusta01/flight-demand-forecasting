FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg
WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[app]"

COPY configs ./configs
COPY app ./app
COPY .streamlit/config.toml ./.streamlit/config.toml

# O CSV da ANAC é montado em tempo de execução:
#   docker build -t anac-demanda .
#   docker run --rm -v "$PWD/data:/app/data" -v "$PWD/artifacts:/app/artifacts" \
#              -v "$PWD/outputs:/app/outputs" -v "$PWD/reports:/app/reports" -v "$PWD/figures:/app/figures" \
#              anac-demanda tudo
#   docker run --rm -p 8501:8501 -v "$PWD/outputs:/app/outputs" -v "$PWD/reports:/app/reports" \
#              --entrypoint streamlit anac-demanda run app/painel.py --server.address=0.0.0.0
ENTRYPOINT ["anac-demanda"]
CMD ["tudo"]
