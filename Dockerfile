FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml requirements.txt README.md ./
COPY mars ./mars
COPY config ./config
RUN pip install --no-cache-dir ".[web]"
EXPOSE 8000
VOLUME ["/app/runs"]
ENTRYPOINT ["mars"]
CMD ["--help"]
