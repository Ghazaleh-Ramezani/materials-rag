FROM python:3.11-slim

WORKDIR /app

# System deps kept minimal; faiss-cpu ships manylinux wheels.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Build indexes at image build time so the container is query-ready.
# Uses the bundled sample_corpus and the offline hashing backend by default.
ENV EMBEDDING_BACKEND=hashing \
    LLM_PROVIDER=mock
RUN python -m src.ingestion.ingest_papers && python -m scripts.build_indexes

EXPOSE 8000
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
