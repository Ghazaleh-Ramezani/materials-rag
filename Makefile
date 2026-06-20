.PHONY: install ingest build query api eval test docker

install:
	pip install -r requirements.txt

ingest:
	python -m src.ingestion.ingest_papers

build: ingest
	python -m scripts.build_indexes

query:
	python -m scripts.run_query "How does rGO reduction degree affect gauge factor?"

api:
	uvicorn src.api.main:app --reload --port 8000

eval:
	python -m src.evaluation.eval_retrieval --k 5

test:
	pytest -q

docker:
	docker build -t materials-rag . && docker run -p 8000:8000 materials-rag
