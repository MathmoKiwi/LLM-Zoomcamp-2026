# Convenience targets. Everything also works as plain docker compose commands.

up:
	docker compose up -d --build

down:
	docker compose down

ingest:
	docker compose run --rm ingest

eval-sets:
	docker compose run --rm --no-deps ingest python evals/build_eval_set.py

eval-retrieval:
	docker compose run --rm ingest python evals/eval_retrieval.py

eval-llm:
	docker compose run --rm ingest python evals/eval_llm.py --sample 100

logs:
	docker compose logs -f app

psql:
	docker compose exec postgres psql -U technote -d technote
