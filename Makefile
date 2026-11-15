
.PHONY: setup test run-server run-worker

setup:
	bash scripts/setup.sh

test:
	uv run pytest tests/

generate:
	buf generate

run-server:
	uv run python -m src.server.main

run-worker:
	uv run python -m src.worker.main
