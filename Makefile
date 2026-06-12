.PHONY: install install-dev test test-fast lint train score pipeline validate clean

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

test:
	pytest tests/ -m "not slow" --cov=retail_analytics --cov-report=term-missing

test-all:
	pytest tests/ --cov=retail_analytics --cov-report=term-missing

test-fast:
	pytest tests/ -m "not slow" -q

lint:
	ruff check src tests

validate:
	retail-analytics validate-data

ingest:
	retail-analytics ingest

train:
	retail-analytics train

score:
	retail-analytics score

pipeline:
	retail-analytics run-pipeline

clean:
	rm -rf artifacts data/processed data/outputs .pytest_cache .coverage htmlcov
