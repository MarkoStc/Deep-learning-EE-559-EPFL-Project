# Red Queen Moderation. Run `make help` for targets.
# Every target goes through uv, so no virtualenv activation is needed.

CONFIG ?= configs/default.yaml
RUN ?=
TORCH ?= cpu
IMAGE ?= red-queen
RUN_ARG = $(if $(RUN),--run-dir $(RUN),)

.DEFAULT_GOAL := help
.PHONY: help install lint format typecheck test test-fast coverage audit check smoke run arms-race docker-build docker-build-cuda clean

help:  ## list targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-18s %s\n", $$1, $$2}'

install:  ## install deps and dev tools (TORCH=cpu or TORCH=cu126)
	uv sync --extra $(TORCH) --extra tensorboard
	uv run pre-commit install

lint:  ## ruff lint and format check
	uv run ruff check .
	uv run ruff format --check .

format:  ## apply ruff fixes and formatting
	uv run ruff check --fix .
	uv run ruff format .

typecheck:  ## strict mypy
	uv run mypy

test:  ## full test suite with coverage
	uv run pytest --cov --cov-report=term-missing

test-fast:  ## tests without the slow end-to-end pipeline
	uv run pytest -m "not slow"

audit:  ## scan locked dependencies for known vulnerabilities
	uv export --frozen --no-hashes --extra cpu --no-emit-project -o .audit-requirements.txt
	sed -i 's/+cpu//' .audit-requirements.txt  # CPU wheel is not on PyPI; audit the same release
	uv run pip-audit --strict --no-deps --disable-pip -r .audit-requirements.txt --ignore-vuln GHSA-8mgp-746c-j5xp
	rm -f .audit-requirements.txt

check: lint typecheck test  ## everything CI runs, except Docker

smoke:  ## full pipeline on synthetic data with a tiny model (CPU, about a minute)
	uv run red-queen pipeline --config configs/smoke.yaml

run:  ## full pipeline with CONFIG; continue a run with RUN=runs/<id>
	uv run red-queen pipeline --config $(CONFIG) $(RUN_ARG)

arms-race:  ## multi-round loop; RUN=runs/<id> reuses that run's Guardian
	uv run red-queen arms-race --config $(CONFIG) $(RUN_ARG)

docker-build:  ## CPU image
	docker build --target cpu -t $(IMAGE):cpu .

docker-build-cuda:  ## CUDA 12.6 image
	docker build --target cuda -t $(IMAGE):cuda .

clean:  ## remove caches (keeps runs/ and data/)
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage coverage.xml htmlcov
