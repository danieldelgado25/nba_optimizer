.PHONY: install lint fmt typecheck test check

install:
	pip install -e ".[dev]"
	pre-commit install

lint:
	ruff check .
	ruff format --check .

fmt:
	ruff check --fix .
	ruff format .

typecheck:
	mypy

test:
	pytest

check: lint typecheck test
