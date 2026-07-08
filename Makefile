.PHONY: help install install-all test lint format typecheck dry-run clean

help:
	@echo "Targets:"
	@echo "  install      Editable install with core dependencies"
	@echo "  install-all  Editable install with all extras (rl, viz, analysis, dev)"
	@echo "  test         Run the test suite"
	@echo "  lint         Ruff lint"
	@echo "  format       Black + Ruff autofix"
	@echo "  typecheck    Mypy"
	@echo "  dry-run      Resolve and print the default experiment wiring"
	@echo "  clean        Remove caches and build artefacts"

install:
	pip install -e .

install-all:
	pip install -e ".[rl,viz,analysis,dev]"

test:
	pytest

lint:
	ruff check .

format:
	black .
	ruff check --fix .

typecheck:
	mypy .

dry-run:
	nci --config configs/default.yaml --dry-run

clean:
	rm -rf build dist *.egg-info .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
