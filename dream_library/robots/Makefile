# robots inference control loop. `make check` is what CI runs.
.PHONY: test lint typecheck doxygen check

test:
	uv run pytest -q

lint:
	uv run ruff check . && uv run ruff format --check .

typecheck:
	uv run mypy

# Maintainer docs: docstring check + Doxygen (warnings as errors) into documentation/html.
doxygen:
	documentation/build.sh

check: lint typecheck test doxygen
