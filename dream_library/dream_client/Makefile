# dream policy client. `make check` is what CI runs; `make test-full` is the nightly stress suite.
.PHONY: test test-full lint typecheck docs publish-docs doxygen check

DOCS_OUT := build/docs

test:
	uv run pytest -q

test-full:
	DREAM_STRESS=full uv run pytest -q tests/test_stress.py

lint:
	uv run ruff check . && uv run ruff format --check .

typecheck:
	uv run mypy

docs:
	uv run python docs/build_api_reference.py --out $(DOCS_OUT)

publish-docs: docs
	sh docs/publish.sh "$(DOCS_OUT)"

# Maintainer docs: docstring check + Doxygen (warnings as errors) into documentation/html.
doxygen:
	documentation/build.sh

check: lint typecheck test doxygen
