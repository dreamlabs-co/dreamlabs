#!/usr/bin/env bash
# Regenerate documentation/html and documentation/xml for this product. Run from anywhere:
#     documentation/build.sh
# Needs only uv: the dev group installs doxytools, whose `doxygen` and `dot`
# console scripts fetch the pinned binaries into ~/.cache/doxytools on first use. Fails on
# any Doxygen warning (WARN_AS_ERROR in the Doxyfile) and on any docstring the checker rejects.
set -euo pipefail
cd "$(dirname "$0")/.."

command -v uv >/dev/null || { echo "uv not on PATH (https://docs.astral.sh/uv/)"; exit 1; }

echo "== docstring check"
python3 ../../infra/docs/check_docstrings.py --doxyfile documentation/Doxyfile

echo "== tools (first run downloads doxygen + graphviz into ~/.cache/doxytools)"
uv sync --group dev --inexact -q
uv run -q doxytools-install

echo "== doxygen"
rm -rf documentation/html documentation/xml
uv run -q doxygen documentation/Doxyfile
warnings=$(grep -c . documentation/doxygen-warnings.log || true)
echo "doxygen warnings: $warnings (documentation/doxygen-warnings.log)"
echo "open documentation/html/index.html"
