# documentation/

The maintainer's view of `dream`: every module, class and function (internals included), call and
caller graphs, a source browser, and the product's Markdown pages, rendered by Doxygen. The
rendered parts (`html/`, `xml/`, `doxygen-warnings.log`) are generated and not committed.

```
documentation/
  README.md       this file
  Doxyfile        Doxygen configuration (non-default settings only); INPUT is the documented scope
  MAINPAGE.md     the start page (index.html)
  build.sh        docstring check + Doxygen; fails on any warning
  html/           GENERATED: open html/index.html
  xml/            GENERATED: machine-readable mirror
```

```bash
documentation/build.sh          # from the product root; needs only uv
make doxygen                    # the same
```

The docstring checker is `check_docstrings.py` (repo level, shared by every product);
`doxygen` and `dot` are the pinned binaries `doxytools` fetches into
`~/.cache/doxytools` on first use. CI runs both on every change to this product. The rendered
site is published under `/docs/doxygen/dream/` on the documentation endpoint by `make publish-docs`
at the repo root. Docstring form: see the house standards (`@param` / `@return` / `@throws`,
`Attributes:` bullets, `##` constants).
