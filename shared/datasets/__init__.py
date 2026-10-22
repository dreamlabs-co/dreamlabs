"""The embodiment config parser and the client-side dataset checks, with two consumers: the dream client, which
parses a config and validates a local LeRobot dataset against it before uploading (`dream.datasets`, the wheel
carries a copy of this package), and the data service, which parses a config before storing it, so the one line
the client shows for a bad config is the line the service answers, and joins the paths a dataset names when it
processes one.

`embodiments` is the parser (`parse()`, standard library only, so it imports in the service image with neither numpy
nor pyarrow); `validation` checks the layout from the file list alone (`check_layout()`) and then the content through
a reader (`check_dataset()`: the `meta/` files, every episode parquet and every mp4 header, via `content_checks`)
against a parsed `embodiments.Embodiment`, and answers the counts the client's report shows. numpy and pyarrow are
imported by `content_checks` alone: a consumer that validates selects `dreamlabs-shared[datasets]`, one that only
parses needs the base package. `paths` is the one join for a path a dataset's `meta/info.json` names (`safe_join()`,
standard library only), used by the data service. Python 3.10 compatible, the dream client's floor.
"""
