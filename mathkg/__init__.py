"""MathKG -- a reproducible pipeline from Wikidata + Wikipedia to a coloured graph.

Stages (each is a pure function of the previous stage's output):

    wikidata      SPARQL          -> data/raw/nodes.csv, data/raw/fields.csv
    import-legacy df_filtered.csv -> data/raw/nodes.csv, data/raw/links.csv
    fetch         action API      -> data/raw/nodes.csv, data/raw/links.csv
    build         graph + colour  -> data/artifacts/*.json|gexf|graphml|html

Run ``python run.py --help``.
"""

from __future__ import annotations

import logging

__version__ = "1.0.0"


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("urllib3").setLevel(logging.WARNING)
