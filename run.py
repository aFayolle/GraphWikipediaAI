#!/usr/bin/env python3
"""One entry point for every stage of the pipeline.

    python run.py wikidata          # SPARQL -> data/raw/nodes.csv, fields.csv
    python run.py fetch             # MediaWiki API -> data/raw/pages
    python run.py import-legacy     # df_filtered.csv -> data/raw (no network)
    python run.py build             # graph + colours + all exports
    python run.py all               # wikidata -> fetch -> build
    python run.py all-offline       # import-legacy -> build
    python run.py curriculum        # the 29-node prerequisite DAG

Every stage writes to ``data/`` and reads only what the previous stage wrote,
so a run is reproducible end to end. ``build`` verifies that the artifacts it
just wrote actually agree with each other -- the shipped files in this repo did
not (6,623 nodes in the GEXF vs 7,444 in the JSON).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import networkx as nx
import pandas as pd

from mathkg import (
    color,
    config,
    curriculum,
    edges,
    export,
    fetch,
    graph,
    render,
    scope,
    wikidata,
)

log = logging.getLogger("run")


# --------------------------------------------------------------------------- #
# Stages
# --------------------------------------------------------------------------- #

def stage_wikidata(args: argparse.Namespace) -> None:  # noqa: ARG001 (uniform signature)
    wikidata.run(out_nodes=config.NODES_CSV, out_fields=config.FIELDS_CSV)


def stage_import_legacy(args: argparse.Namespace) -> None:
    nodes, links = fetch.from_legacy_csv(Path(args.legacy_csv))
    fetch.write_raw(nodes, links)


def stage_fetch(args: argparse.Namespace) -> None:  # noqa: ARG001 (uniform signature)
    fetch.run(nodes_csv=config.NODES_CSV)


def _load_fields() -> pd.DataFrame | None:
    path = config.FIELDS_CSV if config.FIELDS_CSV.exists() else config.LEGACY_FIELDS_CSV
    if not path.exists():
        log.warning("no fields CSV found; every node will be grey")
        return None
    log.info("anchors: %s", path)
    return graph.load_fields(path)


def _colored_graph(
    args: argparse.Namespace,
) -> tuple[nx.DiGraph, list[str], dict[str, list[int | None]]]:
    """Load, build and colour the article graph.

    Shared by `build` and `render` so a picture can never come from a different
    graph than the artifacts.
    """
    for path in (config.NODES_CSV, config.LINKS_CSV):
        if not path.exists():
            sys.exit(f"missing {path}: run `wikidata` + `fetch`, or `import-legacy`, first")

    nodes = graph.load_nodes(config.NODES_CSV)
    links = graph.load_links(config.LINKS_CSV)

    # Ordered: the palette is positional, so a field keeps its colour.
    anchor_list = graph.anchor_order(_load_fields())
    article_graph = graph.build_graph(
        nodes, links, anchors=set(anchor_list), max_degree=args.max_degree
    )

    order: list[str] = []
    distances: dict[str, list[int | None]] = {}
    if anchor_list:
        order, distances = color.color_graph(article_graph, anchors=anchor_list)
    return article_graph, order, distances


def stage_build(args: argparse.Namespace) -> None:
    article_graph, order, distances = _colored_graph(args)

    log.info("graph summary: %s", graph.summary(article_graph))
    log.info("graph shape: %s", graph.shape_metrics(article_graph))

    written = export.export_all(
        article_graph,
        distances=distances,
        order=order,
        out_dir=config.ARTIFACT_DIR,
        stem="graph",
        pyvis=not args.no_html,
    )
    verify_artifacts(article_graph, written)


def field_legend(order: list[str]) -> list[tuple[str, str]]:
    """(field, colour) pairs for the figure legend, matching the palette order."""
    palette = config.ANCHOR_PALETTE
    legend = [(field, palette[i % len(palette)]) for i, field in enumerate(order)]
    legend.append(("no reachable field", config.UNREACHABLE_COLOR))
    return legend


RENDER_VIEWS = ("fields",)


def stage_render(args: argparse.Namespace) -> None:
    """Render a static PNG (the channel through which I can look at the graph)."""
    if args.view not in RENDER_VIEWS:
        sys.exit(f"unknown view {args.view!r}; available: {', '.join(RENDER_VIEWS)}")

    article_graph, order, distances = _colored_graph(args)
    stats = graph.summary(article_graph)

    out = render.render_graph(
        article_graph,
        config.ARTIFACT_DIR / "render" / f"{args.view}.png",
        title=f"Fields of mathematics — {args.view} view",
        subtitle=(
            f"{stats['nodes']:,} articles · {stats['edges']:,} links · "
            f"{stats['components']:,} components · "
            f"colour = distance to the 12 field anchors"
        ),
        legend=field_legend(order),
        label_top=args.labels,
        only_largest_component=args.largest_component,
    )
    log.info("rendered %s", out)


def stage_scope(args: argparse.Namespace) -> None:  # noqa: ARG001 (uniform signature)
    scope.run()


def stage_edges(args: argparse.Namespace) -> None:
    edges.run(use_wikidata=not args.no_wikidata)


def stage_curriculum(args: argparse.Namespace) -> None:  # noqa: ARG001 (uniform signature)
    curriculum.run(config.ARTIFACT_DIR)


def stage_all(args: argparse.Namespace) -> None:
    stage_wikidata(args)
    stage_fetch(args)
    stage_build(args)


def stage_all_offline(args: argparse.Namespace) -> None:
    stage_import_legacy(args)
    stage_build(args)


# --------------------------------------------------------------------------- #
# Consistency check
# --------------------------------------------------------------------------- #

def verify_artifacts(article_graph: nx.Graph, written: dict[str, Path]) -> None:
    """Re-read every artifact and check it describes the graph we just built.

    This is the guard against the failure that prompted the refactor: three
    files in one folder describing three different graphs.
    """
    expected_nodes = article_graph.number_of_nodes()
    expected_edges = article_graph.number_of_edges()
    problems: list[str] = []

    if "gexf" in written:
        reloaded = nx.read_gexf(written["gexf"])
        if reloaded.number_of_nodes() != expected_nodes:
            problems.append(f"gexf nodes {reloaded.number_of_nodes()} != {expected_nodes}")
        if reloaded.number_of_edges() != expected_edges:
            problems.append(f"gexf edges {reloaded.number_of_edges()} != {expected_edges}")

    if "json" in written:
        payload = json.loads(written["json"].read_text(encoding="utf-8"))
        if len(payload["nodes"]) != expected_nodes:
            problems.append(f"json nodes {len(payload['nodes'])} != {expected_nodes}")
        if len(payload["links"]) != expected_edges:
            problems.append(f"json links {len(payload['links'])} != {expected_edges}")

    if problems:
        for problem in problems:
            log.error("artifact mismatch: %s", problem)
        raise SystemExit("artifacts disagree with the graph; refusing to declare success")

    log.info(
        "verified: json, gexf and the in-memory graph all describe %d nodes / %d edges",
        expected_nodes,
        expected_edges,
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

STAGES = {
    "wikidata": (stage_wikidata, "fetch Wikidata items and field anchors (network)"),
    "import-legacy": (stage_import_legacy, "rebuild intermediates from df_filtered.csv"),
    "fetch": (stage_fetch, "fetch pages/links through the MediaWiki API (network)"),
    "build": (stage_build, "build, colour and export the graph"),
    "render": (stage_render, "render a static PNG of the graph (for review)"),
    "scope": (stage_scope, "classify items, fold duplicates, write the scoping report"),
    "edges": (stage_edges, "extract dependency edges (phrase + Wikidata)"),
    "all": (stage_all, "wikidata -> fetch -> build"),
    "all-offline": (stage_all_offline, "import-legacy -> build (no network)"),
    "curriculum": (stage_curriculum, "the hand-written prerequisite DAG"),
}

NETWORK_STAGES = {"wikidata", "fetch", "all"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run.py", description="Wikidata + Wikipedia -> coloured graph"
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--max-degree",
        type=int,
        default=config.DEFAULT_MAX_DEGREE,
        help="drop nodes whose degree exceeds this (anchors are always kept)",
    )
    parser.add_argument(
        "--legacy-csv",
        default=str(config.LEGACY_ENRICHED_CSV),
        help="df_filtered.csv used by `import-legacy`",
    )
    parser.add_argument(
        "--no-html", action="store_true", help="skip the pyvis HTML (slow on big graphs)"
    )
    parser.add_argument(
        "--view",
        default="fields",
        help=f"which view to render: {', '.join(RENDER_VIEWS)}",
    )
    parser.add_argument(
        "--labels", type=int, default=40, help="how many top-degree nodes to label"
    )
    parser.add_argument(
        "--no-wikidata",
        action="store_true",
        help="edges stage: skip the Wikidata extractor (phrase edges only, no network)",
    )
    parser.add_argument(
        "--largest-component",
        action="store_true",
        help="render only the biggest connected component",
    )

    sub = parser.add_subparsers(dest="stage", required=True)
    for name, (_, help_text) in STAGES.items():
        sub.add_parser(name, help=help_text)
    return parser


def main(argv: list[str] | None = None) -> int:
    from mathkg import setup_logging

    args = build_parser().parse_args(argv)
    setup_logging(args.verbose)
    config.ensure_dirs()

    if args.stage in NETWORK_STAGES and not config.contact_is_set():
        log.warning(
            "MATHKG_CONTACT is not set (export MATHKG_CONTACT='you@example.org'). "
            "Wikimedia's User-Agent policy requires a real contact."
        )

    STAGES[args.stage][0](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
