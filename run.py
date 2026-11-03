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
import urllib.error
import urllib.request
from pathlib import Path

import networkx as nx
import pandas as pd

from mathkg import (
    color,
    config,
    curriculum,
    dependency,
    edges,
    export,
    fetch,
    graph,
    render,
    render_focus,
    scope,
    view_render,
    wikidata,
)
from mathkg import (
    server as server_module,
)
from mathkg import (
    view as view_module,
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
    edges.run(use_wikidata=not args.no_wikidata, use_proof_sections=not args.no_proof)


def stage_dependency(args: argparse.Namespace) -> None:
    """Build the prerequisite DAG, report its structure, and render it."""
    edges_frame = dependency.load_edges()
    _, titles = edges.load_summaries()

    graph = dependency.build_dag(
        edges_frame,
        titles=set(titles),
        min_confidence=args.min_confidence,
        include_classification=args.include_classification,
    )
    stats = dependency.summary(graph)
    log.info("dependency DAG: %s", stats)

    cycles = dependency.cycle_report(graph)
    for component in cycles[:3]:
        log.info("  cycle (%d): %s", len(component), ", ".join(component[:8]))

    dag, dropped = dependency.break_cycles(graph)
    log.info("acyclic after dropping %d weak edge(s) = %s",
             dropped, nx.is_directed_acyclic_graph(dag))

    path = dependency.critical_path(dag)
    if path:
        log.info("critical path (%d): %s", len(path), " -> ".join(path))

    out = config.ARTIFACT_DIR / "dependency"
    out.mkdir(parents=True, exist_ok=True)

    if args.focus:
        focus = args.focus
        if focus not in dag:
            sys.exit(f"{focus!r} is not in the DAG ({dag.number_of_nodes()} nodes)")
        pre = dependency.prerequisites_of(dag, focus)
        enables = dependency.enables(dag, focus)
        log.info("%s: %d prerequisites, %d enabled", focus, len(pre) - 1, len(enables) - 1)
        ego = dependency.ego_graph(dag, focus, radius=args.radius)
        log.info("ego graph: %d nodes", ego.number_of_nodes())
        render_focus.render_prerequisite_view(
            dag, focus, out / "focus.png", radius=args.radius
        )
        export.write_pyvis(ego, out / "focus.html")

    biggest = render.largest_component(dag)
    if biggest.number_of_nodes() > 2:
        log.info("largest component: %d nodes", biggest.number_of_nodes())
        render.render_graph(
            biggest,
            out / "dependency.png",
            title="Prerequisite DAG — largest component",
            subtitle=(
                f"{biggest.number_of_nodes():,} nodes · {biggest.number_of_edges():,} edges · "
                "columns = prerequisite depth"
            ),
            positions=dependency.layered_positions(biggest, x_spacing=1.0, y_spacing=0.06),
            label_top=35,
            figsize=(18, 12),
        )
        export.write_pyvis(biggest, out / "dependency.html")


def _graph_for_spec(spec: dict) -> nx.Graph:
    """The graph a spec draws: the prerequisite DAG or the link graph."""
    if spec.get("source") == "link":
        article_graph, _, _ = _colored_graph(argparse.Namespace(
            max_degree=None, no_html=True, view="fields", labels=40,
            largest_component=False,
        ))
        return article_graph
    edges_frame = dependency.load_edges()
    _, titles = edges.load_summaries()
    dag = dependency.build_dag(edges_frame, titles=set(titles))
    return dependency.enrich_with_attributes(dag)


def stage_view(args: argparse.Namespace) -> None:
    """Render a graph from a declarative spec (no Python editing required)."""
    spec = view_module.load_spec(Path(args.spec) if args.spec else None)
    if args.set:
        view_module.apply_overrides(spec, args.set)
    view_module.validate_spec(spec)

    graph = _graph_for_spec(spec)
    mentions_path = config.RAW_DIR / "mentions.csv"
    mentions = pd.read_csv(mentions_path) if mentions_path.exists() else None

    name = args.out or f"{spec.get('name', 'view')}"
    out = Path(name) if name.endswith(".png") else config.ARTIFACT_DIR / "views" / f"{name}.png"
    view_render.render_spec(graph, spec, out, mentions=mentions)
    log.info("view rendered: %s", out)


def stage_serve(args: argparse.Namespace) -> None:
    """Start the live viewer: the tab updates in place as the spec changes."""
    spec = view_module.load_spec(Path(args.spec) if args.spec else None)
    if args.set:
        view_module.apply_overrides(spec, args.set)
    view_module.validate_spec(spec)
    graph = _graph_for_spec(spec)

    mentions_path = config.RAW_DIR / "mentions.csv"
    mentions = pd.read_csv(mentions_path) if mentions_path.exists() else None

    log.info("graph for the live view: %d nodes, %d edges", graph.number_of_nodes(), graph.number_of_edges())
    server_module.serve(graph, spec, mentions, port=args.port)


def stage_set(args: argparse.Namespace) -> None:
    """Push a spec patch to a running viewer (no reload)."""
    if not args.set:
        sys.exit("nothing to set: pass --set key=value (repeatable)")
    payload = json.dumps({"set": args.set}).encode("utf-8")
    request = urllib.request.Request(
        f"http://127.0.0.1:{args.port}/spec",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            reply = json.loads(response.read())
    except urllib.error.URLError as exc:
        sys.exit(f"no viewer on port {args.port} ({exc}). Start one with `run.py serve`.")
    log.info("pushed: %s", reply)


def _dag() -> nx.DiGraph:
    edges_frame = dependency.load_edges()
    _, titles = edges.load_summaries()
    dag = dependency.build_dag(edges_frame, titles=set(titles))
    return dependency.break_cycles(dag)[0]


def stage_ego(args: argparse.Namespace) -> None:
    """Everything within N hops of one notion."""
    if not args.node:
        sys.exit("ego needs --node")
    dag = _dag()
    if args.node not in dag:
        sys.exit(f"{args.node!r} is not in the DAG ({dag.number_of_nodes()} nodes)")
    out = config.ARTIFACT_DIR / "queries" / "ego.png"
    render_focus.render_ego(dag, args.node, out, radius=args.radius)
    distances = nx.single_source_shortest_path_length(dag.to_undirected(), args.node, cutoff=args.radius)
    by_hop: dict[int, list[str]] = {}
    for node, distance in distances.items():
        if node != args.node:
            by_hop.setdefault(distance, []).append(node)
    for distance in sorted(by_hop):
        log.info("%d hop(s): %d nodes — %s", distance, len(by_hop[distance]),
                 ", ".join(sorted(by_hop[distance])[:10]))
    log.info("rendered %s", out)


def stage_path(args: argparse.Namespace) -> None:
    """The prerequisite chain between two notions."""
    if not args.source or not args.target:
        sys.exit("path needs --from and --to")
    dag = _dag()
    for title in (args.source, args.target):
        if title not in dag:
            sys.exit(f"{title!r} is not in the DAG ({dag.number_of_nodes()} nodes)")
    chain, kind = dependency.path_between(dag, args.source, args.target)
    if chain is None:
        sys.exit(f"no path from {args.source!r} to {args.target!r} in the current edge set")
    log.info("%s path (%d steps): %s", kind, len(chain), " -> ".join(chain))
    if kind == "undirected":
        log.warning("this is not a directed prerequisite chain; the direction is mixed")
    out = config.ARTIFACT_DIR / "queries" / "path.png"
    render_focus.render_chain(chain, out, kind=kind)
    log.info("rendered %s", out)


def stage_describe(args: argparse.Namespace) -> None:
    """Text summary of one notion."""
    if not args.node:
        sys.exit("describe needs --node")
    dag = _dag()
    info = dependency.describe(dag, args.node)
    if not info.get("present"):
        sys.exit(f"{args.node!r} is not in the DAG ({dag.number_of_nodes()} nodes)")
    log.info("%s: %d prerequisites, %d enabled (%d in / %d out)",
             info["node"], info["prerequisites"], info["enables"],
             info["in_degree"], info["out_degree"])
    log.info("needs directly: %s", ", ".join(info["immediate_prerequisites"]) or "nothing")
    log.info("enables directly: %s", ", ".join(info["immediate_enables"]) or "nothing")
    if info["longest_chain_through"]:
        log.info("longest chain through it: %s", " -> ".join(info["longest_chain_through"]))


def stage_gephi(args: argparse.Namespace) -> None:
    """Write a GEXF that carries our layout, ready to open in Gephi."""
    spec = view_module.load_spec(Path(args.spec) if args.spec else None)
    graph = _graph_for_spec(spec)
    positions = render.forceatlas2_positions(graph)
    out = config.ARTIFACT_DIR / "gephi" / "graph_with_positions.gexf"
    export.write_gexf(graph, out, positions=positions)
    log.info("open this in Gephi: %s (%d nodes with positions)", out, len(positions))


def stage_import_layout(args: argparse.Namespace) -> None:
    """Adopt coordinates from a GEXF — e.g. one you arranged by hand in Gephi."""
    if not args.gexf:
        sys.exit("import-layout needs --gexf path/to/file.gexf")
    source = Path(args.gexf)
    if not source.exists():
        sys.exit(f"no such file: {source}")

    spec = view_module.load_spec(Path(args.spec) if args.spec else None)
    graph = _graph_for_spec(spec)
    positions = render.positions_from_gexf(source)
    if not positions:
        sys.exit(f"{source} contains no viz:position elements")
    unknown = [node for node in positions if node not in graph]
    if unknown:
        log.warning(
            "%d positions refer to nodes not in the graph (first: %s)",
            len(unknown), ", ".join(unknown[:3]),
        )
    installed = render.install_positions(graph, positions)
    log.info("layout installed from %s -> %s", source.name, installed.name)


def stage_gephi_layout(args: argparse.Namespace) -> None:
    """Run Gephi's own ForceAtlas2 headlessly, then adopt its coordinates.

    This is the parity the user asked for: Gephi's Toolkit is Gephi's engine
    without the window, so the layout is produced by Gephi itself rather than by
    a reimplementation -- and it is far faster (300 iterations in ~0.9 s).
    """
    import subprocess

    jar = config.DATA_DIR / "tools" / "gephi-toolkit-all.jar"
    source_java = config.PROJECT_ROOT / "tools" / "GephiToolkitLayout.java"
    if not jar.exists():
        sys.exit(
            f"missing {jar}. Fetch it once with:\n"
            "  curl -sL -o data/tools/gephi-toolkit-all.jar "
            "https://repo1.maven.org/maven2/org/gephi/gephi-toolkit/0.10.1/"
            "gephi-toolkit-0.10.1-all.jar"
        )

    spec = view_module.load_spec(Path(args.spec) if args.spec else None)
    graph = _graph_for_spec(spec)

    gexf = config.ARTIFACT_DIR / "gephi" / "graph_with_positions.gexf"
    if not gexf.exists():
        export.write_gexf(graph, gexf, positions=render.forceatlas2_positions(graph))

    tsv = config.ARTIFACT_DIR / "gephi" / "gephi_layout.tsv"
    command = [
        "java", "-cp", str(jar), str(source_java),
        str(gexf), str(tsv), str(args.iterations), "true" if args.linlog else "false",
    ]
    log.info("running Gephi Toolkit: %d iterations (linlog=%s)", args.iterations, args.linlog)
    result = subprocess.run(command, capture_output=True, text=True, timeout=600)
    for line in (result.stdout or "").splitlines():
        if line.startswith("Gephi") or line.startswith("wrote"):
            log.info("%s", line)
    if result.returncode != 0:
        sys.exit(f"Gephi Toolkit failed:\n{result.stderr[-2000:]}")

    positions = render.positions_from_tsv(tsv)
    if not positions:
        sys.exit(f"Gephi wrote no positions to {tsv}")
    render.install_positions(graph, positions)
    out = config.ARTIFACT_DIR / "views" / "gephi_toolkit.png"
    view_render.render_spec(graph, {**spec, "layout": "forceatlas2"}, out, mentions=None)
    log.info("rendered %s", out)


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
    "dependency": (stage_dependency, "build + render the prerequisite DAG"),
    "view": (stage_view, "render a graph from a declarative spec"),
    "ego": (stage_ego, "nodes within N hops of one notion"),
    "path": (stage_path, "prerequisite chain between two notions"),
    "describe": (stage_describe, "text summary of one notion"),
    "gephi": (stage_gephi, "export a GEXF carrying our layout, for Gephi"),
    "import-layout": (stage_import_layout, "adopt coordinates from a GEXF"),
    "gephi-layout": (stage_gephi_layout, "run Gephi's own ForceAtlas2 headlessly"),
    "serve": (stage_serve, "start the live view server"),
    "set": (stage_set, "push a spec patch to the running viewer"),
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
    parser.add_argument("--spec", default="", help="view/dependency: spec file (yaml or json)")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="view stage: patch the spec, e.g. --set node.size=\"degree ** 1.5\"")
    parser.add_argument("--out", default="", help="view stage: output path")
    parser.add_argument("--port", type=int, default=8765, help="serve/set stage: HTTP port")
    parser.add_argument("--gexf", default="", help="import-layout: GEXF file to read positions from")
    parser.add_argument("--iterations", type=int, default=300, help="gephi-layout: ForceAtlas2 iterations")
    parser.add_argument("--linlog", action="store_true", help="gephi-layout: ForceAtlas2 LinLog mode")
    parser.add_argument("--node", default="", help="ego/describe: the notion")
    parser.add_argument("--from", dest="source", default="", help="path: starting notion")
    parser.add_argument("--to", dest="target", default="", help="path: target notion")
    parser.add_argument("--focus", default="", help="dependency stage: render this notion's ego view")
    parser.add_argument("--radius", type=int, default=2, help="dependency stage: ego radius")
    parser.add_argument("--min-confidence", type=float, default=0.0,
                        help="dependency stage: drop edges below this confidence")
    parser.add_argument(
        "--include-classification",
        action="store_true",
        help="dependency stage: include Wikidata P279 subclass edges (taxonomy, not prerequisite)",
    )
    parser.add_argument(
        "--no-proof",
        action="store_true",
        help="edges stage: skip the proof-section extractor (no wikitext fetching)",
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


# --------------------------------------------------------------------------- #
# argv normalisation
# --------------------------------------------------------------------------- #

#: Options that live on the main parser; argparse requires them *before* the
#: subcommand. Nobody types `run.py --set node.size=degree set`, so the order is
#: normalised instead of documented and hoped for.
GLOBAL_FLAGS = {
    "--verbose", "--no-html", "--largest-component", "--no-wikidata", "--no-proof",
    "--include-classification", "--linlog",
}
GLOBAL_VALUED = {
    "--set", "--spec", "--out", "--port", "--focus", "--radius", "--min-confidence",
    "--labels", "--view", "--max-degree", "--legacy-csv",
    "--node", "--from", "--to", "--gexf", "--iterations",
}


def normalise_argv(argv: list[str]) -> list[str]:
    """Move global options in front of the subcommand, preserving their order."""
    head: list[str] = []
    tail: list[str] = []
    index = 0
    while index < len(argv):
        token = argv[index]
        name = token.split("=", 1)[0]
        if name in GLOBAL_FLAGS:
            head.append(token)
            index += 1
            continue
        if name in GLOBAL_VALUED:
            head.append(token)
            if "=" not in token and index + 1 < len(argv):
                head.append(argv[index + 1])
                index += 2
                continue
            index += 1
            continue
        tail.append(token)
        index += 1
    return head + tail


def main(argv: list[str] | None = None) -> int:
    from mathkg import setup_logging

    raw = sys.argv[1:] if argv is None else argv
    args = build_parser().parse_args(normalise_argv(list(raw)))
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
