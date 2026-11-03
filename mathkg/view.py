"""Declarative view specs: style the graph without touching Python.

The point of this module is that "make node size proportional to degree" must be
a change to a *document*, not an edit to matplotlib code by an agent. A view is
therefore a small dict of expressions, evaluated against per-node and per-edge
columns:

    node:
      size: "degree ** 1.2"          # or "20 if degree > 10 else 6"
      color: "#4E79A7"               # or an expression returning a hex string
    edge:
      width: "0.2 + 1.5 * weight"
    edge_weight:
      formula: "occurrences * (1 - relative_position) / log(1 + popularity)"

Expressions are evaluated by :func:`compile_expression`, a small AST
interpreter with a whitelist of names and functions. It never calls ``eval``:
a view file is data, and data must not be able to run code.
"""

from __future__ import annotations

import ast
import json
import logging
import math
import re
from pathlib import Path
from typing import Any

import networkx as nx
import pandas as pd
import yaml

from . import config

log = logging.getLogger(__name__)

VIEWS_DIR = config.PROJECT_ROOT / "views"


# --------------------------------------------------------------------------- #
# The default spec
# --------------------------------------------------------------------------- #

DEFAULT_SPEC: dict[str, Any] = {
    "name": "default",
    "source": "dependency",          # dependency | link
    "layout": "layered",             # layered | spring
    "node": {
        "size": "6 + 0.9 * degree",
        "color": "#4E79A7",
        "opacity": 0.9,
        "shape": "o",
    },
    "label": {
        "size": 7.0,
        "count": 40,
        "min_separation": 0.11,
    },
    "edge": {
        "width": "0.25",
        "color": "#8899aa",
        "opacity": 0.10,
    },
    "edge_weight": {
        "formula": "confidence",
        "minimum": 0.0,
    },
    "filter": {
        "min_degree": 0,
        "kinds": [],
        "component": None,           # "largest" or None
        "ego": None,                 # {"node": "...", "radius": 2}
    },
}

#: Functions and constants an expression may use.
FUNCTIONS: dict[str, Any] = {
    "min": min,
    "max": max,
    "abs": abs,
    "sqrt": math.sqrt,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "floor": math.floor,
    "ceil": math.ceil,
    "round": round,
    "pow": pow,
}

CONSTANTS: dict[str, float] = {"pi": math.pi, "e": math.e}


class ExpressionError(ValueError):
    """Raised when a view expression is invalid or uses a forbidden construct."""


# --------------------------------------------------------------------------- #
# Safe expression evaluation (no eval)
# --------------------------------------------------------------------------- #

_ALLOWED_NODES = (
    ast.Expression,
    ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.IfExp,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod, ast.FloorDiv,
    ast.USub, ast.UAdd,
    ast.And, ast.Or, ast.Not,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
    ast.Call, ast.Name, ast.Constant, ast.Load,
)


def _check(tree: ast.Expression, allowed_names: set[str]) -> None:
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise ExpressionError(
                f"forbidden construct {type(node).__name__} in expression"
            )
        if (
            isinstance(node, ast.Name)
            and node.id not in allowed_names
            and node.id not in FUNCTIONS
            and node.id not in CONSTANTS
        ):
            raise ExpressionError(f"unknown name {node.id!r} in expression")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in FUNCTIONS:
                raise ExpressionError("only whitelisted functions may be called")
            if node.keywords:
                raise ExpressionError("keyword arguments are not allowed")
        if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float, str, bool)):
            raise ExpressionError(f"literal {node.value!r} is not allowed")


def compile_expression(source: str, allowed_names: set[str]) -> CompiledExpression:
    """Parse and validate an expression against a whitelist of column names."""
    try:
        tree = ast.parse(str(source), mode="eval")
    except SyntaxError as exc:
        raise ExpressionError(f"invalid expression {source!r}: {exc}") from exc
    _check(tree, allowed_names)
    return CompiledExpression(tree, str(source))


class CompiledExpression:
    """An expression bound to a column whitelist, evaluated per row."""

    def __init__(self, tree: ast.Expression, source: str) -> None:
        self.tree = tree
        self.source = source

    def __call__(self, row: dict[str, Any]) -> Any:
        return self._eval(self.tree.body, row)

    def _eval(self, node: ast.AST, row: dict[str, Any]) -> Any:
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            if node.id in row:
                return row[node.id]
            if node.id in CONSTANTS:
                return CONSTANTS[node.id]
            return FUNCTIONS.get(node.id)
        if isinstance(node, ast.BinOp):
            left = self._eval(node.left, row)
            right = self._eval(node.right, row)
            operator = node.op
            if isinstance(operator, ast.Add):
                return left + right
            if isinstance(operator, ast.Sub):
                return left - right
            if isinstance(operator, ast.Mult):
                return left * right
            if isinstance(operator, ast.Div):
                return left / right if right else 0.0
            if isinstance(operator, ast.Pow):
                return left ** right
            if isinstance(operator, ast.Mod):
                return left % right if right else 0.0
            if isinstance(operator, ast.FloorDiv):
                return left // right if right else 0.0
            raise ExpressionError("unsupported operator")
        if isinstance(node, ast.UnaryOp):
            value = self._eval(node.operand, row)
            return -value if isinstance(node.op, ast.USub) else +value
        if isinstance(node, ast.BoolOp):
            values = [self._eval(value, row) for value in node.values]
            return all(values) if isinstance(node.op, ast.And) else any(values)
        if isinstance(node, ast.Compare):
            left = self._eval(node.left, row)
            for operator, comparator in zip(node.ops, node.comparators, strict=False):
                right = self._eval(comparator, row)
                if not _compare(operator, left, right):
                    return False
                left = right
            return True
        if isinstance(node, ast.IfExp):
            return (
                self._eval(node.body, row)
                if self._eval(node.test, row)
                else self._eval(node.orelse, row)
            )
        if isinstance(node, ast.Call):
            function = FUNCTIONS[node.func.id]  # validated in _check
            return function(*[self._eval(arg, row) for arg in node.args])
        raise ExpressionError(f"cannot evaluate {type(node).__name__}")


def _compare(operator: ast.cmpop, left: Any, right: Any) -> bool:
    if isinstance(operator, ast.Eq):
        return left == right
    if isinstance(operator, ast.NotEq):
        return left != right
    if isinstance(operator, ast.Lt):
        return left < right
    if isinstance(operator, ast.LtE):
        return left <= right
    if isinstance(operator, ast.Gt):
        return left > right
    if isinstance(operator, ast.GtE):
        return left >= right
    raise ExpressionError("unsupported comparison")


def is_expression(value: Any, allowed: set[str] | None = None) -> bool:
    """Is this spec value an expression, or a literal (a colour, a number)?

    A bare column name counts as an expression: ``edge_weight.formula:
    confidence`` must evaluate to the row's confidence, not to the string
    "confidence". Ignoring that produced a ``could not convert string to float``
    in the live server.
    """
    if not isinstance(value, str):
        return False
    text = value.strip()
    if any(token in value for token in "+-*/()"):
        return True
    if text.startswith("#") or text in CONSTANTS:
        return False
    if allowed is not None:
        return text in allowed
    # No column list supplied: any bare identifier is treated as an expression,
    # which keeps `is_expression("degree")` true for callers that only have the
    # value in hand.
    return bool(re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]{0,40}", text))


def evaluate(value: Any, row: dict[str, Any], allowed: set[str]) -> Any:
    """Evaluate a spec value if it is an expression, else return it unchanged."""
    if not is_expression(value, allowed):
        return value
    return compile_expression(value, allowed)(row)


# --------------------------------------------------------------------------- #
# Spec loading / patching
# --------------------------------------------------------------------------- #

def load_spec(path: Path | None = None) -> dict[str, Any]:
    """Load a YAML/JSON spec, deep-merged onto DEFAULT_SPEC."""
    spec = json.loads(json.dumps(DEFAULT_SPEC))  # deep copy
    if path is None:
        return spec

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    text = path.read_text(encoding="utf-8")
    loaded = yaml.safe_load(text) if path.suffix in {".yaml", ".yml"} else json.loads(text)
    return _deep_merge(spec, loaded or {})


def _deep_merge(base: dict, override: dict) -> dict:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def apply_overrides(spec: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply ``dotted.key=value`` patches, e.g. ``node.size=degree ** 1.5``."""
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override {item!r} must look like key.path=value")
        key, _, raw = item.partition("=")
        target = spec
        parts = key.strip().split(".")
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        try:
            value: Any = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        target[parts[-1]] = value
        log.info("spec: %s = %r", key.strip(), value)
    return spec


def save_spec(spec: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix in {".yaml", ".yml"}:
        path.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
    else:
        path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# Column extraction
# --------------------------------------------------------------------------- #

def node_rows(graph: nx.Graph) -> dict[str, dict[str, Any]]:
    """Per-node columns available to expressions."""
    degree = dict(graph.degree())
    return {
        node: {
            "degree": degree.get(node, 0),
            "sitelinks": float(data.get("sitelinks", 0) or 0),
            "kind": str(data.get("kind", "")) or "unknown",
            "is_anchor": bool(data.get("anchor", False)),
        }
        for node, data in graph.nodes(data=True)
    }


def edge_rows(graph: nx.Graph, mentions: pd.DataFrame | None) -> dict[tuple[str, str], dict[str, Any]]:
    """Per-edge columns available to expressions, including mention stats."""
    popularity = {node: float(data.get("sitelinks", 0) or 0) for node, data in graph.nodes(data=True)}
    stats: dict[tuple[str, str], dict[str, Any]] = {}

    if mentions is not None and not mentions.empty:
        indexed = mentions.set_index(["source", "target"])
        # Edges are stored prerequisite -> dependent, which is the reverse of the
        # extraction direction; look up both orientations and keep the best hit.
        for source, target, data in graph.edges(data=True):
            row = None
            for key in ((source, target), (target, source)):
                if key in indexed.index:
                    row = indexed.loc[key]
                    if isinstance(row, pd.DataFrame):
                        row = row.iloc[0]
                    break
            base = {
                "confidence": float(data.get("confidence", 0.5)),
                "popularity": popularity.get(target, 0.0),
                "occurrences": 0.0,
                "relative_position": 0.0,
                "in_lead": 0.0,
                "in_proof": 0.0,
            }
            if row is not None:
                base.update(
                    occurrences=float(row.get("occurrences", 0) or 0),
                    relative_position=float(row.get("relative_position", 0) or 0),
                    in_lead=float(bool(row.get("in_lead", False))),
                    in_proof=float(bool(row.get("in_proof", False))),
                )
            stats[(source, target)] = base
        return stats

    for source, target, data in graph.edges(data=True):
        stats[(source, target)] = {
            "confidence": float(data.get("confidence", 0.5)),
            "popularity": popularity.get(target, 0.0),
            "occurrences": 0.0,
            "relative_position": 0.0,
            "in_lead": 0.0,
            "in_proof": 0.0,
        }
    return stats


#: Columns an expression may reference, by context.
NODE_COLUMNS = {"degree", "sitelinks", "kind", "is_anchor"}
EDGE_COLUMNS = {
    "confidence", "popularity", "occurrences", "relative_position", "in_lead", "in_proof",
}
LABEL_COLUMNS: set[str] = set()


LAYOUTS = {"physics", "installed", "gephi", "layered", "forceatlas2", "spring"}


def validate_spec(spec: dict[str, Any]) -> None:
    """Compile every expression in a spec, so a bad one fails immediately.

    Without this, a typo in ``node.size`` was stored silently and only surfaced
    as a logged warning on the next render -- the user would change the spec and
    see nothing happen, with no error. Raises :class:`ExpressionError`.
    """
    checks = [
        (spec.get("node", {}), ("size", "color", "opacity"), NODE_COLUMNS),
        (spec.get("label", {}), ("size", "count", "min_separation"), LABEL_COLUMNS),
        (spec.get("edge", {}), ("width", "opacity"), EDGE_COLUMNS),
        (spec.get("edge_weight", {}), ("formula", "minimum"), EDGE_COLUMNS),
    ]
    for block, keys, columns in checks:
        for key in keys:
            value = (block or {}).get(key)
            if is_expression(value, columns):
                compile_expression(str(value), columns)

    layout = str(spec.get("layout", ""))
    if layout not in LAYOUTS:
        raise ExpressionError(
            f"unknown layout {layout!r}; expected one of {sorted(LAYOUTS)}"
            " (an unclosed quote in your shell command often ends up here)"
        )


# --------------------------------------------------------------------------- #
# Colour by rule
# --------------------------------------------------------------------------- #

DEFAULT_PALETTE = ["#264653", "#2a9d8f", "#e9c46a", "#f4a261", "#e76f51",
                   "#6a4c93", "#1982c4", "#8ac926"]


def node_colors(
    graph, spec: dict, columns: dict[str, dict[str, Any]]
) -> dict[str, str]:
    """Assign a colour per node: either one literal, or a binned rule.

    ``node.color_by`` takes a column and a palette; values are split into equal
    quantile bins so the colouring is robust to the heavy tails these graphs have
    (degree 1 dominates, one node has degree 35). Explicit ``bins`` may be given
    instead of quantiles.
    """
    node_spec = spec.get("node") or {}
    literal = node_spec.get("color", "#4E79A7")

    rule = node_spec.get("color_by")
    if not rule:
        return {node: literal for node in graph.nodes}

    column = rule.get("column", "degree")
    palette = rule.get("palette") or DEFAULT_PALETTE
    values = {node: float(columns.get(node, {}).get(column, 0) or 0) for node in graph.nodes}

    bins = rule.get("bins")
    if bins:
        edges_list = sorted({float(b) for b in bins})
    else:
        ordered = sorted(values.values())
        count = max(1, len(palette) - 1)
        edges_list = [
            ordered[min(len(ordered) - 1, int(len(ordered) * (i + 1) / count))]
            for i in range(count - 1)
        ]
        edges_list = sorted(set(edges_list))

    def colour_for(value: float) -> str:
        index = 0
        for edge_value in edges_list:
            if value > edge_value:
                index += 1
        return palette[min(index, len(palette) - 1)]

    return {node: colour_for(value) for node, value in values.items()}
