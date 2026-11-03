"""View specs and the expression evaluator.

The expression evaluator is the one place where a *document* could try to run
code, so the rejection tests matter more than the happy-path ones.
"""

from __future__ import annotations

import pytest

from mathkg.view import (
    ExpressionError,
    apply_overrides,
    compile_expression,
    is_expression,
    load_spec,
)

NAMES = {"degree", "sitelinks", "kind", "occurrences", "confidence"}


def eval_expr(source, **row):
    return compile_expression(source, NAMES)(row)


# --- happy path ------------------------------------------------------------- #

@pytest.mark.parametrize(
    ("expression", "row", "expected"),
    [
        ("6 + 2 * degree", {"degree": 3}, 12),
        ("degree ** 2", {"degree": 4}, 16),
        ("log(1 + sitelinks)", {"sitelinks": 99}, pytest.approx(4.605, abs=0.01)),
        ("max(degree, 5)", {"degree": 2}, 5),
        ("20 if degree > 10 else 5", {"degree": 12}, 20),
        ("20 if degree > 10 else 5", {"degree": 3}, 5),
        ("1 / 0", {}, 0.0),          # guarded: no ZeroDivisionError
        ("2 - 5", {}, -3),
    ],
)
def test_expressions_evaluate(expression, row, expected):
    assert eval_expr(expression, **row) == expected


def test_missing_column_is_an_error_not_a_silent_zero():
    with pytest.raises(ExpressionError):
        eval_expr("unknown_column * 2", degree=1)


# --- rejection -------------------------------------------------------------- #

@pytest.mark.parametrize(
    "hostile",
    [
        "__import__('os').system('echo pwned')",
        "open('/etc/passwd').read()",
        "degree.__class__",
        "().__class__.__bases__",
        "[x for x in range(10)]",
        "lambda: 1",
        "degree[0]",
        "{'a': 1}",
        "exec('x=1')",
        "eval('1+1')",
        "getattr(degree, 'real')",
        "print('hi')",
    ],
)
def test_hostile_expressions_are_rejected(hostile):
    with pytest.raises(ExpressionError):
        compile_expression(hostile, NAMES)


def test_syntax_errors_are_reported_as_expression_errors():
    with pytest.raises(ExpressionError):
        compile_expression("degree +", NAMES)


def test_keyword_arguments_are_rejected():
    with pytest.raises(ExpressionError):
        compile_expression("round(degree, ndigits=2)", NAMES)


# --- spec handling ---------------------------------------------------------- #

def test_apply_overrides_patches_nested_keys():
    spec = load_spec()
    apply_overrides(spec, ["node.size=degree ** 1.5", "label.count=80"])
    assert spec["node"]["size"] == "degree ** 1.5"
    assert spec["label"]["count"] == 80


def test_apply_overrides_parses_json_values():
    spec = load_spec()
    apply_overrides(spec, ["filter.min_degree=3", "edge.opacity=0.4"])
    assert spec["filter"]["min_degree"] == 3
    assert spec["edge"]["opacity"] == 0.4


def test_apply_overrides_requires_an_equals_sign():
    with pytest.raises(ValueError):
        apply_overrides(load_spec(), ["node.size"])


def test_spec_files_merge_onto_defaults(tmp_path):
    path = tmp_path / "x.yaml"
    path.write_text("name: mine\nnode:\n  size: degree\n", encoding="utf-8")

    spec = load_spec(path)

    assert spec["name"] == "mine"
    assert spec["node"]["size"] == "degree"
    # untouched keys survive the merge
    assert spec["edge"]["opacity"] == 0.10
    assert spec["layout"] == "layered"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("degree", True),
        ("6 + degree", True),
        ("#4E79A7", False),
        (0.5, False),
        ("degree", True),
    ],
)
def test_is_expression(value, expected):
    assert is_expression(value) is expected


def test_bare_column_name_is_an_expression():
    """`formula: confidence` must read the column, not produce the string.

    This is a regression test: the live server crashed with 'could not convert
    string to float: confidence' because a bare column name was not recognised
    as an expression.
    """
    assert is_expression("confidence", {"confidence", "occurrences"}) is True
    assert eval_expr("confidence", confidence=0.75) == 0.75


def test_unknown_bare_string_stays_literal():
    assert is_expression("not_a_column", {"confidence"}) is False


# --- ForceAtlas2 ------------------------------------------------------------ #

def test_forceatlas2_is_deterministic_and_cached(tmp_path, monkeypatch):
    """Gephi's layout must be reproducible, and cheap on the second call."""
    import networkx as nx

    from mathkg import render

    monkeypatch.setattr(render, "LAYOUT_CACHE", tmp_path)
    graph = nx.cycle_graph(8)

    first = render.forceatlas2_positions(graph, seed=7, max_iter=20)
    second = render.forceatlas2_positions(graph, seed=7, max_iter=20)  # cache hit

    assert set(first) == set(graph.nodes)
    assert first == second
    assert list(tmp_path.glob("*_fa2.json"))


def test_forceatlas2_different_seeds_differ(tmp_path, monkeypatch):
    import networkx as nx

    from mathkg import render

    monkeypatch.setattr(render, "LAYOUT_CACHE", tmp_path)
    graph = nx.cycle_graph(6)

    a = render.forceatlas2_positions(graph, seed=1, max_iter=15)
    b = render.forceatlas2_positions(graph, seed=2, max_iter=15)

    assert a != b
