from __future__ import annotations

from rca._parse import NONE, UNPARSED, QueryShape

QUERY_LEVELS = ["kinds", "scope", "measures", "windows", "aggregates", "shown", "values"]
# Moves that ask the same question again: the telemetry kinds and the scope are those of the query before.
SAME_QUESTION = ("repeat", "revalue", "reshape")


def _join(labels: tuple[str, ...]) -> str:
    return " ".join(labels) or NONE


def kinds_label(shape: QueryShape) -> str:
    return "+".join(shape.kinds) or NONE


def query_path(shape: QueryShape) -> list[str]:
    """One query as a path through QUERY_LEVELS, from what it asks to how it is written."""
    facets = shape.facets
    if facets is None:
        return [kinds_label(shape), *([UNPARSED] * (len(QUERY_LEVELS) - 1))]
    return [kinds_label(shape), _join(facets.scope), _join(facets.measures), "+".join(shape.windows) or NONE,
            _join(facets.aggregates), _join(facets.shown), " ; ".join(facets.values) or NONE]


def move(before: QueryShape, now: QueryShape) -> str:
    """How a query relates to the query before it.

    switch when the telemetry kinds differ; otherwise repeat for the same query path, revalue for the same scope
    with other filter values, reshape for the same scope and values with other measures or shown columns, narrow
    when the scope gains columns, widen when it loses columns, shift when it does both; unparsed when either query
    does not parse.
    """
    if before.facets is None or now.facets is None:
        return "unparsed"
    if before.kinds != now.kinds:
        return "switch"
    if before.facets.scope == now.facets.scope:
        if query_path(before) == query_path(now):
            return "repeat"
        return "revalue" if before.facets.values != now.facets.values else "reshape"
    old, new = set(before.facets.scope), set(now.facets.scope)
    if old < new:
        return "narrow"
    return "widen" if new < old else "shift"
