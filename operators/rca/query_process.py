from __future__ import annotations

from typing import Any

from traj_analyzer.operators.base import FeatureSpec, NoParams, Operator
from traj_analyzer.schema import Trajectory

from rca._parse import QUERY, calls, outcome, query_shape
from rca._queries import SAME_QUESTION, move

RATE = (0.0, 1.0)


def outputs(params: NoParams) -> list[FeatureSpec]:
    def rate(name: str, description: str) -> FeatureSpec:
        return FeatureSpec(name=name, type="scalar", range=RATE, description=description)

    return [
        rate("question_distinct_ratio",
             "Distinct questions among the parsed queries, divided by their number; a question is the telemetry "
             "kinds a query reads together with the columns that scope it. Empty when no query parses."),
        rate("question_return_rate",
             "Share of parsed queries that ask a question asked earlier in the trajectory, though not by the query "
             "right before. Empty when no query parses."),
        rate("unscoped_query_share",
             "Share of parsed queries without any filter, group or join column. Empty when no query parses."),
        rate("row_query_share",
             "Share of parsed queries without an aggregate, which read rows as they are. Empty when no query parses."),
        rate("baseline_paired_share",
             "Among the distinct questions with their measured columns asked of the incident window, the share "
             "also asked of the baseline window. Empty when no parsed query reads the incident window."),
        FeatureSpec(name="longest_narrowing_run", type="scalar",
                    description="Most consecutive queries that each add scope columns to the query before."),
        FeatureSpec(name="widest_scope", type="scalar",
                    description="Most filter, group and join columns in one query."),
        rate("switch_rate", "Share of queries after the first that read other telemetry kinds than the query "
                            "before. Empty with fewer than two queries."),
        rate("error_retry_rate",
             "Among the queries that follow a failed query, the share that ask the same question again: same "
             "kinds and same scope. Empty when no query follows a failed one."),
        rate("error_switch_rate",
             "Among the queries that follow a failed query, the share that read other telemetry kinds. Empty when "
             "no query follows a failed one."),
        rate("empty_revalue_rate",
             "Among the queries that follow a query that returned no row, the share that keep the question and "
             "change the filter values. Empty when no query follows an empty result."),
    ]


def _share(hits: int, total: int) -> float | None:
    return hits / total if total else None


def compute(trajectory: Trajectory, params: NoParams) -> dict[str, Any]:
    queries = [c for c in calls(trajectory) if c.name == QUERY]
    shapes = [query_shape(c) for c in queries]
    outcomes = [outcome(c) for c in queries]
    moves = [move(before, now) for before, now in zip(shapes, shapes[1:], strict=False)]
    parsed = [shape for shape in shapes if shape.facets is not None]

    seen: set[Any] = set()
    returns = 0
    previous = None
    for shape in parsed:
        question = (shape.kinds, shape.facets.scope)
        returns += question in seen and question != previous
        seen.add(question)
        previous = question

    asked: dict[Any, set[str]] = {}
    for shape in parsed:
        asked.setdefault((shape.kinds, shape.facets.scope, shape.facets.measures), set()).update(shape.windows)
    incident = [windows for windows in asked.values() if "abnormal" in windows]

    run = longest = 0
    for step in moves:
        run = run + 1 if step == "narrow" else 0
        longest = max(longest, run)

    after_error = [step for step, before in zip(moves, outcomes, strict=False) if before == "error"]
    after_empty = [step for step, before in zip(moves, outcomes, strict=False) if before == "empty"]
    return {
        "question_distinct_ratio": _share(len(seen), len(parsed)),
        "question_return_rate": _share(returns, len(parsed)),
        "unscoped_query_share": _share(sum(not shape.facets.scope for shape in parsed), len(parsed)),
        "row_query_share": _share(sum(not shape.facets.aggregates for shape in parsed), len(parsed)),
        "baseline_paired_share": _share(sum("normal" in windows for windows in incident), len(incident)),
        "longest_narrowing_run": float(longest),
        "widest_scope": float(max((len(shape.facets.scope) for shape in parsed), default=0)),
        "switch_rate": _share(moves.count("switch"), len(moves)),
        "error_retry_rate": _share(sum(step in SAME_QUESTION for step in after_error), len(after_error)),
        "error_switch_rate": _share(after_error.count("switch"), len(after_error)),
        "empty_revalue_rate": _share(after_empty.count("revalue"), len(after_empty)),
    }


OPERATOR = Operator(
    kind="code",
    description="How the investigation moves from query to query: how many questions it asks, whether it returns "
                "to them, pairs them with the baseline, narrows them, and what it does after an error or an "
                "empty result.",
    tags=("agent", "rca"),
    outputs=outputs,
    compute=compute,
)
