from __future__ import annotations

from typing import Any

from traj_analyzer.operators.base import FeatureSpec, NoParams, Operator
from traj_analyzer.schema import Trajectory

from rca._parse import QUERY, calls, outcome, query_shape
from rca._queries import QUERY_LEVELS, kinds_label, move, query_path

MOVE_LEVELS = ["move", "after", "kinds"]


def outputs(params: NoParams) -> list[FeatureSpec]:
    return [
        FeatureSpec(
            name="query_paths", type="paths", levels=QUERY_LEVELS,
            description="One path per query in order, each level refining the one before: the telemetry kinds it "
                        "reads (traces, logs, metrics, metrics_sum, metrics_histogram); the columns that scope it, "
                        "as filter:, group: and join: with the column; the columns it measures inside aggregates; "
                        "the windows it reads (abnormal, normal); the aggregates as <aggregate>:<column>; the columns "
                        "it shows or orders by; and the string values it filters on. A level without entries is "
                        "`-`, and the levels after kinds of a query sqlglot cannot parse are `(unparsed)`."),
        FeatureSpec(
            name="query_moves", type="paths", levels=MOVE_LEVELS,
            description="One path per query in order, relating it to the query before it. move: start for the "
                        "first query; switch when the telemetry kinds differ; otherwise repeat for the same query "
                        "path, revalue for the same scope with other filter values, reshape for the same scope and "
                        "values with other measures or shown columns, narrow when the scope gains columns, widen "
                        "when it loses columns, shift when it does both; unparsed when either query does not "
                        "parse. after: how the query before ended, as start, ok, empty, error or unanswered. "
                        "kinds: the telemetry kinds read, as <before>><now> for a switch."),
    ]


def compute(trajectory: Trajectory, params: NoParams) -> dict[str, Any]:
    queries = [c for c in calls(trajectory) if c.name == QUERY]
    shapes = [query_shape(c) for c in queries]
    moves = []
    for position, shape in enumerate(shapes):
        if position == 0:
            moves.append(["start", "start", kinds_label(shape)])
            continue
        before = shapes[position - 1]
        step = move(before, shape)
        described = f"{kinds_label(before)}>{kinds_label(shape)}" if step == "switch" else kinds_label(shape)
        moves.append([step, outcome(queries[position - 1]), described])
    return {"query_paths": [query_path(shape) for shape in shapes], "query_moves": moves}


OPERATOR = Operator(
    kind="code",
    description="Every query as a path through a tree of what it asks, and as a move from the query before it.",
    tags=("agent", "rca"),
    outputs=outputs,
    compute=compute,
)
