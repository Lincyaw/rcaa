from __future__ import annotations

from collections import Counter
from typing import Any

from traj_analyzer.operators.base import FeatureSpec, NoParams, Operator
from traj_analyzer.schema import Trajectory

from rca._parse import QUERY, calls, filter_values, query_atoms


def outputs(params: NoParams) -> list[FeatureSpec]:
    return [
        FeatureSpec(name="query_files", type="map", range=(0, 1),
                    description="Share of queries that read each telemetry file, named <window>_<kind> such as "
                                "abnormal_traces; empty when the agent ran no query."),
        FeatureSpec(name="query_atoms", type="map", range=(0, 1),
                    description="Share of parsed queries in which a data column takes a role, as <role>:<column>. "
                                "Roles: filter (WHERE or HAVING), group (GROUP BY), order (ORDER BY), join (JOIN ON), "
                                "show (select list outside aggregates), and the aggregate around it in the select "
                                "list: count, avg, sum, min, max, quantile (percentiles and median) or stddev; "
                                "SELECT * gives show:* and COUNT(*) gives count:*. The FILTER clause of an aggregate "
                                "counts as filter, and GROUP BY and ORDER BY by ordinal or alias count for the column "
                                "they stand for. Empty when no query parses."),
        FeatureSpec(name="query_filter_values", type="map", range=(0, 1),
                    description="Share of parsed queries that compare a data column with a string value, as "
                                "'<column> <op> <value>' with op =, !=, like or ilike, such as 'level = ERROR', "
                                "'metric like %cpu%' or 'lower(level) = error'. Values keep the agent's exact text, "
                                "since a value the data does not hold matches nothing without an error. Service and "
                                "span names, trace, span and pod ids, times, and names the query defines itself are "
                                "left out; empty when no query parses."),
    ]


def compute(trajectory: Trajectory, params: NoParams) -> dict[str, Any]:
    queries = [c for c in calls(trajectory) if c.name == QUERY]
    files: Counter[str] = Counter()
    atoms: Counter[str] = Counter()
    values: Counter[str] = Counter()
    parsed = 0
    for call in queries:
        files.update({f"{window}_{kind}" for window, kind in call.telemetry})
        tree = call.tree
        if tree is None:
            continue
        parsed += 1
        atoms.update(query_atoms(tree))
        values.update(filter_values(tree))
    return {
        "query_files": {name: n / len(queries) for name, n in files.items()} if queries else None,
        "query_atoms": {name: n / parsed for name, n in atoms.items()} if parsed else None,
        "query_filter_values": {name: n / parsed for name, n in values.items()} if parsed else None,
    }


OPERATOR = Operator(
    kind="code",
    description="What the agent's SQL is made of: files read, the role each column takes, and the values it filters "
                "on, each as the share of queries.",
    tags=("agent", "rca"),
    outputs=outputs,
    compute=compute,
)
