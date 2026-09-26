from __future__ import annotations

import json
from collections import Counter
from typing import Any

from traj_analyzer.operators.base import FeatureSpec, NoParams, Operator
from traj_analyzer.schema import Trajectory

from rca._parse import QUERY, STORED_CUT_MARKER, calls, error_kind, normalize_sql

ERROR_LABELS = {
    "token_budget": "The result was larger than the token budget and was replaced by an error.",
    "access_denied": "The query read a file that is not a telemetry parquet.",
    "file_not_found": "The query named a file, table or path that does not exist.",
    "type_overflow": "DuckDB failed on a numeric overflow or type conversion.",
    "sql_error": "DuckDB rejected the SQL: binder, parser or catalog errors.",
    "other": "Any other tool error.",
}


def outputs(params: NoParams) -> list[FeatureSpec]:
    return [
        FeatureSpec(name="n_turns", type="scalar", description="Assistant turns: runs of assistant steps between "
                                                               "tool results."),
        FeatureSpec(name="max_calls_per_turn", type="scalar", description="Most tool calls issued in one turn."),
        FeatureSpec(name="n_queries", type="scalar", description="Calls to query_parquet_files."),
        FeatureSpec(name="query_error_rate", type="scalar", range=(0, 1), thresholds={"high": 0.3},
                    description="Share of query_parquet_files calls whose result is an error."),
        FeatureSpec(name="error_kinds", type="set", labels=ERROR_LABELS,
                    description="Kinds of errors among all tool results."),
        FeatureSpec(name="unrecovered_error_rate", type="scalar", range=(0, 1),
                    description="Share of failed queries after which no later query on the same files succeeded."),
        FeatureSpec(name="duplicate_query_rate", type="scalar", range=(0, 1),
                    description="Share of queries whose whitespace-normalized SQL repeats an earlier query."),
        FeatureSpec(name="empty_result_rate", type="scalar", range=(0, 1),
                    description="Share of successful queries that returned an empty list."),
        FeatureSpec(name="silent_truncation_count", type="scalar",
                    description="Successful queries without a limit argument that returned exactly 10 rows, the "
                                "harness default, so the result was probably cut off."),
    ]


def compute(trajectory: Trajectory, params: NoParams) -> dict[str, Any]:
    all_calls = calls(trajectory)
    queries = [c for c in all_calls if c.name == QUERY]
    per_turn = Counter(c.turn for c in all_calls)
    errors = [c for c in all_calls if c.failed]
    failed_queries = [c for c in queries if c.failed]
    unrecovered = 0
    for failed in failed_queries:
        later = [c for c in queries if c.step.index > failed.step.index and c.files == failed.files]
        unrecovered += not any(not c.failed for c in later)
    seen: set[str] = set()
    duplicates = 0
    for query in queries:
        key = normalize_sql(query.sql)
        duplicates += key in seen
        seen.add(key)
    succeeded = [c for c in queries if c.succeeded]
    empty = sum(1 for c in succeeded if c.result is not None and c.result.content.strip() == "[]")
    truncated = 0
    for query in succeeded:
        assert query.result is not None
        text = query.result.content.strip()
        if "limit" not in query.args and text.startswith("[") and not text.endswith(STORED_CUT_MARKER.strip()):
            truncated += len(json.loads(text)) == 10
    return {
        "n_turns": float(max(per_turn) + 1 if per_turn else 0),
        "max_calls_per_turn": float(max(per_turn.values()) if per_turn else 0),
        "n_queries": float(len(queries)),
        "query_error_rate": len(failed_queries) / len(queries) if queries else None,
        "error_kinds": sorted({error_kind(c.result.content) for c in errors if c.result is not None}),
        "unrecovered_error_rate": unrecovered / len(failed_queries) if failed_queries else None,
        "duplicate_query_rate": duplicates / len(queries) if queries else None,
        "empty_result_rate": empty / len(succeeded) if succeeded else None,
        "silent_truncation_count": float(truncated),
    }


OPERATOR = Operator(
    kind="code",
    description="How fluently the agent used the RCABench tools: turns, parallelism, errors, repeats and truncation.",
    tags=("agent", "rca"),
    outputs=outputs,
    compute=compute,
)
