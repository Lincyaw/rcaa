from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import cached_property, lru_cache
from typing import Any

import sqlglot
from sqlglot import exp
from traj_analyzer.schema import Step, Trajectory

QUERY = "query_parquet_files"
# Conventions of adapters/rcabench_eval.py, which marks stored results that were cut and names two special steps.
STORED_CUT_MARKER = "\n[cut when stored]"
FORCE_SUBMIT = "force_submit"

# Error texts of the RCABench harness, matched in order; the first match names the kind.
ERROR_KINDS = [
    ("token_budget", re.compile(r"exceeds token budget", re.I)),
    ("access_denied", re.compile(r"Access denied", re.I)),
    ("file_not_found", re.compile(r"not found|No files found|does not exist", re.I)),
    ("type_overflow", re.compile(r"Out of Range|Overflow|Conversion Error", re.I)),
    ("sql_error", re.compile(r"Binder Error|Parser Error|Catalog Error|syntax error|Query execution failed", re.I)),
]
# Telemetry files are named <window>_<kind>.parquet: abnormal for the incident window, normal for the baseline.
WINDOWS = ("abnormal", "normal")
FILE_KINDS = ("traces", "logs", "metrics", "metrics_sum", "metrics_histogram")


@dataclass
class Call:
    step: Step
    args: dict[str, Any]
    turn: int
    result: Step | None = None

    @property
    def name(self) -> str:
        assert self.step.name is not None
        return self.step.name

    @property
    def sql(self) -> str:
        return str(self.args.get("query", ""))

    @property
    def files(self) -> str:
        files = self.args.get("parquet_files", "")
        return files if isinstance(files, str) else json.dumps(files)

    @property
    def failed(self) -> bool:
        return self.result is not None and self.result.is_error

    @property
    def succeeded(self) -> bool:
        return self.result is not None and not self.result.is_error

    @property
    def tree(self) -> exp.Expression | None:
        return parse_sql(self.sql)

    @cached_property
    def telemetry(self) -> set[tuple[str, str]]:
        """(window, kind) of every telemetry file the query reads, from its SQL and its parquet_files argument."""
        given = self.args.get("parquet_files") or []
        paths = files_read(self.tree) | set([given] if isinstance(given, str) else given)
        found = set()
        for path in paths:
            window, _, kind = path.rsplit("/", 1)[-1].removesuffix(".parquet").partition("_")
            if window in WINDOWS and kind in FILE_KINDS:
                found.add((window, kind))
        return found


def calls(trajectory: Trajectory) -> list[Call]:
    """Tool calls in order, each paired with its result and numbered by assistant turn.

    Results of one turn arrive in the order of its calls, so each result pairs with the earliest unanswered call of
    the same tool. A turn is a run of assistant steps between tool results.
    """
    out: list[Call] = []
    pending: list[Call] = []
    turn = 0
    previous_kind = None
    for step in trajectory.steps:
        if step.role == "assistant" and previous_kind == "tool_result":
            turn += 1
        if step.kind == "tool_call":
            call = Call(step=step, args=json.loads(step.content), turn=turn)
            out.append(call)
            pending.append(call)
        elif step.kind == "tool_result":
            candidates = [c for c in pending if c.name == step.name]
            if not candidates:
                raise ValueError(f"{trajectory.key} step #{step.index}: result of {step.name} without a pending call")
            candidates[0].result = step
            pending.remove(candidates[0])
        previous_kind = step.kind
    return out


def error_kind(text: str) -> str:
    for kind, pattern in ERROR_KINDS:
        if pattern.search(text):
            return kind
    return "other"


def normalize_sql(sql: str) -> str:
    return re.sub(r"\s+", " ", sql.strip().rstrip(";")).lower()


@lru_cache(maxsize=256)
def parse_sql(sql: str) -> exp.Expression | None:
    """The syntax tree of an agent's query, or None when sqlglot cannot parse it.

    Agents write invalid SQL too, which DuckDB also rejects; such a query filters and reads nothing here.
    Every operator of a trajectory parses the same queries, so trees are cached; callers only read them.
    """
    try:
        return sqlglot.parse_one(sql, read="duckdb")
    except sqlglot.errors.SqlglotError:
        return None


def service_key(name: str) -> str:
    return name.strip().strip("%").lower().replace("_", "-")


def services_in(tree: exp.Expression | None) -> set[str]:
    """Service names compared with the service_name column through =, IN, LIKE or ILIKE, as service keys."""
    if tree is None:
        return set()
    services = set()
    for node in tree.find_all(exp.EQ, exp.In, exp.Like, exp.ILike):
        sides = [node.this, *([] if isinstance(node, exp.In) else [node.expression])]
        if not any(isinstance(s, exp.Column) and s.name.lower() == "service_name" for s in sides):
            continue
        values = node.expressions if isinstance(node, exp.In) else sides
        services.update(service_key(v.this) for v in values if isinstance(v, exp.Literal) and v.is_string)
    return {s for s in services if s}


def files_read(tree: exp.Expression | None) -> set[str]:
    """Names and paths of the files a query reads: its table names and the string literals inside its tables."""
    if tree is None:
        return set()
    names = set()
    for table in tree.find_all(exp.Table):
        names.add(table.name)
        names.update(lit.this for lit in table.find_all(exp.Literal) if lit.is_string)
    return {n for n in names if n}


def submission(trajectory: Trajectory) -> dict[str, Any] | None:
    """Arguments of the last submit_findings call, which the harness accepted as the answer."""
    submitted = [c for c in calls(trajectory) if c.name == "submit_findings"]
    return submitted[-1].args if submitted else None


def has_submission(trajectory: Trajectory) -> bool:
    return submission(trajectory) is not None


def root_causes(answer: dict[str, Any]) -> list[dict[str, Any]]:
    """Well-formed root causes, which are objects naming a service.

    A few submissions in ops-lite hold root causes without a service or as bare strings.
    """
    return [rc for rc in answer.get("root_causes", []) if isinstance(rc, dict) and "service" in rc]


def submitted_services(answer: dict[str, Any]) -> set[str]:
    return {service_key(str(rc["service"])) for rc in root_causes(answer)}


def true_services(trajectory: Trajectory) -> list[str]:
    return [service_key(s) for s in trajectory.metadata["rc_services"]]
