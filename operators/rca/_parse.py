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
        if isinstance(files, str):
            return files
        if isinstance(files, list) and all(isinstance(path, str) for path in files):
            # File order does not affect a query. Canonicalizing it lets recovery analysis match retries that only
            # reorder the same inputs.
            return json.dumps(sorted(files), ensure_ascii=False)
        return json.dumps(files, ensure_ascii=False, sort_keys=True)

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
        supplied = [given] if isinstance(given, str) else given if isinstance(given, list) else []
        paths = files_read(self.tree) | {path for path in supplied if isinstance(path, str)}
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
    result_since_assistant = False
    for step in trajectory.steps:
        if step.role == "assistant" and result_since_assistant:
            turn += 1
            result_since_assistant = False
        if step.kind == "tool_call":
            args = json.loads(step.content)
            if not isinstance(args, dict):
                raise ValueError(f"{trajectory.key} step #{step.index}: tool arguments must be a JSON object")
            call = Call(step=step, args=args, turn=turn)
            out.append(call)
            pending.append(call)
        elif step.kind == "tool_result":
            candidates = [c for c in pending if c.name == step.name]
            if not candidates:
                raise ValueError(f"{trajectory.key} step #{step.index}: result of {step.name} without a pending call")
            candidates[0].result = step
            pending.remove(candidates[0])
            result_since_assistant = True
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


# Aggregate functions by the name a query atom uses for them; percentile and deviation variants share one name.
AGGREGATES: dict[type[exp.Expression], str] = {
    exp.Count: "count", exp.Avg: "avg", exp.Sum: "sum", exp.Min: "min", exp.Max: "max",
    exp.Quantile: "quantile", exp.ApproxQuantile: "quantile", exp.PercentileCont: "quantile",
    exp.PercentileDisc: "quantile", exp.Median: "quantile",
    exp.Stddev: "stddev", exp.StddevPop: "stddev", exp.StddevSamp: "stddev", exp.Variance: "stddev",
}
# Columns whose compared values name one service, operation, trace, span, pod or moment of a system, not a kind of
# signal.
IDENTIFIER_COLUMNS = {"service_name", "span_name", "trace_id", "span_id", "parent_span_id", "attr.k8s.pod.name", "time"}


def _aliases(tree: exp.Expression) -> set[str]:
    """Names a query defines itself, as aliases or CTEs, which are not data columns.

    An alias that only repeats a column's own name, as in a.service_name AS service_name, still names that column.
    """
    aliases = {a.alias.lower() for a in tree.find_all(exp.Alias)
               if not (isinstance(a.this, exp.Column) and a.this.name.lower() == a.alias.lower())}
    return aliases | {c.alias_or_name.lower() for c in tree.find_all(exp.CTE)}


def _select_item(select: exp.Select, key: exp.Expression) -> exp.Expression:
    """The select-list expression a GROUP BY or ORDER BY key stands for when it is an ordinal or an alias."""
    if isinstance(key, exp.Literal) and key.is_int and 0 < int(key.this) <= len(select.expressions):
        key = select.expressions[int(key.this) - 1]
        return key.this if isinstance(key, exp.Alias) else key
    if isinstance(key, exp.Column) and not key.table:
        for item in select.expressions:
            if isinstance(item, exp.Alias) and item.alias.lower() == key.name.lower():
                return item.this
    return key


def query_atoms(tree: exp.Expression | None) -> set[str]:
    """`<role>:<column>` for every data column of every SELECT of a query.

    Roles: filter (WHERE, HAVING, or the FILTER clause of an aggregate), group (GROUP BY), order (ORDER BY), join
    (JOIN ... ON), the aggregate name for a column inside an aggregate of the select list, and show for any other
    column of the select list. SELECT * gives show:* and COUNT(*) gives count:*. GROUP BY and ORDER BY keys written as
    an ordinal or an alias count as the select-list expression they stand for. Other names the query defines itself as
    aliases or CTEs are not data columns.
    """
    if tree is None:
        return set()
    aliases = _aliases(tree)
    atoms: set[str] = set()

    def mark(scope: exp.Expression | None, role: str) -> None:
        if scope is None:
            return
        for column in scope.find_all(exp.Column):
            name = column.name.lower()
            if name and name not in aliases:
                atoms.add(f"{role}:{name}")

    for select in tree.find_all(exp.Select):
        mark(select.args.get("where"), "filter")
        mark(select.args.get("having"), "filter")
        for clause, role in (("group", "group"), ("order", "order")):
            keys = select.args.get(clause)
            for key in keys.expressions if keys else []:
                mark(_select_item(select, key.this if isinstance(key, exp.Ordered) else key), role)
        for join in select.args.get("joins") or []:
            mark(join.args.get("on"), "join")
        for expression in select.expressions:
            if isinstance(expression, exp.Star) or isinstance(expression, exp.Column) and expression.is_star:
                atoms.add("show:*")
            for filtered in expression.find_all(exp.Filter):
                mark(filtered.expression, "filter")
            aggregates = list(expression.find_all(*AGGREGATES))
            for aggregate in aggregates:
                mark(aggregate, AGGREGATES[type(aggregate)])
                if isinstance(aggregate, exp.Count) and isinstance(aggregate.this, exp.Star):
                    atoms.add("count:*")
            if not aggregates:
                mark(expression, "show")
    return atoms


COMPARISONS: dict[type[exp.Expression], str] = {exp.EQ: "=", exp.In: "=", exp.NEQ: "!=", exp.Like: "like",
                                                exp.ILike: "ilike"}


def _compared_column(side: exp.Expression) -> tuple[str, str] | None:
    """How one side of a comparison reads a data column, and that column's name.

    The side is a column, or a function of exactly one column, written as lower(level).
    """
    if isinstance(side, exp.Column):
        return side.name.lower(), side.name.lower()
    if isinstance(side, exp.Func):
        columns = list(side.find_all(exp.Column))
        if len(columns) == 1:
            name = columns[0].name.lower()
            return f"{type(side).__name__.lower()}({name})", name
    return None


def filter_values(tree: exp.Expression | None) -> set[str]:
    """`<column> <op> <value>` for string literals compared with a data column, op being =, !=, like or ilike.

    IN counts as =. Values keep the agent's exact text, case and LIKE wildcards included, because DuckDB compares them
    exactly and a value the data does not hold matches nothing without an error. A column wrapped in one function is
    named with it, as lower(level). Identifier columns such as service_name and span_name, and names the query defines
    itself, are left out.
    """
    if tree is None:
        return set()
    aliases = _aliases(tree)
    found = set()
    for node in tree.find_all(*COMPARISONS):
        sides = [node.this] if isinstance(node, exp.In) else [node.this, node.expression]
        columns = [c for c in map(_compared_column, sides) if c]
        if not columns:
            continue
        column, name = columns[0]
        if name in IDENTIFIER_COLUMNS or name in aliases:
            continue
        values = node.expressions if isinstance(node, exp.In) else sides
        for value in values:
            if isinstance(value, exp.Literal) and value.is_string:
                found.add(f"{column} {COMPARISONS[type(node)]} {value.this}")
    return found


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
    root_causes = answer.get("root_causes")
    if not isinstance(root_causes, list):
        return []
    return [rc for rc in root_causes if isinstance(rc, dict) and rc.get("service") not in (None, "")]


def submitted_services(answer: dict[str, Any]) -> set[str]:
    return {service_key(str(rc["service"])) for rc in root_causes(answer)}


def true_services(trajectory: Trajectory) -> list[str]:
    services = trajectory.metadata.get("rc_services") or []
    return [service_key(str(service)) for service in services if service not in (None, "")]
