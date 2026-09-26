from __future__ import annotations

from typing import Any

from sqlglot import exp
from traj_analyzer.operators.base import FeatureSpec, NoParams, Operator
from traj_analyzer.schema import Trajectory

from rca._parse import QUERY, calls, services_in

MODALITIES = {
    "traces": "abnormal_traces or normal_traces",
    "logs": "abnormal_logs or normal_logs",
    "metrics": "abnormal_metrics, abnormal_metrics_sum, abnormal_metrics_histogram or their normal counterparts",
}
FILE_MODALITY = {"traces": "traces", "logs": "logs", "metrics": "metrics", "metrics_sum": "metrics",
                 "metrics_histogram": "metrics"}
INJECTION_TERMS = ("mutat", "inject", "chaos", "fault")
PATTERN_MATCHES = (exp.Like, exp.ILike, exp.RegexpLike, exp.RegexpILike)


def outputs(params: NoParams) -> list[FeatureSpec]:
    return [
        FeatureSpec(name="baseline_query_ratio", type="scalar", range=(0, 1), thresholds={"low": 0.1},
                    description="Share of queries that read a normal-period parquet."),
        FeatureSpec(name="first_baseline_position", type="scalar", range=(0, 1),
                    description="Position of the first normal-period query as a fraction of all steps; empty when "
                                "the agent never read the baseline."),
        FeatureSpec(name="modality_query_share", type="map", labels=MODALITIES, range=(0, 1),
                    description="Share of queries that read each telemetry modality; a query reading several "
                                "modalities counts equally toward each, so the shares sum to 1."),
        FeatureSpec(name="first_modality", type="category", labels=MODALITIES,
                    description="Telemetry modality of the first successful query."),
        FeatureSpec(name="span_parent_join", type="boolean",
                    description="Some query joins on parent_span_id, to pair client and server spans or to map "
                                "callers."),
        FeatureSpec(name="span_gap_query", type="boolean",
                    description="Some query that relates spans through parent_span_id subtracts a time or duration "
                                "column of one span from one of another, locating latency between caller and callee."),
        FeatureSpec(name="injection_log_hunt", type="boolean",
                    description="Some query matches a message or body column against fault injection terms (mutat, "
                                "inject, chaos, fault)."),
        FeatureSpec(name="severe_level_exclusion", type="boolean",
                    description="TrainTicket only: the agent filtered logs by level but never queried SEVERE, the "
                                "level its Java services use for their worst errors; empty for other systems."),
        FeatureSpec(name="probed_service_count", type="scalar",
                    description="Distinct service names the agent filtered on in its queries."),
    ]


def _columns(node: exp.Expression) -> list[exp.Column]:
    return list(node.find_all(exp.Column))


def _named(node: exp.Expression, *names: str) -> bool:
    return any(c.name.lower() in names for c in _columns(node))


def _timelike(node: exp.Expression) -> set[tuple[str, str]]:
    return {(c.table, c.name.lower()) for c in _columns(node)
            if "time" in c.name.lower() or "dur" in c.name.lower()}


def parent_join(tree: exp.Expression) -> bool:
    return any(_named(j, "parent_span_id") for j in tree.find_all(exp.Join)) or any(
        _named(eq, "parent_span_id") for w in tree.find_all(exp.Where) for eq in w.find_all(exp.EQ))


def span_gap(tree: exp.Expression) -> bool:
    if not any(_named(eq, "parent_span_id") for eq in tree.find_all(exp.EQ)):
        return False
    for sub in tree.find_all(exp.Sub):
        left, right = _timelike(sub.left), _timelike(sub.right)
        if left and right and left != right:
            return True
    return any(len(_timelike(diff)) > 1 for diff in tree.find_all(exp.DateDiff))


def injection_hunt(tree: exp.Expression) -> bool:
    for match in tree.find_all(*PATTERN_MATCHES):
        if not _named(match.this, "message", "body"):
            continue
        texts = [lit.this.lower() for lit in match.find_all(exp.Literal) if lit.is_string]
        if any(term in text for text in texts for term in INJECTION_TERMS):
            return True
    return False


def level_filter(tree: exp.Expression) -> bool:
    return any(_named(p.this, "level") for p in tree.find_all(exp.EQ, exp.In, exp.Like, exp.ILike))


def compute(trajectory: Trajectory, params: NoParams) -> dict[str, Any]:
    queries = [c for c in calls(trajectory) if c.name == QUERY]
    baseline = [c for c in queries if any(window == "normal" for window, _ in c.telemetry)]
    weights = dict.fromkeys(MODALITIES, 0.0)
    read = 0
    for call in queries:
        found = {FILE_MODALITY[kind] for _, kind in call.telemetry}
        read += bool(found)
        for modality in found:
            weights[modality] += 1 / len(found)
    first = next((FILE_MODALITY[min(c.telemetry)[1]] for c in queries if c.succeeded and c.telemetry), None)
    trees = [c.tree for c in queries if c.tree is not None]
    literals = [lit.this.upper() for tree in trees for lit in tree.find_all(exp.Literal) if lit.is_string]
    return {
        "baseline_query_ratio": len(baseline) / len(queries) if queries else None,
        "first_baseline_position": baseline[0].step.index / len(trajectory.steps) if baseline else None,
        "modality_query_share": {m: w / read for m, w in weights.items()} if read else None,
        "first_modality": first,
        "span_parent_join": any(parent_join(t) for t in trees),
        "span_gap_query": any(span_gap(t) for t in trees),
        "injection_log_hunt": any(injection_hunt(t) for t in trees),
        "severe_level_exclusion": any(level_filter(t) for t in trees) and not any("SEVERE" in s for s in literals)
        if trajectory.metadata["system"] == "ts" else None,
        "probed_service_count": float(len(set().union(*(services_in(c.tree) for c in queries)))),
    }


OPERATOR = Operator(
    kind="code",
    description="Which telemetry the agent looked at and how: baseline use, modalities, span pairing and gaps, log "
                "hunting.",
    tags=("agent", "rca"),
    outputs=outputs,
    compute=compute,
)
