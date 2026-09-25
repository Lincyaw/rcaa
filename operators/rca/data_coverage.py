from __future__ import annotations

import re
from typing import Any

from traj_analyzer.operators.base import FeatureSpec, NoParams, Operator
from traj_analyzer.schema import Trajectory

from rca._parse import BASELINE, MODALITY, QUERY, calls, services_in

MODALITIES = {
    "traces": "abnormal_traces or normal_traces",
    "logs": "abnormal_logs or normal_logs",
    "metrics": "abnormal_metrics or normal_metrics",
    "metrics_sum": "abnormal_metrics_sum or normal_metrics_sum",
    "metrics_histogram": "abnormal_metrics_histogram or normal_metrics_histogram",
}
PAIRING = re.compile(r"parent_span_id", re.I)
JOIN = re.compile(r"\bjoin\b", re.I)
INJECTION_HUNT = re.compile(r"(message|body)[^;]*?(like|ilike|regexp|~)[^;]*?(mutat|inject|chaos|fault)", re.I | re.S)
LEVEL_FILTER = re.compile(r"level\"?\s*(=|in\s*\(|ilike|like)", re.I)


def outputs(params: NoParams) -> list[FeatureSpec]:
    return [
        FeatureSpec(name="baseline_query_ratio", type="scalar", range=(0, 1), thresholds={"low": 0.1},
                    description="Share of queries that read a normal-period parquet."),
        FeatureSpec(name="first_baseline_position", type="scalar", range=(0, 1),
                    description="Position of the first normal-period query as a fraction of all steps; empty when "
                                "the agent never read the baseline."),
        FeatureSpec(name="modalities", type="set", labels=MODALITIES,
                    description="Telemetry files the agent queried successfully."),
        FeatureSpec(name="first_modality", type="category", labels=MODALITIES,
                    description="Telemetry file of the first successful query."),
        FeatureSpec(name="span_parent_join", type="boolean",
                    description="Some query joins spans on parent_span_id, to pair client and server spans or to map "
                                "callers."),
        FeatureSpec(name="injection_log_hunt", type="boolean",
                    description="Some query searched log messages for fault injection terms (mutat, inject, "
                                "chaos, fault)."),
        FeatureSpec(name="severe_level_excluded", type="boolean",
                    description="TrainTicket only: the agent filtered logs by level but never queried SEVERE, the "
                                "level its Java services use for their worst errors; empty for other systems."),
        FeatureSpec(name="services_probed", type="scalar",
                    description="Distinct service names the agent filtered on in its queries."),
    ]


def compute(trajectory: Trajectory, params: NoParams) -> dict[str, Any]:
    queries = [c for c in calls(trajectory) if c.name == QUERY]
    succeeded = [c for c in queries if c.result is not None and not c.failed]
    baseline = [c for c in queries if BASELINE.search(c.sql + " " + c.files)]
    modalities: list[str] = []
    for call in succeeded:
        modalities += [m.lower() for m in MODALITY.findall(call.sql + " " + call.files)]
    first = None
    for call in succeeded:
        found = MODALITY.findall(call.sql + " " + call.files)
        if found:
            first = found[0].lower()
            break
    sqls = [c.sql for c in queries]
    level_filtered = any(LEVEL_FILTER.search(s) for s in sqls)
    return {
        "baseline_query_ratio": len(baseline) / len(queries) if queries else None,
        "first_baseline_position": baseline[0].step.index / len(trajectory.steps) if baseline else None,
        "modalities": sorted(set(modalities)),
        "first_modality": first,
        "span_parent_join": any(PAIRING.search(s) and JOIN.search(s) for s in sqls),
        "injection_log_hunt": any(INJECTION_HUNT.search(s) for s in sqls),
        "severe_level_excluded": level_filtered and not any("SEVERE" in s.upper() for s in sqls)
        if trajectory.metadata["system"] == "ts" else None,
        "services_probed": float(len(set().union(*(services_in(s) for s in sqls)))),
    }


OPERATOR = Operator(
    kind="code",
    description="Which telemetry the agent looked at and how: baseline use, modalities, span pairing, log hunting.",
    tags=("agent", "rca"),
    outputs=outputs,
    compute=compute,
)
