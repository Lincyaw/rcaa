from __future__ import annotations

from typing import Any

from traj_analyzer.operators.base import FeatureSpec, NoParams, Operator
from traj_analyzer.schema import Trajectory

from rca._parse import QUERY, calls, services_in, true_services


def outputs(params: NoParams) -> list[FeatureSpec]:
    return [
        FeatureSpec(name="gt_focused_query", type="boolean",
                    description="Some query filters service_name on ground-truth root-cause services only."),
        FeatureSpec(name="gt_probe_share", type="scalar", range=(0, 1),
                    description="Share of service-filtered queries whose filter includes a ground-truth root-cause "
                                "service; empty when no query filters on service_name."),
        FeatureSpec(name="gt_metric_probed", type="boolean",
                    description="Some successful query filters on a ground-truth root-cause service and reads a "
                                "metrics file."),
        FeatureSpec(name="gt_baseline_compared", type="boolean",
                    description="Some ground-truth root-cause service is filtered on in a successful query of an "
                                "abnormal file and in a successful query of a normal file."),
    ]


def compute(trajectory: Trajectory, params: NoParams) -> dict[str, Any]:
    truth = set(true_services(trajectory))
    filtered = 0
    on_truth = 0
    focused = False
    metric = False
    windows: dict[str, set[str]] = {service: set() for service in truth}
    for call in calls(trajectory):
        if call.name != QUERY:
            continue
        services = services_in(call.tree)
        if not services:
            continue
        filtered += 1
        hit = services & truth
        on_truth += bool(hit)
        focused = focused or services <= truth
        if not hit or not call.succeeded:
            continue
        metric = metric or any(kind.startswith("metrics") for _, kind in call.telemetry)
        for service in hit:
            windows[service].update(window for window, _ in call.telemetry)
    return {
        "gt_focused_query": focused,
        "gt_probe_share": on_truth / filtered if filtered else None,
        "gt_metric_probed": metric,
        "gt_baseline_compared": any(len(w) == 2 for w in windows.values()),
    }


OPERATOR = Operator(
    kind="code",
    description="How the queries covered the ground-truth root-cause services: focus, share, metrics and baseline.",
    tags=("agent", "rca"),
    outputs=outputs,
    compute=compute,
)
