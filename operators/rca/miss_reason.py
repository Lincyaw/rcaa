from __future__ import annotations

from traj_analyzer.operators.base import FeatureSpec, NoParams, Operator
from traj_analyzer.schema import Trajectory

from rca._parse import submission, submitted_services

REASONS = {
    "never_examined": "No query or reasoning ever looks at the missed service.",
    "filtered_out": "Its signal was removed by how the agent queried: a log level or status literal that excludes "
                    "it, a default row limit, or a failed query that was not retried.",
    "dismissed_as_victim": "The agent looked at it and decided it was only affected by another service's fault.",
    "masked": "Its symptoms were hidden because an upstream fault stopped traffic to it, and the agent read the "
              "silence as health.",
    "wrong_end_of_edge": "The agent blamed the other end of a call edge the missed service is on.",
    "stopped_after_first": "The agent found another root cause and stopped searching.",
    "misread_evidence": "A result pointing at the missed service was misread or its numbers were misinterpreted.",
}


def missed_ground_truth(trajectory: Trajectory) -> bool:
    answer = submission(trajectory)
    return answer is not None and bool(set(trajectory.metadata["rc_services"]) - submitted_services(answer))


def outputs(params: NoParams) -> list[FeatureSpec]:
    return [FeatureSpec(
        name="gt_miss_reasons", type="set", labels=REASONS,
        description="For each service in the ground truth rc_services that is not among the submitted root causes, "
                    "why the agent missed it; give every reason that applies to any missed service. Empty when "
                    "every ground-truth service was submitted.",
    )]


def guidance(params: NoParams) -> str:
    return ("This operator uses the ground truth: compare rc_services in the metadata with the services of the "
            "submitted root causes, then trace how the agent handled each missed service.")


OPERATOR = Operator(
    kind="llm",
    description="Why the agent missed ground-truth root-cause services.",
    tags=("agent", "rca"),
    outputs=outputs,
    guidance=guidance,
    requires=missed_ground_truth,
)
