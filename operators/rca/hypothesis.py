from __future__ import annotations

from traj_analyzer.operators.base import FeatureSpec, NoParams, Operator

ORIGIN = {
    "error_volume": "The service with the most errors or error logs, without checking the normal baseline.",
    "baseline_delta": "A change against the normal window: error rate, latency or volume that differs from normal.",
    "latency_outlier": "One extreme latency value or a single slow trace.",
    "missing_telemetry": "Data that disappeared: a service's spans, a callee span or metrics missing in the incident.",
    "infra_state": "Kubernetes or container state: pods not ready, restarts, deployments unavailable.",
    "log_content": "The wording of specific log messages, such as exception names.",
    "call_structure": "The call graph or the gap between a client span and its server span.",
    "injection_trace": "Logs of the fault injection tooling itself, such as a mutation agent.",
}
ANCHORING = {
    "none": "The agent kept candidates open until evidence separated them.",
    "symptom_anchor": "It took the loudest symptom, the service with most errors or latency, as the root cause.",
    "outlier_anchor": "It fixed on one extreme data point, such as a single slow span, over the distribution.",
    "single_chain_anchor": "It followed one causal chain and stopped once that chain explained part of the symptoms.",
    "early_commit": "It committed within the first few queries and only gathered confirming evidence afterwards.",
}
ALTERNATIVES = {
    "none": "No other candidate service or fault kind is mentioned.",
    "named_only": "Other candidates are mentioned but not checked with a query.",
    "tested": "At least one other candidate is checked with a query and rejected by citing its result.",
}


def outputs(params: NoParams) -> list[FeatureSpec]:
    return [
        FeatureSpec(name="hypothesis_origin", type="category", labels=ORIGIN,
                    description="The kind of signal that first made the agent suspect the service it finally "
                                "submitted as its first root cause."),
        FeatureSpec(name="anchoring", type="category", labels=ANCHORING,
                    description="How the agent settled on its answer."),
        FeatureSpec(name="alternatives", type="category", labels=ALTERNATIVES,
                    description="How the agent treated candidates other than the ones it submitted."),
        FeatureSpec(name="commit_position", type="scalar", range=(0, 1),
                    description="Step number where the agent first commits to its final root-cause service, divided "
                                "by the number of the last step. Commitment means it stops looking at other services "
                                "and only gathers evidence for this one."),
    ]


OPERATOR = Operator(
    kind="llm",
    description="How the agent formed and settled its root-cause hypothesis.",
    tags=("agent", "rca"),
    outputs=outputs,
    guidance=lambda params: "Pick the label that describes the step where the final root-cause service first became "
                            "the agent's leading suspect; cite that step in the evidence.",
)
