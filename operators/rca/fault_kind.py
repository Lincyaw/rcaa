from __future__ import annotations

from traj_analyzer.operators.base import FeatureSpec, NoParams, Operator

from rca._parse import has_submission

BASIS = {
    "discriminating_signature": "A signal that separates this fault kind from others, e.g. restarts and not-ready "
                                "pods for pod failure, a client-server gap with a healthy callee for network delay, "
                                "retransmission-like delay steps for packet loss.",
    "salient_metric": "The most striking metric, even though it may be an effect of another fault, e.g. CPU after "
                      "a restart taken as CPU stress.",
    "log_wording": "Words in an exception or log message, e.g. 'JDBC' leading to a JDBC fault kind.",
    "prior_guess": "A default or a typical choice the agent states without evidence ('usually', 'typical').",
    "contradicted": "The agent's own results contradict the kind it submitted, e.g. pod_failure while the pod stayed "
                    "ready with no restarts.",
}


def outputs(params: NoParams) -> list[FeatureSpec]:
    return [FeatureSpec(
        name="fault_kind_basis", type="category", labels=BASIS,
        description="What the fault_kind of the first submitted root cause rests on, judged from the agent's "
                    "reasoning and results before submission.",
    )]


OPERATOR = Operator(
    kind="llm",
    description="Why the agent chose the fault kind it submitted.",
    tags=("agent", "rca"),
    outputs=outputs,
    requires=has_submission,
)
