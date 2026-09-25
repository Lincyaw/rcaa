from __future__ import annotations

from traj_analyzer.operators.base import FeatureSpec, NoParams, Operator

from rca._parse import has_submission

SIGNALS = {
    "error_count": "Counts or rates of errors, failed spans or error logs.",
    "latency": "Durations, latency percentiles or slow spans.",
    "pod_state": "Kubernetes or container state: ready, restarts, available replicas.",
    "resource_usage": "CPU, memory, network or disk usage.",
    "log_message": "The text of specific log messages or exceptions.",
    "missing_data": "Spans, traces, metrics or pods that are absent.",
    "call_relation": "Which service calls which, or parent and child spans.",
    "injection_log": "Logs of fault injection tooling, such as a mutation or chaos agent.",
}


def outputs(params: NoParams) -> list[FeatureSpec]:
    return [
        FeatureSpec(name="evidence_signals", type="set", labels=SIGNALS,
                    description="Kinds of signal that the claims of the submitted evidence describe. Look only at the "
                                "step named final_answer and give every kind that at least one claim describes."),
        FeatureSpec(name="evidence_compares_baseline", type="boolean",
                    description="At least one submitted claim states a value for both the incident window and the "
                                "normal window, such as 'p95 went from 8 ms to 1714 ms'. Look only at the step named "
                                "final_answer."),
    ]


OPERATOR = Operator(
    kind="llm",
    description="What the submitted evidence claims talk about, read from the final answer alone.",
    tags=("agent", "rca"),
    outputs=outputs,
    requires=has_submission,
)
