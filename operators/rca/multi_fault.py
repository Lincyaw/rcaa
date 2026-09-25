from __future__ import annotations

from traj_analyzer.operators.base import FeatureSpec, NoParams, Operator

SEARCH = {
    "none": "After finding a first root cause the agent never asks whether another independent fault exists.",
    "superficial": "It asks, but checks only for errors, error logs or restarts elsewhere, which misses faults that "
                   "show as latency, missing traffic or subtle changes.",
    "systematic": "It compares other services against the normal baseline, or checks which symptoms the first root "
                  "cause does not explain, before concluding.",
}


def outputs(params: NoParams) -> list[FeatureSpec]:
    return [
        FeatureSpec(name="multi_fault_search", type="category", labels=SEARCH,
                    description="How the agent looked for a second, independent fault after its first root cause. "
                                "Judge the search itself, whether or not the incident really had several faults."),
        FeatureSpec(name="unexplained_symptoms_left", type="boolean",
                    description="The agent itself observed an anomaly, such as a slow or failing service, that its "
                                "submitted root causes and propagation do not explain, and submitted anyway."),
    ]


OPERATOR = Operator(
    kind="llm",
    description="Whether the agent looked for more than one fault before concluding.",
    tags=("agent", "rca"),
    outputs=outputs,
)
