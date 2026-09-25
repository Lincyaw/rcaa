from __future__ import annotations

from traj_analyzer.operators.base import FeatureSpec, NoParams, Operator

COUNTER = {
    "none_observed": "No result the agent saw contradicts its final answer.",
    "addressed": "A contradicting result appeared and the agent explained it or changed its answer.",
    "ignored": "A contradicting result appeared and the agent submitted without addressing it.",
}
GROUNDING = {
    "grounded": "Every number and quoted string in the submitted claims appears in an earlier tool result.",
    "numbers_from_other_queries": "Claims cite real values, but from results of queries other than the evidence SQL "
                                  "attached to them.",
    "contains_fabricated": "At least one claim states a value or message that no earlier tool result shows and "
                           "that cannot come from the cut-off part of a result.",
    "too_vague_to_check": "The claims are qualitative and give nothing to check.",
}


def outputs(params: NoParams) -> list[FeatureSpec]:
    return [
        FeatureSpec(name="counter_evidence", type="category", labels=COUNTER,
                    description="Whether results contradicting the submitted root causes or propagation appeared, "
                                "and what the agent did about them. A contradiction is, for example, the suspect "
                                "service being healthy against its baseline, or a propagation edge whose callee "
                                "returned quickly."),
        FeatureSpec(name="absence_signal_used", type="boolean",
                    description="The agent explicitly used missing data as evidence: spans of a service that vanish "
                                "in the incident window, a client span with no server child, or a pod that stops "
                                "reporting."),
        FeatureSpec(name="unsupported_reasoning_claims", type="scalar",
                    description="Number of factual statements in the agent's reasoning or final claims that no "
                                "earlier tool result supports, counting each distinct statement once."),
        FeatureSpec(name="evidence_grounding", type="category", labels=GROUNDING,
                    description="How the submitted evidence claims relate to the tool results the agent saw."),
    ]


def guidance(params: NoParams) -> str:
    return ("For evidence_grounding and unsupported_reasoning_claims, compare each claim with the tool results that "
            "precede it. Cite the claim's step and the result it should have come from.")


OPERATOR = Operator(
    kind="llm",
    description="Whether the agent's claims rest on what it observed, including absences and contradictions.",
    tags=("agent", "rca"),
    outputs=outputs,
    guidance=guidance,
)
