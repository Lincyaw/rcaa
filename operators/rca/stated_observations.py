from __future__ import annotations

from traj_analyzer.operators.base import FeatureSpec, NoParams, Operator

GUIDANCE = """\
Each feature is true only when the agent itself writes such a statement in an assistant message or a think_tool call;
cite that step. A tool result showing the same thing does not count. Do not judge whether the statement is correct.
"""


def outputs(params: NoParams) -> list[FeatureSpec]:
    return [
        FeatureSpec(name="agent_mentions_multiple_faults", type="boolean",
                    description="The agent writes that there may be, or are, more than one independent fault."),
        FeatureSpec(name="agent_notes_missing_data", type="boolean",
                    description="The agent writes that expected data is missing: a service's spans, traces, metrics "
                                "or pods absent in the incident window."),
        FeatureSpec(name="agent_notes_contradiction", type="boolean",
                    description="The agent writes that a result does not fit, contradicts, or is not explained by "
                                "its current hypothesis."),
        FeatureSpec(name="agent_mentions_fault_injection", type="boolean",
                    description="The agent writes about fault injection tooling, such as chaos experiments or a "
                                "mutation agent."),
        FeatureSpec(name="agent_explains_fault_kind", type="boolean",
                    description="Before submitting, the agent writes why it chose the fault kind it submits, e.g. "
                                "'restarts and not-ready pods mean pod failure'."),
    ]


OPERATOR = Operator(
    kind="llm",
    description="Whether the agent itself writes certain kinds of statement during the investigation.",
    tags=("agent", "rca"),
    outputs=outputs,
    guidance=lambda params: GUIDANCE,
)
