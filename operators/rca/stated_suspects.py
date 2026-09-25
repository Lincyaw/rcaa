from __future__ import annotations

from traj_analyzer.operators.base import FeatureSpec, NoParams, Operator

GUIDANCE = """\
Copy service names exactly as the agent writes them. Read the agent's own assistant messages and think_tool calls;
do not count services that only appear in tool results or in SQL. Do not judge whether the agent was right.
"""


def outputs(params: NoParams) -> list[FeatureSpec]:
    return [
        FeatureSpec(name="suspected_services", type="set",
                    description="Services the agent writes may be, or are, a root cause, e.g. 'likely the root cause', "
                                "'suspect', 'the fault is in'."),
        FeatureSpec(name="ruled_out_services", type="set",
                    description="Services the agent writes are not a root cause: healthy, unaffected, only a victim, "
                                "or only propagating another service's fault."),
        FeatureSpec(name="first_suspect", type="category",
                    description="The first service the agent writes may be a root cause; 'none' if it never does."),
        FeatureSpec(name="first_suspect_step", type="scalar",
                    description="Step number of the message where first_suspect is first written; -1 if none."),
        FeatureSpec(name="suspected_gt_services", type="set",
                    description="The services of rc_services in the metadata that appear in suspected_services."),
        FeatureSpec(name="ruled_out_gt_services", type="set",
                    description="The services of rc_services in the metadata that appear in ruled_out_services."),
    ]


OPERATOR = Operator(
    kind="llm",
    description="Which services the agent itself writes it suspects or rules out.",
    tags=("agent", "rca"),
    outputs=outputs,
    guidance=lambda params: GUIDANCE,
)
