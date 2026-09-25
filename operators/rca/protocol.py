from __future__ import annotations

from collections import defaultdict
from typing import Any

from traj_analyzer.operators.base import FeatureSpec, NoParams, Operator
from traj_analyzer.schema import Trajectory

from rca._parse import FORCE_SUBMIT, QUERY, calls


def outputs(params: NoParams) -> list[FeatureSpec]:
    return [
        FeatureSpec(name="think_per_query", type="scalar",
                    description="think_tool calls per query_parquet_files call; the prompt asks for one per query."),
        FeatureSpec(name="think_distinct_ratio", type="scalar", range=(0, 1), thresholds={"low": 0.5},
                    description="Distinct think_tool texts divided by think_tool calls; low values mean the agent "
                                "repeated the same reasoning."),
        FeatureSpec(name="think_only_turn_share", type="scalar", range=(0, 1),
                    description="Share of turns whose calls are all think_tool: planning without acting."),
        FeatureSpec(name="force_submitted", type="boolean",
                    description="The harness injected a force_submit message because the budget ran out."),
        FeatureSpec(name="submit_with_other_calls", type="boolean",
                    description="submit_findings was issued in the same turn as other tool calls, which the prompt "
                                "forbids."),
    ]


def compute(trajectory: Trajectory, params: NoParams) -> dict[str, Any]:
    all_calls = calls(trajectory)
    thinks = [c for c in all_calls if c.name == "think_tool"]
    queries = [c for c in all_calls if c.name == QUERY]
    turns: dict[int, list[str]] = defaultdict(list)
    for call in all_calls:
        turns[call.turn].append(call.name)
    submit_turns = [names for names in turns.values() if "submit_findings" in names]
    return {
        "think_per_query": len(thinks) / len(queries) if queries else None,
        "think_distinct_ratio": len({c.step.content for c in thinks}) / len(thinks) if thinks else None,
        "think_only_turn_share": sum(1 for names in turns.values() if set(names) == {"think_tool"}) / len(turns)
        if turns else None,
        "force_submitted": any(s.name == FORCE_SUBMIT for s in trajectory.steps),
        "submit_with_other_calls": any(len(names) > 1 for names in submit_turns),
    }


OPERATOR = Operator(
    kind="code",
    description="Whether the agent followed the harness protocol: explicit reasoning, budget and submission rules.",
    tags=("agent", "rca"),
    outputs=outputs,
    compute=compute,
)
