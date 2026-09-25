from __future__ import annotations

from typing import Any

from traj_analyzer.operators.base import FeatureSpec, NoParams, Operator
from traj_analyzer.schema import Trajectory

from rca._parse import (
    QUERY,
    calls,
    has_submission,
    normalize_sql,
    root_causes,
    services_in,
    submission,
    submitted_services,
)


def outputs(params: NoParams) -> list[FeatureSpec]:
    return [
        FeatureSpec(name="n_root_causes", type="scalar",
                    description="Well-formed root causes, objects naming a service, in the submitted answer."),
        FeatureSpec(name="malformed_root_causes", type="scalar",
                    description="Submitted root-cause entries without a service or not given as objects."),
        FeatureSpec(name="root_cause_gap", type="scalar",
                    description="Ground-truth root-cause services minus submitted root causes; positive means the "
                                "agent submitted fewer causes than the incident has."),
        FeatureSpec(name="n_evidence", type="scalar", description="Evidence items over all submitted root causes."),
        FeatureSpec(name="evidence_kinds", type="set", description="Kinds of the submitted evidence items."),
        FeatureSpec(name="submitted_fault_kinds", type="set", description="Fault kinds of the submitted root causes."),
        FeatureSpec(name="evidence_preexecuted_share", type="scalar", range=(0, 1),
                    description="Share of evidence SQL that the agent had already run successfully, compared after "
                                "normalizing whitespace and case."),
        FeatureSpec(name="gt_services_probed_share", type="scalar", range=(0, 1),
                    description="Share of ground-truth root-cause services the agent ever filtered on in a query."),
        FeatureSpec(name="gt_probed_not_submitted", type="boolean",
                    description="The agent filtered on some ground-truth service but did not submit it as a root "
                                "cause: it looked at the right service and dropped it."),
    ]


def compute(trajectory: Trajectory, params: NoParams) -> dict[str, Any]:
    answer = submission(trajectory)
    assert answer is not None
    valid, malformed = root_causes(answer)
    evidence = [item for rc in valid for item in rc.get("evidence", []) if isinstance(item, dict)]
    queries = [c for c in calls(trajectory) if c.name == QUERY]
    executed = {normalize_sql(c.sql) for c in queries if c.result is not None and not c.failed}
    evidence_sql = [normalize_sql(str(item.get("sql", ""))) for item in evidence if item.get("sql")]
    probed = set().union(*(services_in(c.sql) for c in queries))
    truth = set(trajectory.metadata["rc_services"])
    submitted = submitted_services(answer)
    return {
        "n_root_causes": float(len(valid)),
        "malformed_root_causes": float(malformed),
        "root_cause_gap": float(len(truth) - len(valid)),
        "n_evidence": float(len(evidence)),
        "evidence_kinds": sorted({str(item.get("kind")) for item in evidence}),
        "submitted_fault_kinds": sorted({str(rc.get("fault_kind")) for rc in valid}),
        "evidence_preexecuted_share": sum(s in executed for s in evidence_sql) / len(evidence_sql)
        if evidence_sql else None,
        "gt_services_probed_share": len(truth & probed) / len(truth),
        "gt_probed_not_submitted": bool((truth & probed) - submitted),
    }


OPERATOR = Operator(
    kind="code",
    description="What the agent submitted, how its evidence relates to the queries it ran, and how it covered the "
                "ground-truth services.",
    tags=("agent", "rca"),
    outputs=outputs,
    compute=compute,
    requires=has_submission,
)
