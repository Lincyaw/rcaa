from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from traj_analyzer.schema import Step, Trajectory

TOOLS = {"query_parquet_files", "get_schema", "list_tables_in_directory", "think_tool", "submit_findings"}
QUERY = "query_parquet_files"
# Conventions of adapters/rcabench_eval.py, which marks stored results that were cut and names two special steps.
STORED_CUT_MARKER = "\n[cut when stored]"
FORCE_SUBMIT = "force_submit"

# Error texts of the RCABench harness, matched in order; the first match names the kind.
ERROR_KINDS = [
    ("token_budget", re.compile(r"exceeds token budget", re.I)),
    ("access_denied", re.compile(r"Access denied", re.I)),
    ("file_not_found", re.compile(r"not found|No files found|does not exist", re.I)),
    ("type_overflow", re.compile(r"Out of Range|Overflow|Conversion Error", re.I)),
    ("sql_error", re.compile(r"Binder Error|Parser Error|Catalog Error|syntax error|Query execution failed", re.I)),
]
BASELINE = re.compile(r"(?<![a-z_])normal_(traces|logs|metrics)", re.I)
ABNORMAL = re.compile(r"abnormal_(traces|logs|metrics)", re.I)
MODALITY = re.compile(r"(?:ab)?normal_(traces|logs|metrics_histogram|metrics_sum|metrics)", re.I)
SERVICE_FILTER = re.compile(r"service_name\"?\s*(?:=|IN\s*\(|LIKE|ILIKE)\s*([^)]*?)(?:\)|\bAND\b|\bOR\b|\bGROUP\b|\bORDER\b|$)",
                            re.I | re.S)
QUOTED = re.compile(r"'([^']+)'")


@dataclass
class Call:
    step: Step
    args: dict[str, Any]
    turn: int
    result: Step | None = None

    @property
    def name(self) -> str:
        assert self.step.name is not None
        return self.step.name

    @property
    def sql(self) -> str:
        return str(self.args.get("query", ""))

    @property
    def files(self) -> str:
        files = self.args.get("parquet_files", "")
        return files if isinstance(files, str) else json.dumps(files)

    @property
    def failed(self) -> bool:
        return self.result is not None and self.result.is_error


def calls(trajectory: Trajectory) -> list[Call]:
    """Tool calls in order, each paired with its result and numbered by assistant turn.

    Results of one turn arrive in the order of its calls, so each result pairs with the earliest unanswered call of
    the same tool. A turn is a run of assistant steps between tool results.
    """
    out: list[Call] = []
    pending: list[Call] = []
    turn = 0
    previous_kind = None
    for step in trajectory.steps:
        if step.role == "assistant" and previous_kind == "tool_result":
            turn += 1
        if step.kind == "tool_call":
            call = Call(step=step, args=json.loads(step.content), turn=turn)
            out.append(call)
            pending.append(call)
        elif step.kind == "tool_result":
            candidates = [c for c in pending if c.name == step.name]
            if not candidates:
                raise ValueError(f"{trajectory.key} step #{step.index}: result of {step.name} without a pending call")
            candidates[0].result = step
            pending.remove(candidates[0])
        previous_kind = step.kind
    return out


def error_kind(text: str) -> str:
    for kind, pattern in ERROR_KINDS:
        if pattern.search(text):
            return kind
    return "other"


def normalize_sql(sql: str) -> str:
    return re.sub(r"\s+", " ", sql.strip().rstrip(";")).lower()


def services_in(sql: str) -> set[str]:
    services = set()
    for match in SERVICE_FILTER.finditer(sql):
        services.update(value.strip("%") for value in QUOTED.findall(match.group(1)))
    return {s for s in services if s}


def submission(trajectory: Trajectory) -> dict[str, Any] | None:
    """Arguments of the last submit_findings call, which the harness accepted as the answer."""
    submitted = [c for c in calls(trajectory) if c.name == "submit_findings"]
    return submitted[-1].args if submitted else None


def has_submission(trajectory: Trajectory) -> bool:
    return submission(trajectory) is not None


def root_causes(answer: dict[str, Any]) -> tuple[list[dict[str, Any]], int]:
    """Well-formed root causes, which are objects naming a service, and the number of malformed entries.

    A few submissions in ops-lite hold root causes without a service or as bare strings.
    """
    entries = answer.get("root_causes", [])
    valid = [rc for rc in entries if isinstance(rc, dict) and "service" in rc]
    return valid, len(entries) - len(valid)


def submitted_services(answer: dict[str, Any]) -> set[str]:
    return {str(rc["service"]) for rc in root_causes(answer)[0]}
