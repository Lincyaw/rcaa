from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from traj_analyzer.schema import Role, Step, Trajectory

QUERY = """
SELECT e.exp_id, e.model_name, e.agent_type, e.dataset_index, e.source, e.time_cost, e.eval_metrics,
       e.trajectories, d.meta, s.input_tokens, s.output_tokens, s.cache_hit_tokens, s.n_llm_calls
FROM evaluation_data e
JOIN data d ON d.dataset = e.dataset AND d.source = e.source
LEFT JOIN evaluation_rollout_stats s ON s.id = e.id
WHERE e.stage = ? AND e.exp_id IN ({experiments}) {model_filter}
ORDER BY e.exp_id, e.model_name, e.dataset_index
"""

MESSAGE_ROLES: dict[str, Role] = {"system": "system", "human": "user"}
ERROR_OPENING = re.compile(r'\s*\{\s*"error"\s*:')


class RcabenchEvalAdapter:
    """Rows of the RCABench evaluation database, one trajectory per (experiment, model, case).

    Each row's `trajectories` column holds the agent harness event log.
    `llm_start` messages become system and user steps; messages injected at a named node, such as `force_submit`,
    become system steps marked with the node name.
    `llm_end` becomes an assistant step for its text and one tool call step per call; the separate `tool_call`
    events repeat those calls and are skipped.
    `tool_result` becomes a tool result step, flagged as an error when the result is a JSON object with an `error` key.
    `result` becomes a final assistant step holding the submitted answer.
    Case facts, evaluation metrics and token usage go into metadata.
    """

    def __init__(self, experiments: list[str], models: list[str] | None = None, stage: str = "judged") -> None:
        self.experiments = experiments
        self.models = models
        self.stage = stage

    def read(self, path: Path, dataset: str) -> Iterator[Trajectory]:
        db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        model_filter = ""
        params: list[Any] = [self.stage, *self.experiments]
        if self.models is not None:
            model_filter = f"AND e.model_name IN ({','.join('?' * len(self.models))})"
            params += self.models
        query = QUERY.format(experiments=",".join("?" * len(self.experiments)), model_filter=model_filter)
        for row in db.execute(query, params):
            yield self._trajectory(row, dataset)
        db.close()

    def _trajectory(self, row: tuple[Any, ...], dataset: str) -> Trajectory:
        (exp_id, model, agent, index, source, time_cost, metrics_json, events_json, case_json,
         input_tokens, output_tokens, cache_hit_tokens, n_llm_calls) = row
        case = json.loads(case_json)
        metrics = json.loads(metrics_json)
        metadata: dict[str, Any] = {
            "exp_id": exp_id, "model": model, "agent": agent, "case_index": index, "case_source": source,
            "system": case["system"], "fault_type": str(case["fault_type"]), "rc_services": case["rc_services"],
            "time_cost": time_cost, "input_tokens": input_tokens, "output_tokens": output_tokens,
            "cache_hit_tokens": cache_hit_tokens, "n_llm_calls": n_llm_calls,
        }
        for key, value in metrics.items():
            if value is None or isinstance(value, bool | int | float | str):
                metadata[f"eval_{key}"] = value
        metadata["eval_fault_status"] = sorted({f["status"] for f in metrics.get("per_fault") or []})
        tid = f"{exp_id}__{model}__{index:03d}"
        steps = self._steps(json.loads(events_json)["events"], metadata)
        return Trajectory(id=tid, dataset=dataset, metadata=metadata, steps=steps)

    def _steps(self, events: list[dict[str, Any]], metadata: dict[str, Any]) -> list[Step]:
        steps: list[Step] = []

        def add(role: Role, kind: str, content: str, stamp: str, name: str | None = None,
                is_error: bool = False) -> None:
            steps.append(Step(index=len(steps), role=role, kind=kind, content=content, name=name,  # type: ignore[arg-type]
                              is_error=is_error, timestamp=stamp))

        for event in events:
            if "_meta" in event:
                continue
            kind, data, stamp = event["event_type"], event["data"], event["timestamp"]
            match kind:
                case "run_start":
                    metadata["prompt_path"] = data["prompt_path"]
                case "llm_start":
                    node = data.get("node")
                    for message in data["messages"]:
                        if node is not None:
                            add("system", "message", f"[{node}] {message['content']}", stamp)
                        else:
                            add(MESSAGE_ROLES[message["type"]], "message", message["content"], stamp)
                case "llm_end":
                    if data["content"]:
                        add("assistant", "message", data["content"], stamp)
                    for call in data.get("tool_calls") or []:
                        add("assistant", "tool_call", json.dumps(call["args"], ensure_ascii=False), stamp,
                            name=call["name"])
                case "tool_call":
                    continue
                case "tool_result":
                    result = data["result"]
                    text = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
                    add("tool", "tool_result", text, stamp, name=data["tool_name"], is_error=_is_error(text))
                case "result":
                    add("assistant", "message", f"[final answer]\n{data['final_output']}", stamp)
                case "run_complete":
                    metadata["elapsed_s"] = data["elapsed_s"]
                    metadata["total_steps"] = data["total_steps"]
                case other:
                    raise ValueError(f"unknown event type {other!r}")
        return steps


def _is_error(text: str) -> bool:
    """Tool errors are JSON objects whose first key is `error`.

    The harness truncates long tool results, which leaves some error objects and think_tool echoes as incomplete
    JSON, so the check reads the opening of the result instead of parsing it.
    """
    return ERROR_OPENING.match(text) is not None
