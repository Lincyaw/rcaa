from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from importlib.metadata import distribution
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TRACKED_INPUTS = (
    "traj.yaml", "pyproject.toml", "uv.lock", "engine", "adapters", "operators", "samplers", "studies",
    ".claude/skills", "scripts",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def input_hashes() -> dict[str, str]:
    files: list[Path] = []
    for item in TRACKED_INPUTS:
        path = ROOT / item
        files.extend(path.rglob("*") if path.is_dir() else [path])
    return {
        str(path.relative_to(ROOT)): digest(path)
        for path in sorted(files)
        if path.is_file() and "__pycache__" not in path.parts
    }


def git_state(root: Path) -> dict[str, Any]:
    def git(*args: str) -> str:
        result = subprocess.run(
            ["git", *args], cwd=root, text=True, capture_output=True, check=False
        )
        return result.stdout.strip() if result.returncode == 0 else ""

    return {"commit": git("rev-parse", "HEAD") or None, "dirty": bool(git("status", "--porcelain"))}


def installed_package(name: str) -> dict[str, Any]:
    package = distribution(name)
    direct_url = package.read_text("direct_url.json")
    return {
        "name": package.metadata["Name"],
        "version": package.version,
        "direct_url": json.loads(direct_url) if direct_url is not None else None,
    }


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def run(command: list[str], output: Path) -> None:
    print("+", " ".join(command), flush=True)
    partial = output.with_suffix(output.suffix + ".partial")
    with partial.open("w", encoding="utf-8") as stream:
        # Keep stderr attached so long extraction runs continue to show progress.
        result = subprocess.run(command, cwd=ROOT, text=True, stdout=stream, check=False)
    if result.returncode:
        raise subprocess.CalledProcessError(result.returncode, command)
    os.replace(partial, output)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the reproducible RCABench trajectory-analysis workflow.")
    parser.add_argument("--dataset", action="append", help="Dataset to ingest/extract; repeatable (default: all).")
    parser.add_argument("--sampler", action="append", help="Sampler to run; repeatable (default: default).")
    parser.add_argument("--skip-ingest", action="store_true")
    parser.add_argument("--skip-extract", action="store_true")
    parser.add_argument("--skip-sample", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Only use cached LLM extraction results.")
    parser.add_argument("--limit", type=int, help="Limit extraction for a smoke run.")
    args = parser.parse_args()

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    run_dir = ROOT / ".traj" / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    manifest: dict[str, Any] = {
        "run_id": run_id,
        "started_at": datetime.now(UTC).isoformat(),
        "status": "running",
        "arguments": vars(args),
        "git": git_state(ROOT),
        "traj_analyzer": installed_package("traj-analyzer"),
        "inputs": input_hashes(),
        "commands": [],
    }
    write_json(run_dir / "manifest.json", manifest)
    traj = [sys.executable, "-m", "traj_analyzer.cli"]

    def execute(command: list[str], output: Path) -> None:
        manifest["commands"].append(command)
        write_json(run_dir / "manifest.json", manifest)
        run(command, output)

    try:
        if not args.skip_ingest:
            command = [*traj, "ingest", *(args.dataset or [])]
            execute(command, run_dir / "ingest.json")

        if not args.skip_extract:
            command = [*traj, "extract"]
            for dataset in args.dataset or []:
                command += ["--dataset", dataset]
            if args.limit is not None:
                command += ["--limit", str(args.limit)]
            if args.dry_run:
                command.append("--dry-run")
            execute(command, run_dir / "extract.json")

        command = [*traj, "table"]
        for dataset in args.dataset or []:
            command += ["--dataset", dataset]
        execute(command, run_dir / "features.csv")

        if not args.skip_sample:
            for sampler in args.sampler or ["default"]:
                command = [*traj, "sample", "--sampler", sampler]
                execute(command, run_dir / f"sample-{sampler}.json")
    except KeyboardInterrupt:
        manifest.update(status="interrupted", finished_at=datetime.now(UTC).isoformat())
        write_json(run_dir / "manifest.json", manifest)
        print(f"run interrupted; manifest: {run_dir / 'manifest.json'}", file=sys.stderr)
        return 130
    except Exception as error:
        manifest.update(status="failed", finished_at=datetime.now(UTC).isoformat(), error=repr(error))
        write_json(run_dir / "manifest.json", manifest)
        print(f"run failed; manifest: {run_dir / 'manifest.json'}", file=sys.stderr)
        return 1

    manifest.update(status="complete", finished_at=datetime.now(UTC).isoformat())
    write_json(run_dir / "manifest.json", manifest)
    print(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
