import sys
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import pandas as pd
from traj_analyzer.ingest import load_index, load_trajectory
from traj_analyzer.project import Project

PROJECT = Project.find()
sys.path.insert(0, str(PROJECT.operators_dir))
from rca._parse import QUERY, calls, error_kind, filter_values, query_atoms, services_in  # noqa: E402

# One row per query_parquet_files call of every ingested trajectory, in the same terms as the rca.sql_shape operator.
OUT = PROJECT.data_dir / "queries.parquet"


def rows(key: str) -> list[dict[str, Any]]:
    queries = [c for c in calls(load_trajectory(PROJECT, key)) if c.name == QUERY]
    out = []
    for position, call in enumerate(queries):
        tree = call.tree
        out.append({
            "key": key,
            "position": position,
            "step": call.step.index,
            "turn": call.turn,
            "sql": call.sql,
            "parsed": tree is not None,
            "succeeded": call.succeeded,
            "error_kind": error_kind(call.result.content) if call.failed else None,
            "files": sorted(f"{window}_{kind}" for window, kind in call.telemetry),
            "atoms": sorted(query_atoms(tree)),
            "filter_values": sorted(filter_values(tree)),
            "services": sorted(services_in(tree)),
        })
    return out


if __name__ == "__main__":
    keys = [row.key for row in load_index(PROJECT)]
    with ProcessPoolExecutor(PROJECT.config.extract.code_workers) as pool:
        table = pd.DataFrame([row for part in pool.map(rows, keys, chunksize=64) for row in part])
    table.to_parquet(OUT)
    print(f"{len(table)} queries of {table['key'].nunique()} trajectories -> {OUT}")
