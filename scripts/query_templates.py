import hashlib
import sys
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import pandas as pd
from traj_analyzer import sql
from traj_analyzer.project import Project

PROJECT = Project.find()
sys.path.insert(0, str(PROJECT.operators_dir))
from rca._parse import DIALECT  # noqa: E402

# One row per query of .traj/queries.parquet with the template it follows at three levels, and one row per template.
QUERIES = PROJECT.data_dir / "queries.parquet"
OUT_QUERIES = PROJECT.data_dir / "query_templates.parquet"
OUT_TEMPLATES = PROJECT.data_dir / "templates.parquet"

WINDOWS = ("abnormal_", "normal_")


def digest(text: str) -> str:
    return hashlib.blake2b(text.encode(), digest_size=8).hexdigest()


def file_name(path: str) -> str:
    """A telemetry file by its base name, with its window."""
    return path.rsplit("/", 1)[-1].lower().removesuffix(".parquet")


def file_kind(path: str) -> str:
    """A telemetry file by its kind, so the incident and baseline files share one name."""
    name = file_name(path)
    for prefix in WINDOWS:
        name = name.removeprefix(prefix)
    return name


def templates(text: str) -> dict[str, Any]:
    """Template ids of one query at three levels, or the error that kept the query from being resolved.

    sqlglot fails on some of the agents' queries in several ways; each failure is recorded by its type and counted
    in the output.
    """
    tree = sql.parse(text, DIALECT)
    assert tree is not None
    try:
        resolved = sql.resolve(tree, DIALECT)
    except Exception as error:
        return {"error": type(error).__name__}
    return {
        "error": None,
        "literal": digest(sql.literal_template(tree, DIALECT, file_name)),
        "structure": digest(sql.Canonical(resolved, file_name).text),
        "merged": digest(sql.Canonical(resolved, file_kind).text),
    }


def parts(text: str) -> tuple[str, list[str]]:
    """The canonical text of a query and the texts of its parts, for comparing templates by the parts they share."""
    tree = sql.parse(text, DIALECT)
    assert tree is not None
    canonical = sql.Canonical(sql.resolve(tree, DIALECT), file_kind)
    return canonical.text, sorted(canonical.parts)


if __name__ == "__main__":
    queries = pd.read_parquet(QUERIES, columns=["key", "position", "sql", "parsed", "succeeded"])
    parsed = queries[queries["parsed"]].reset_index(drop=True)
    unique = parsed["sql"].drop_duplicates().tolist()
    with ProcessPoolExecutor(PROJECT.config.extract.code_workers * 2) as pool:
        results = dict(zip(unique, pool.map(templates, unique, chunksize=512), strict=True))
        for level in ("error", "literal", "structure", "merged"):
            parsed[level] = parsed["sql"].map(lambda text, level=level: results[text].get(level))
        parsed.drop(columns=["sql"]).to_parquet(OUT_QUERIES)
        kept = parsed[parsed["error"].isna()]
        first = kept.drop_duplicates("merged").set_index("merged")
        summary = kept.groupby("merged").agg(queries=("key", "size"), trajectories=("key", "nunique"),
                                             structures=("structure", "nunique"), literals=("literal", "nunique"),
                                             succeeded=("succeeded", "mean"))
        summary["example"] = first["sql"]
        described = list(pool.map(parts, summary["example"], chunksize=512))
    summary["text"] = [text for text, _ in described]
    summary["parts"] = [found for _, found in described]
    summary.sort_values("queries", ascending=False).to_parquet(OUT_TEMPLATES)
    print(f"{len(queries)} queries, {len(parsed)} parsed, {len(unique)} distinct texts")
    print("errors:", parsed["error"].value_counts().to_dict())
    print({level: int(kept[level].nunique()) for level in ("literal", "structure", "merged")})
