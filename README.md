# RCABench trajectory analysis

This repository is the application layer for analysing RCABench evaluation runs.
It supplies the RCABench adapter, RCA-specific operators, study questions, and sampling policy; [traj-analyzer](https://github.com/Lincyaw/traj-analyzer) supplies the generic `traj` CLI and execution engine.

Both supported coding-agent backends use the same project skills:

- Claude Code discovers the canonical files under `.claude/skills/`.
- Codex discovers `.agents/skills/`, whose entries are relative symlinks to the canonical Claude files.

Edit only `.claude/skills/`; the shared source keeps both backends on the same analysis workflow.

## Inputs and outputs

Put the evaluation database at `data/eval.db`. The configured `ops-lite` dataset reads the `ops-lite` experiment from that database.

Generated state lives under `.traj/` and is intentionally ignored by Git:

- `.traj/datasets/`: normalized trajectories and readable Markdown
- `.traj/features/`: extracted Parquet feature groups
- `.traj/samples/`: timestamped review selections
- `.traj/runs/`: application-run manifests and command outputs

The committed files are the reproducibility contract. `traj.yaml` defines ingestion and extraction, `samplers/` defines review selection, and `studies/` records analysis questions used during report writing. The current `traj` CLI does **not** execute files in `studies/` directly.

## Setup

From this directory:

```bash
uv sync --locked
uv run traj adapters list
uv run traj operators list
```

`traj-analyzer` is installed directly from GitHub at the commit pinned in `pyproject.toml` and `uv.lock`.
No sibling checkout or local editable installation is required.

The LLM-backed `rca_extract` group also needs `LITELLM_API_KEY` and `LITELLM_BASE_URL`. Code operators do not need those credentials.

## Reproducible workflow

Run the whole pipeline with:

```bash
uv run python scripts/run_pipeline.py
```

The wrapper runs `ingest → extract → table → sample`, records exact commands, the application Git revision/dirty flag, installed `traj-analyzer` version and source metadata, and SHA-256 hashes of the analysis configuration and code.
It writes a stable CSV snapshot and JSON command outputs beneath `.traj/runs/<UTC timestamp>/`.

Useful variants:

```bash
# Small extraction smoke run. Sampling is skipped because an existing feature
# store may contain rows outside this limited extraction.
uv run python scripts/run_pipeline.py --limit 10 --skip-sample

# Reuse normalized data and cached LLM results
uv run python scripts/run_pipeline.py --skip-ingest --dry-run

# Produce both review selections
uv run python scripts/run_pipeline.py --sampler default --sampler query_behavior
```

`--dry-run` applies to extraction: uncached LLM results remain unavailable rather than triggering model calls. Sampling uses only trajectories with complete values for its configured features.

For manual steps, the equivalent commands are:

```bash
uv run traj ingest ops-lite
uv run traj extract --dataset ops-lite
uv run traj table > features.csv
uv run traj sample --sampler default
uv run traj view
```

## Analysis policy

The default sampler intentionally uses investigation-process features only. Evaluation results and ground truth are excluded from its distance space, so outcome labels do not leak into qualitative case selection. Outliers are scored within each model before the remaining budget is filled with globally diverse behaviour.

`query_behavior` narrows the same policy to SQL file, expression, and filter-value patterns. Run it when the report question concerns how agents interrogate telemetry.

The files in `studies/` define explicit comparisons for report analysis:

- `gt_probing`: whether the agent queried a true root-cause service
- `service_choice`: whether a queried true service was submitted
- `fault_kind`: whether a submitted true service received the right fault kind
- `network_kind`: the fault-kind question restricted to network incidents

These study specifications include ground-truth and evaluation fields by design. Treat their findings as retrospective evaluation, not as unbiased discovery samples.

## Query-level export

To inspect every SQL call rather than one row per trajectory:

```bash
uv run python scripts/export_queries.py
```

This writes `.traj/queries.parquet` using the same parser and vocabulary as `rca.sql_shape`. Run ingestion first.
