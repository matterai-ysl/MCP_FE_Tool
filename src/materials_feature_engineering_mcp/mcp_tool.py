"""
MCP tools for chemistry-focused feature pipelines.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from fastmcp import Context, FastMCP

from .cif_pipeline_runner import (
    fit_cif_pipeline as fit_cif_pipeline_runner,
    inspect_cif_pipeline as inspect_cif_pipeline_runner,
    list_cif_pipelines as list_cif_pipelines_runner,
    summarize_cif_archive as summarize_cif_archive_runner,
    transform_with_cif_pipeline as transform_with_cif_pipeline_runner,
)
from .pipeline_runner import (
    fit_feature_pipeline as fit_feature_pipeline_runner,
    inspect_pipeline as inspect_pipeline_runner,
    list_pipelines as list_pipelines_runner,
    summarize_dataset as summarize_dataset_runner,
    transform_with_pipeline as transform_with_pipeline_runner,
)
from .utils import _resolve_path_within_data_dir, _sanitize_user_id


mcp = FastMCP("Materials Feature Engineering Tool")


def _create_user_output_dir(user_id: Optional[str] = None) -> Tuple[str, str]:
    """
    Backward-compatible helper kept for existing callers and regression tests.
    """
    safe_user_id = _sanitize_user_id(user_id)
    run_uuid = str(uuid.uuid4())
    output_dir = _resolve_path_within_data_dir(Path("data") / safe_user_id / run_uuid)
    output_dir.mkdir(parents=True, exist_ok=True)
    return str(output_dir), run_uuid


def _get_user_id_from_context(ctx: Optional[Context]) -> Optional[str]:
    if ctx is None:
        return None

    try:
        return ctx.request_context.request.headers.get("user_id", None)  # type: ignore[attr-defined]
    except Exception:
        return None


def _json_safe(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if isinstance(obj, pd.Series):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(key): _json_safe(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_json_safe(item) for item in obj]

    try:
        json.dumps(obj)
        return obj
    except Exception:
        return str(obj)


@mcp.tool()
def summarize_dataset(
    data_path: str,
    sample_rows: int = 5,
    example_values: int = 5,
) -> Dict[str, Any]:
    """
    Summarize a dataset for agent-side column understanding.
    """
    result = summarize_dataset_runner(
        data_path=data_path,
        sample_rows=sample_rows,
        example_values=example_values,
    )
    return _json_safe(result)  # type: ignore[return-value]


@mcp.tool()
def fit_feature_pipeline(
    data_path: str,
    target_column: str,
    task_type: str,
    composition_columns: Optional[list[str]] = None,
    smiles_columns: Optional[list[str]] = None,
    passthrough_columns: Optional[list[str]] = None,
    pipeline_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Fit a chemistry feature pipeline on a training dataset and persist it locally.
    """
    result = fit_feature_pipeline_runner(
        data_path=data_path,
        target_column=target_column,
        task_type=task_type,
        composition_columns=composition_columns,
        smiles_columns=smiles_columns,
        passthrough_columns=passthrough_columns,
        pipeline_config=pipeline_config,
    )
    return _json_safe(result)  # type: ignore[return-value]


@mcp.tool()
def transform_with_pipeline(
    pipeline_id: str,
    data_path: str,
) -> Dict[str, Any]:
    """
    Transform a test or prediction dataset with an existing persisted pipeline.
    """
    result = transform_with_pipeline_runner(
        pipeline_id=pipeline_id,
        data_path=data_path,
    )
    return _json_safe(result)  # type: ignore[return-value]


@mcp.tool()
def inspect_pipeline(pipeline_id: str) -> Dict[str, Any]:
    """
    Inspect a persisted chemistry feature pipeline.
    """
    result = inspect_pipeline_runner(pipeline_id=pipeline_id)
    return _json_safe(result)  # type: ignore[return-value]


@mcp.tool()
def list_pipelines() -> Dict[str, Any]:
    """
    List persisted local chemistry feature pipelines.
    """
    result = list_pipelines_runner()
    return _json_safe(result)  # type: ignore[return-value]


@mcp.tool()
def summarize_cif_archive(
    structure_archive: str,
    metadata_table: str,
    cif_filename_column: str,
    sample_files: int = 10,
    sample_rows: int = 5,
) -> Dict[str, Any]:
    """
    Summarize a CIF archive and metadata table before fitting a CIF pipeline.
    """
    result = summarize_cif_archive_runner(
        structure_archive=structure_archive,
        metadata_table=metadata_table,
        cif_filename_column=cif_filename_column,
        sample_files=sample_files,
        sample_rows=sample_rows,
    )
    return _json_safe(result)  # type: ignore[return-value]


@mcp.tool()
def fit_cif_pipeline(
    structure_archive: str,
    metadata_table: str,
    cif_filename_column: str,
    target_column: str,
    task_type: str,
    pipeline_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Fit a CIF feature pipeline from a structure archive and metadata table.
    """
    result = fit_cif_pipeline_runner(
        structure_archive=structure_archive,
        metadata_table=metadata_table,
        cif_filename_column=cif_filename_column,
        target_column=target_column,
        task_type=task_type,
        pipeline_config=pipeline_config,
    )
    return _json_safe(result)  # type: ignore[return-value]


@mcp.tool()
def transform_with_cif_pipeline(
    pipeline_id: str,
    structure_archive: str,
    metadata_table: str,
    cif_filename_column: str,
) -> Dict[str, Any]:
    """
    Transform CIF structures with an existing persisted CIF pipeline.
    """
    result = transform_with_cif_pipeline_runner(
        pipeline_id=pipeline_id,
        structure_archive=structure_archive,
        metadata_table=metadata_table,
        cif_filename_column=cif_filename_column,
    )
    return _json_safe(result)  # type: ignore[return-value]


@mcp.tool()
def inspect_cif_pipeline(pipeline_id: str) -> Dict[str, Any]:
    """
    Inspect a persisted CIF pipeline.
    """
    result = inspect_cif_pipeline_runner(pipeline_id=pipeline_id)
    return _json_safe(result)  # type: ignore[return-value]


@mcp.tool()
def list_cif_pipelines() -> Dict[str, Any]:
    """
    List persisted local CIF pipelines.
    """
    result = list_cif_pipelines_runner()
    return _json_safe(result)  # type: ignore[return-value]


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
