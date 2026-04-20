"""
CIF pipeline fit/transform orchestration.
"""

from __future__ import annotations

import hashlib
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .cif_archive import (
    align_cif_metadata,
    extract_cif_archive,
    load_cif_metadata as load_cif_metadata_table,
    summarize_cif_inputs,
)
from .cif_featurizer import DEFAULT_CIF_COMPOSITION_FEATURE_TYPES, DEFAULT_STRUCTURE_FEATURE_TYPES, CifFeaturizer
from .cif_pipeline_store import (
    create_cif_pipeline_dir,
    list_cif_pipelines as list_saved_cif_pipelines,
    load_cif_metadata,
    save_cif_metadata,
    save_cif_transform_output,
)
from .config import get_download_url
from .pipeline_runner import (
    _artifact_to_public_url,
    _normalize_task_type,
    _prepare_feature_frame,
    _prepare_target_series,
    _run_feature_selection,
)


def summarize_cif_archive(
    structure_archive: str,
    metadata_table: str,
    cif_filename_column: str,
    sample_files: int = 10,
    sample_rows: int = 5,
) -> Dict[str, Any]:
    return summarize_cif_inputs(
        structure_archive=structure_archive,
        metadata_table=metadata_table,
        cif_filename_column=cif_filename_column,
        sample_files=sample_files,
        sample_rows=sample_rows,
    )


def fit_cif_pipeline(
    structure_archive: str,
    metadata_table: str,
    cif_filename_column: str,
    target_column: str,
    task_type: str,
    pipeline_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    normalized_task_type = _normalize_task_type(task_type)
    config = _normalize_cif_pipeline_config(pipeline_config)
    metadata_df = load_cif_metadata_table(metadata_table, cif_filename_column)
    if target_column not in metadata_df.columns:
        raise ValueError(f"target_column not found in metadata_table: {target_column}")

    pipeline_id, pipeline_dir = create_cif_pipeline_dir()
    extraction = extract_cif_archive(structure_archive, pipeline_dir / "extracted_archive")
    alignment = align_cif_metadata(metadata_df, extraction.files_by_name, cif_filename_column)

    ordered_filenames = alignment.matched_metadata[cif_filename_column].astype(str).tolist()
    featurizer = CifFeaturizer(
        structure_feature_types=config["featurization"]["structure_feature_types"],
        composition_feature_types=config["featurization"]["composition_feature_types"],
    )
    features_df, feature_summary, feature_warnings = featurizer.featurize_files(
        extraction.files_by_name,
        ordered_filenames=ordered_filenames,
    )
    prepared_features = _prepare_feature_frame(features_df.reset_index(drop=True))
    prepared_target, target_metadata = _prepare_target_series(
        alignment.matched_metadata[target_column].reset_index(drop=True),
        normalized_task_type,
    )

    _selection_result, selected_features, selection_metadata = _run_feature_selection(
        prepared_features,
        prepared_target,
        target_column,
        alignment.matched_metadata.reset_index(drop=True),
        normalized_task_type,
        config,
        pipeline_dir,
    )

    warnings = extraction.warnings + alignment.warnings + feature_warnings
    metadata = {
        "pipeline_type": "cif",
        "pipeline_id": pipeline_id,
        "created_at": datetime.now().isoformat(),
        "source_data": {
            "structure_archive": structure_archive,
            "metadata_table": metadata_table,
            "matched_rows": int(alignment.matched_metadata.shape[0]),
        },
        "task_type": normalized_task_type,
        "target_column": target_column,
        "cif_filename_column": cif_filename_column,
        "structure_config": {"structure_feature_types": config["featurization"]["structure_feature_types"]},
        "composition_config": {"composition_feature_types": config["featurization"]["composition_feature_types"]},
        "selection_config": config["selection"],
        "target_metadata": target_metadata,
        "feature_summary": feature_summary,
        "warnings": warnings,
        "training_data_fingerprint": _fingerprint_cif_training(alignment.matched_metadata, cif_filename_column),
        "selected_features": selected_features,
        "feature_order": selected_features,
        "artifacts": selection_metadata["artifacts"],
    }
    save_cif_metadata(pipeline_dir, metadata)

    return {
        "pipeline_id": pipeline_id,
        "output_file": get_download_url(selection_metadata["artifacts"]["training_features"]),
        "feature_columns": selected_features,
        "pipeline_summary": {
            "matched_rows": int(alignment.matched_metadata.shape[0]),
            "generated_feature_count": int(prepared_features.shape[1]),
            "selected_feature_count": len(selected_features),
            "structure_feature_types": config["featurization"]["structure_feature_types"],
            "composition_feature_types": config["featurization"]["composition_feature_types"],
        },
        "warnings": warnings,
        "artifacts": {
            key: _artifact_to_public_url(path)
            for key, path in selection_metadata["artifacts"].items()
        },
    }


def transform_with_cif_pipeline(
    pipeline_id: str,
    structure_archive: str,
    metadata_table: str,
    cif_filename_column: str,
) -> Dict[str, Any]:
    metadata = load_cif_metadata(pipeline_id)
    if cif_filename_column != metadata["cif_filename_column"]:
        raise ValueError(
            f"cif_filename_column must match pipeline configuration: {metadata['cif_filename_column']}"
        )

    with tempfile.TemporaryDirectory() as tmp_dir:
        extraction = extract_cif_archive(structure_archive, Path(tmp_dir) / "cifs")
        metadata_df = load_cif_metadata_table(metadata_table, cif_filename_column)
        alignment = align_cif_metadata(metadata_df, extraction.files_by_name, cif_filename_column)
        ordered_filenames = alignment.matched_metadata[cif_filename_column].astype(str).tolist()
        featurizer = CifFeaturizer(
            structure_feature_types=metadata["structure_config"]["structure_feature_types"],
            composition_feature_types=metadata["composition_config"]["composition_feature_types"],
        )
        features_df, feature_summary, feature_warnings = featurizer.featurize_files(
            extraction.files_by_name,
            ordered_filenames=ordered_filenames,
        )

    prepared_features = _prepare_feature_frame(features_df.reset_index(drop=True))
    expected_features = list(metadata["selected_features"])
    missing_columns = [column for column in expected_features if column not in prepared_features.columns]
    dropped_columns = [column for column in prepared_features.columns if column not in expected_features]
    aligned = prepared_features.reindex(columns=expected_features, fill_value=0)

    output_path = save_cif_transform_output(pipeline_id, Path(structure_archive).name, aligned.to_csv(index=False))
    warnings = extraction.warnings + alignment.warnings + feature_warnings
    if missing_columns:
        warnings.append(f"Filled {len(missing_columns)} missing feature columns with 0.")
    if dropped_columns:
        warnings.append(f"Dropped {len(dropped_columns)} columns not present in the trained pipeline.")

    return {
        "pipeline_id": pipeline_id,
        "output_file": get_download_url(str(output_path)),
        "feature_columns": expected_features,
        "missing_columns": missing_columns,
        "dropped_columns": dropped_columns,
        "feature_summary": feature_summary,
        "warnings": warnings,
    }


def inspect_cif_pipeline(pipeline_id: str) -> Dict[str, Any]:
    metadata = load_cif_metadata(pipeline_id)
    metadata["artifacts"] = {
        key: _artifact_to_public_url(path)
        for key, path in metadata.get("artifacts", {}).items()
    }
    return metadata


def list_cif_pipelines() -> Dict[str, Any]:
    return {"pipelines": list_saved_cif_pipelines()}


def _normalize_cif_pipeline_config(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    config = config or {}
    featurization = config.get("featurization", {})
    selection = config.get("selection", {})
    method = str(selection.get("method", "rfecv")).lower()
    if method not in {"rfecv", "none"}:
        raise ValueError("selection.method must be 'rfecv' or 'none'")

    cv_folds = int(selection.get("cv_folds", 5))
    min_features = int(selection.get("min_features", 1))
    step = int(selection.get("step", 1))
    if cv_folds <= 1:
        raise ValueError("selection.cv_folds must be greater than 1")
    if min_features <= 0:
        raise ValueError("selection.min_features must be a positive integer")
    if step <= 0:
        raise ValueError("selection.step must be a positive integer")

    return {
        "featurization": {
            "structure_feature_types": featurization.get(
                "structure_feature_types",
                list(DEFAULT_STRUCTURE_FEATURE_TYPES),
            ),
            "composition_feature_types": featurization.get(
                "composition_feature_types",
                list(DEFAULT_CIF_COMPOSITION_FEATURE_TYPES),
            ),
        },
        "selection": {
            "method": method,
            "cv_folds": cv_folds,
            "min_features": min_features,
            "step": step,
        },
    }


def _fingerprint_cif_training(metadata_df: pd.DataFrame, cif_filename_column: str) -> Dict[str, Any]:
    payload = "|".join(metadata_df[cif_filename_column].astype(str).tolist()).encode("utf-8")
    return {
        "rows": int(metadata_df.shape[0]),
        "filename_hash": hashlib.sha256(payload).hexdigest(),
    }
