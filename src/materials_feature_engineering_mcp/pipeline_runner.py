"""
Feature pipeline fit/transform orchestration for composition and SMILES data.
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np
import pandas as pd
from pymatgen.core import Composition
from sklearn.preprocessing import LabelEncoder

from .config import get_download_url, get_static_url
from .feature_generator import MaterialsFeatureGenerator
from .feature_selector import FeatureSelector
from .pipeline_store import create_pipeline_dir, list_pipelines as list_saved_pipelines, load_metadata, save_metadata, save_transform_output
from .smiles_featurizer import DEFAULT_DESCRIPTOR_NAMES, SmilesFeaturizer
from .utils import _read_tabular_data, _validate_data_path


DEFAULT_COMPOSITION_FEATURE_TYPES = [
    "element_property",
    "stoichiometry",
    "valence_orbital",
    "element_amount",
]

DEFAULT_SMILES_FEATURE_TYPES = ["descriptors", "morgan"]


def summarize_dataset(data_path: str, sample_rows: int = 5, example_values: int = 5) -> Dict[str, Any]:
    _validate_data_path(data_path)
    df = _read_tabular_data(data_path)

    columns = []
    for column in df.columns:
        series = df[column]
        non_null_values = []
        for value in series.dropna().head(example_values).tolist():
            if hasattr(value, "item"):
                try:
                    value = value.item()
                except Exception:
                    pass
            non_null_values.append(value)

        columns.append({
            "name": str(column),
            "dtype": str(series.dtype),
            "missing_count": int(series.isna().sum()),
            "missing_rate": float(series.isna().mean()),
            "unique_count": int(series.nunique(dropna=True)),
            "examples": non_null_values,
        })

    warnings: List[str] = []
    if df.columns.duplicated().any():
        warnings.append("Dataset contains duplicate column names.")
    if df.empty:
        warnings.append("Dataset is empty.")

    preview_rows = _json_safe_dataframe(df.head(sample_rows))

    return {
        "input_file": data_path,
        "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "columns": columns,
        "preview_rows": preview_rows,
        "warnings": warnings,
    }


def fit_feature_pipeline(
    data_path: str,
    target_column: str,
    task_type: str,
    composition_columns: Optional[List[str]] = None,
    smiles_columns: Optional[List[str]] = None,
    passthrough_columns: Optional[List[str]] = None,
    pipeline_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    _validate_data_path(data_path)
    df = _read_tabular_data(data_path)
    _validate_column_exists(df, target_column, "target_column")

    normalized_task_type = _normalize_task_type(task_type)
    config = _normalize_pipeline_config(pipeline_config)
    column_spec = _normalize_column_spec(
        df,
        target_column=target_column,
        composition_columns=composition_columns or [],
        smiles_columns=smiles_columns or [],
        passthrough_columns=passthrough_columns or [],
    )

    features_df, feature_summary, quality_warnings = _build_feature_frame(df, column_spec, config)
    if features_df.empty:
        raise ValueError("No features were generated. Please provide composition, smiles, or passthrough columns.")

    prepared_features = _prepare_feature_frame(features_df)
    prepared_target, target_metadata = _prepare_target_series(df[target_column], normalized_task_type)

    pipeline_id, pipeline_dir = create_pipeline_dir()
    _selection_result, selected_features, selection_metadata = _run_feature_selection(
        prepared_features,
        prepared_target,
        target_column,
        df,
        normalized_task_type,
        config,
        pipeline_dir,
    )

    metadata = {
        "pipeline_id": pipeline_id,
        "created_at": datetime.now().isoformat(),
        "source_data": {
            "input_file": data_path,
            "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
            "basename": Path(data_path).name,
        },
        "task_type": normalized_task_type,
        "target_column": target_column,
        "column_spec": column_spec,
        "featurization_config": config["featurization"],
        "selection_config": config["selection"],
        "target_metadata": target_metadata,
        "feature_summary": feature_summary,
        "warnings": quality_warnings,
        "training_data_fingerprint": _dataset_fingerprint(df),
        "selected_features": selected_features,
        "feature_order": selected_features,
        "artifacts": selection_metadata["artifacts"],
    }
    save_metadata(pipeline_dir, metadata)

    return {
        "pipeline_id": pipeline_id,
        "output_file": get_download_url(selection_metadata["artifacts"]["training_features"]),
        "feature_columns": selected_features,
        "pipeline_summary": {
            "composition_columns": column_spec["composition_columns"],
            "smiles_columns": column_spec["smiles_columns"],
            "passthrough_columns": column_spec["passthrough_columns"],
            "generated_feature_count": int(prepared_features.shape[1]),
            "selected_feature_count": len(selected_features),
        },
        "warnings": quality_warnings,
        "artifacts": {
            key: _artifact_to_public_url(path)
            for key, path in selection_metadata["artifacts"].items()
        },
    }


def transform_with_pipeline(pipeline_id: str, data_path: str) -> Dict[str, Any]:
    metadata = load_metadata(pipeline_id)
    df = _read_tabular_data(data_path)
    column_spec = cast(Dict[str, List[str]], metadata["column_spec"])
    config = {"featurization": metadata["featurization_config"]}

    _validate_required_pipeline_inputs(df, column_spec)
    features_df, feature_summary, quality_warnings = _build_feature_frame(df, column_spec, config)
    prepared_features = _prepare_feature_frame(features_df)

    expected_features = list(metadata["selected_features"])
    missing_columns = [column for column in expected_features if column not in prepared_features.columns]
    dropped_columns = [column for column in prepared_features.columns if column not in expected_features]

    aligned = prepared_features.reindex(columns=expected_features, fill_value=0)
    if metadata["target_column"] in df.columns:
        aligned[metadata["target_column"]] = df[metadata["target_column"]].values

    output_path = _save_transform_output(pipeline_id, data_path, aligned)

    warnings: List[str] = list(quality_warnings)
    if missing_columns:
        warnings.append(f"Filled {len(missing_columns)} missing feature columns with 0.")
    if dropped_columns:
        warnings.append(f"Dropped {len(dropped_columns)} columns not present in the trained pipeline.")
    if expected_features and len(missing_columns) / len(expected_features) >= 0.25:
        warnings.append("Large feature alignment drift detected between training and transform datasets.")

    return {
        "pipeline_id": pipeline_id,
        "input_file": data_path,
        "output_file": get_download_url(str(output_path)),
        "feature_columns": expected_features,
        "missing_columns": missing_columns,
        "dropped_columns": dropped_columns,
        "feature_summary": feature_summary,
        "warnings": warnings,
    }


def inspect_pipeline(pipeline_id: str) -> Dict[str, Any]:
    metadata = load_metadata(pipeline_id)
    artifacts = metadata.get("artifacts", {})
    metadata["artifacts"] = {
        key: _artifact_to_public_url(path) for key, path in artifacts.items()
    }
    return metadata


def list_pipelines() -> Dict[str, Any]:
    return {"pipelines": list_saved_pipelines()}


def _normalize_column_spec(
    df: pd.DataFrame,
    target_column: str,
    composition_columns: List[str],
    smiles_columns: List[str],
    passthrough_columns: List[str],
) -> Dict[str, List[str]]:
    for columns, label in [
        (composition_columns, "composition_columns"),
        (smiles_columns, "smiles_columns"),
        (passthrough_columns, "passthrough_columns"),
    ]:
        for column in columns:
            _validate_column_exists(df, column, label)

    all_feature_columns = composition_columns + smiles_columns + passthrough_columns
    if len(all_feature_columns) != len(set(all_feature_columns)):
        raise ValueError("Feature column groups overlap. Each column can only belong to one group.")
    if target_column in all_feature_columns:
        raise ValueError("target_column cannot also be used as a feature input column.")

    return {
        "composition_columns": composition_columns,
        "smiles_columns": smiles_columns,
        "passthrough_columns": passthrough_columns,
    }


def _normalize_pipeline_config(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    config = config or {}
    featurization = config.get("featurization", {})
    selection = config.get("selection", {})

    method = str(selection.get("method", "rfecv")).lower()
    if method not in {"rfecv", "none"}:
        raise ValueError("selection.method must be 'rfecv' or 'none'")

    morgan_radius = int(featurization.get("morgan_radius", 2))
    morgan_n_bits = int(featurization.get("morgan_n_bits", 2048))
    cv_folds = int(selection.get("cv_folds", 5))
    min_features = int(selection.get("min_features", 1))
    step = int(selection.get("step", 1))

    if morgan_radius <= 0:
        raise ValueError("featurization.morgan_radius must be a positive integer")
    if morgan_n_bits <= 0:
        raise ValueError("featurization.morgan_n_bits must be a positive integer")
    if cv_folds <= 1:
        raise ValueError("selection.cv_folds must be greater than 1")
    if min_features <= 0:
        raise ValueError("selection.min_features must be a positive integer")
    if step <= 0:
        raise ValueError("selection.step must be a positive integer")

    return {
        "featurization": {
            "composition_feature_types": featurization.get(
                "composition_feature_types",
                list(DEFAULT_COMPOSITION_FEATURE_TYPES),
            ),
            "smiles_feature_types": featurization.get(
                "smiles_feature_types",
                list(DEFAULT_SMILES_FEATURE_TYPES),
            ),
            "descriptor_names": featurization.get(
                "descriptor_names",
                list(DEFAULT_DESCRIPTOR_NAMES),
            ),
            "morgan_radius": morgan_radius,
            "morgan_n_bits": morgan_n_bits,
        },
        "selection": {
            "method": method,
            "cv_folds": cv_folds,
            "min_features": min_features,
            "step": step,
        },
    }


def _build_feature_frame(
    df: pd.DataFrame,
    column_spec: Dict[str, List[str]],
    config: Dict[str, Any],
) -> tuple[pd.DataFrame, Dict[str, Any], List[str]]:
    frames: List[pd.DataFrame] = []
    summary: Dict[str, Any] = {
        "composition": {},
        "smiles": {},
        "passthrough": column_spec["passthrough_columns"],
    }

    composition_columns = column_spec["composition_columns"]
    if composition_columns:
        generator = MaterialsFeatureGenerator()
        generator.data = df.copy()
        composition_feature_types = config["featurization"]["composition_feature_types"]
        for column in composition_columns:
            generated = generator.generate_composition_features(
                column,
                feature_types=composition_feature_types,
            )
            composition_metadata = _inspect_composition_series(df[column])
            generated = generated.add_prefix(f"{column}__")
            frames.append(generated)
            summary["composition"][column] = {
                "feature_count": int(generated.shape[1]),
                "feature_types": composition_feature_types,
                "invalid_count": composition_metadata["invalid_count"],
                "invalid_indices": composition_metadata["invalid_indices"],
            }

    smiles_columns = column_spec["smiles_columns"]
    if smiles_columns:
        featurizer = SmilesFeaturizer(
            feature_types=tuple(config["featurization"]["smiles_feature_types"]),
            descriptor_names=tuple(config["featurization"]["descriptor_names"]),
            morgan_radius=config["featurization"]["morgan_radius"],
            morgan_n_bits=config["featurization"]["morgan_n_bits"],
        )
        for column in smiles_columns:
            generated, smiles_metadata = featurizer.featurize_series(df[column])
            generated = generated.add_prefix(f"{column}__")
            frames.append(generated)
            summary["smiles"][column] = {
                "feature_count": int(generated.shape[1]),
                "invalid_count": smiles_metadata["invalid_count"],
                "invalid_indices": smiles_metadata["invalid_indices"],
                "feature_types": list(config["featurization"]["smiles_feature_types"]),
            }

    if column_spec["passthrough_columns"]:
        passthrough_df = df[column_spec["passthrough_columns"]].copy()
        non_numeric = [
            column for column in passthrough_df.columns
            if not pd.api.types.is_numeric_dtype(passthrough_df[column])
        ]
        if non_numeric:
            raise ValueError(
                f"passthrough_columns must be numeric. Unsupported columns: {non_numeric}"
            )
        frames.append(passthrough_df)

    if frames:
        feature_frame = pd.concat(frames, axis=1)
    else:
        feature_frame = pd.DataFrame(index=df.index)

    return feature_frame, summary, _build_feature_quality_warnings(summary)


def _prepare_feature_frame(features_df: pd.DataFrame) -> pd.DataFrame:
    prepared = features_df.copy()
    for column in prepared.columns:
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
    prepared = prepared.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return prepared.astype(float)


def _prepare_target_series(target: pd.Series, task_type: str) -> tuple[pd.Series, Dict[str, Any]]:
    if task_type == "classification":
        encoder = LabelEncoder()
        encoded = encoder.fit_transform(target.astype(str).fillna("missing"))
        return (
            pd.Series(encoded, index=target.index, name=target.name),
            {"classes": encoder.classes_.tolist()},
        )

    numeric = pd.to_numeric(target, errors="coerce")
    if numeric.isna().any():
        fill_value = float(numeric.median()) if not pd.isna(numeric.median()) else 0.0
        numeric = numeric.fillna(fill_value)
    return pd.Series(numeric, index=target.index, name=target.name, dtype=float), {}


def _run_feature_selection(
    features: pd.DataFrame,
    target: pd.Series,
    target_column: str,
    original_df: pd.DataFrame,
    task_type: str,
    config: Dict[str, Any],
    pipeline_dir: Path,
) -> tuple[Optional[Dict[str, Any]], List[str], Dict[str, Any]]:
    selection_config = config["selection"]
    method = selection_config["method"]
    training_output_path = pipeline_dir / "training_features.csv"

    if method == "none" or features.shape[1] <= 1 or features.shape[0] < 3:
        result_df = features.copy()
        result_df[target_column] = original_df[target_column].values
        result_df.to_csv(training_output_path, index=False)
        return None, features.columns.tolist(), {
            "artifacts": {"training_features": str(training_output_path)},
        }

    if task_type == "classification":
        class_counts = target.value_counts()
        if class_counts.min() < selection_config["cv_folds"]:
            result_df = features.copy()
            result_df[target_column] = original_df[target_column].values
            result_df.to_csv(training_output_path, index=False)
            return None, features.columns.tolist(), {
                "artifacts": {"training_features": str(training_output_path)},
            }

    selector = FeatureSelector(
        task_type=task_type,
        cv_folds=selection_config["cv_folds"],
        min_features=selection_config["min_features"],
        step=selection_config["step"],
    )
    result = selector.select_features(features, target)
    data_file, report_file, html_file, details_file = selector.save_results(
        result,
        features,
        target,
        str(training_output_path),
        [target_column],
        original_df[[target_column]],
    )
    return result, list(result["selected_features"]), {
        "artifacts": {
            "training_features": data_file,
            "selection_report_png": report_file,
            "selection_report_html": html_file,
            "selection_report_details": details_file,
        },
    }


def _save_transform_output(pipeline_id: str, source_path: str, transformed: pd.DataFrame) -> Path:
    csv_content = transformed.to_csv(index=False)
    return save_transform_output(pipeline_id, Path(source_path).name, csv_content)


def _validate_column_exists(df: pd.DataFrame, column_name: str, label: str) -> None:
    if column_name not in df.columns:
        raise ValueError(f"{label} column not found in dataset: {column_name}")


def _validate_required_pipeline_inputs(df: pd.DataFrame, column_spec: Dict[str, List[str]]) -> None:
    required_columns = (
        column_spec.get("composition_columns", [])
        + column_spec.get("smiles_columns", [])
        + column_spec.get("passthrough_columns", [])
    )
    for column_name in required_columns:
        if column_name not in df.columns:
            raise ValueError(f"Required pipeline input column not found: {column_name}")


def _artifact_to_public_url(path: str) -> str:
    if path.endswith(".html"):
        return get_static_url(path)
    return get_download_url(path)


def _dataset_fingerprint(df: pd.DataFrame) -> Dict[str, Any]:
    column_signature = "|".join(map(str, df.columns))
    payload = f"{df.shape[0]}::{df.shape[1]}::{column_signature}".encode("utf-8")
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_hash": hashlib.sha256(payload).hexdigest(),
    }


def _normalize_task_type(task_type: str) -> str:
    normalized = task_type.lower().strip()
    if normalized not in {"regression", "classification"}:
        raise ValueError("task_type must be 'regression' or 'classification'")
    return normalized


def _inspect_composition_series(series: pd.Series, max_indices: int = 20) -> Dict[str, Any]:
    invalid_indices: List[Any] = []
    for index, value in series.items():
        if pd.isna(value) or not str(value).strip():
            invalid_indices.append(index)
            continue
        try:
            Composition(str(value).replace(" ", ""))
        except Exception:
            invalid_indices.append(index)

    return {
        "invalid_count": len(invalid_indices),
        "invalid_indices": [_json_safe_scalar(index) for index in invalid_indices[:max_indices]],
    }


def _build_feature_quality_warnings(feature_summary: Dict[str, Any]) -> List[str]:
    warnings: List[str] = []
    for feature_kind in ("composition", "smiles"):
        for column_name, metadata in feature_summary.get(feature_kind, {}).items():
            invalid_count = int(metadata.get("invalid_count", 0) or 0)
            if invalid_count <= 0:
                continue
            invalid_indices = metadata.get("invalid_indices", [])
            warning = f"{feature_kind} column '{column_name}' contains {invalid_count} invalid values."
            if invalid_indices:
                warning += f" Sample row indices: {invalid_indices}."
            warnings.append(warning)
    return warnings


def _json_safe_dataframe(df: pd.DataFrame) -> List[Dict[str, Any]]:
    records = df.replace({np.nan: None}).to_dict(orient="records")
    safe_records: List[Dict[str, Any]] = []
    for record in records:
        safe_record: Dict[str, Any] = {}
        for key, value in record.items():
            safe_record[str(key)] = _json_safe_scalar(value)
        safe_records.append(safe_record)
    return safe_records


def _json_safe_scalar(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    return value
