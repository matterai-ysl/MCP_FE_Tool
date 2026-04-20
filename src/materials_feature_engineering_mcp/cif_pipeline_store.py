"""
Persistent local storage helpers for CIF feature pipelines.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
from uuid import uuid4

from .utils import DATA_DIR, _resolve_path_within_data_dir


CIF_PIPELINES_DIR = DATA_DIR / "cif_pipelines"


def _ensure_cif_pipelines_root() -> Path:
    root = _resolve_path_within_data_dir(CIF_PIPELINES_DIR)
    root.mkdir(parents=True, exist_ok=True)
    return root


def create_cif_pipeline_dir() -> tuple[str, Path]:
    pipeline_id = str(uuid4())
    pipeline_dir = _resolve_path_within_data_dir(_ensure_cif_pipelines_root() / pipeline_id)
    pipeline_dir.mkdir(parents=True, exist_ok=False)
    (pipeline_dir / "transforms").mkdir(exist_ok=True)
    (pipeline_dir / "extracted_archive").mkdir(exist_ok=True)
    return pipeline_id, pipeline_dir


def get_cif_pipeline_dir(pipeline_id: str) -> Path:
    pipeline_dir = _resolve_path_within_data_dir(_ensure_cif_pipelines_root() / pipeline_id)
    if not pipeline_dir.exists():
        raise FileNotFoundError(f"CIF pipeline not found: {pipeline_id}")
    return pipeline_dir


def save_cif_metadata(pipeline_dir: Path, metadata: Dict[str, Any]) -> Path:
    metadata_path = pipeline_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata_path


def load_cif_metadata(pipeline_id: str) -> Dict[str, Any]:
    metadata_path = get_cif_pipeline_dir(pipeline_id) / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing CIF pipeline metadata: {metadata_path}")
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def save_cif_transform_output(pipeline_id: str, source_name: str, csv_content: str) -> Path:
    pipeline_dir = get_cif_pipeline_dir(pipeline_id)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_name = f"{Path(source_name).stem}_{timestamp}_features.csv"
    output_path = pipeline_dir / "transforms" / output_name
    output_path.write_text(csv_content, encoding="utf-8")
    return output_path


def list_cif_pipelines() -> List[Dict[str, Any]]:
    root = _ensure_cif_pipelines_root()
    summaries: List[Dict[str, Any]] = []
    for item in sorted(root.iterdir()):
        if not item.is_dir():
            continue
        metadata_path = item / "metadata.json"
        if not metadata_path.exists():
            continue
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        summaries.append({
            "pipeline_id": metadata.get("pipeline_id", item.name),
            "pipeline_type": metadata.get("pipeline_type"),
            "created_at": metadata.get("created_at"),
            "target_column": metadata.get("target_column"),
            "task_type": metadata.get("task_type"),
            "source_data": metadata.get("source_data"),
            "selected_feature_count": len(metadata.get("selected_features", [])),
            "path": str(item),
        })
    summaries.sort(key=lambda item: item.get("created_at") or "", reverse=True)
    return summaries
