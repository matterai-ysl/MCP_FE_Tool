"""
CIF archive loading, extraction, and metadata alignment helpers.
"""

from __future__ import annotations

import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from .utils import _read_tabular_data, _validate_data_path


@dataclass
class CifArchiveExtraction:
    files_by_name: Dict[str, str]
    warnings: List[str]


@dataclass
class CifAlignment:
    metadata: pd.DataFrame
    matched_metadata: pd.DataFrame
    files_by_name: Dict[str, str]
    missing_in_archive: List[str]
    extra_in_archive: List[str]
    warnings: List[str]


def extract_cif_archive(structure_archive: str, output_dir: str | Path) -> CifArchiveExtraction:
    _validate_data_path(structure_archive)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    files_by_name: Dict[str, str] = {}
    warnings: List[str] = []

    with zipfile.ZipFile(structure_archive) as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue
            member_path = Path(member.filename)
            if member_path.suffix.lower() != ".cif":
                warnings.append(f"Skipped non-CIF file in archive: {member.filename}")
                continue

            filename = member_path.name
            if filename in files_by_name:
                raise ValueError(f"Duplicate CIF filename in archive: {filename}")

            output_path = destination / filename
            with archive.open(member) as source, output_path.open("wb") as target:
                shutil.copyfileobj(source, target)
            files_by_name[filename] = str(output_path)

    if not files_by_name:
        raise ValueError("No .cif files found in structure_archive")

    return CifArchiveExtraction(files_by_name=files_by_name, warnings=warnings)


def load_cif_metadata(metadata_table: str, cif_filename_column: str) -> pd.DataFrame:
    _validate_data_path(metadata_table)
    metadata = _read_tabular_data(metadata_table)
    if cif_filename_column not in metadata.columns:
        raise ValueError(f"cif_filename_column not found in metadata_table: {cif_filename_column}")

    duplicate_mask = metadata[cif_filename_column].duplicated(keep=False)
    if duplicate_mask.any():
        duplicates = sorted(metadata.loc[duplicate_mask, cif_filename_column].astype(str).unique().tolist())
        raise ValueError(f"Duplicate CIF filenames in metadata_table: {duplicates}")

    metadata = metadata.copy()
    metadata[cif_filename_column] = metadata[cif_filename_column].astype(str).map(lambda value: Path(value).name)
    return metadata


def align_cif_metadata(
    metadata: pd.DataFrame,
    files_by_name: Dict[str, str],
    cif_filename_column: str,
) -> CifAlignment:
    metadata_names = set(metadata[cif_filename_column].astype(str).tolist())
    archive_names = set(files_by_name)
    missing_in_archive = sorted(metadata_names - archive_names)
    extra_in_archive = sorted(archive_names - metadata_names)

    matched_metadata = metadata[metadata[cif_filename_column].isin(archive_names)].copy()
    if matched_metadata.empty:
        raise ValueError("No metadata rows matched CIF files in the archive")

    warnings: List[str] = []
    if missing_in_archive:
        warnings.append(f"{len(missing_in_archive)} metadata rows reference CIF files missing from archive.")
    if extra_in_archive:
        warnings.append(f"{len(extra_in_archive)} CIF files in archive are not referenced by metadata.")

    return CifAlignment(
        metadata=metadata,
        matched_metadata=matched_metadata,
        files_by_name=files_by_name,
        missing_in_archive=missing_in_archive,
        extra_in_archive=extra_in_archive,
        warnings=warnings,
    )


def summarize_cif_inputs(
    structure_archive: str,
    metadata_table: str,
    cif_filename_column: str,
    sample_files: int = 10,
    sample_rows: int = 5,
) -> Dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmp_dir:
        extraction = extract_cif_archive(structure_archive, Path(tmp_dir) / "cifs")
        metadata = load_cif_metadata(metadata_table, cif_filename_column)
        alignment = align_cif_metadata(metadata, extraction.files_by_name, cif_filename_column)

    preview_rows = metadata.head(sample_rows).replace({pd.NA: None}).to_dict(orient="records")
    columns = [
        {
            "name": str(column),
            "dtype": str(metadata[column].dtype),
            "missing_count": int(metadata[column].isna().sum()),
            "unique_count": int(metadata[column].nunique(dropna=True)),
        }
        for column in metadata.columns
    ]

    return {
        "archive_summary": {
            "cif_file_count": len(extraction.files_by_name),
            "sample_filenames": sorted(extraction.files_by_name)[:sample_files],
        },
        "metadata_summary": {
            "rows": int(metadata.shape[0]),
            "columns": int(metadata.shape[1]),
            "column_summaries": columns,
            "preview_rows": preview_rows,
        },
        "alignment_summary": {
            "matched_count": int(alignment.matched_metadata.shape[0]),
            "missing_in_archive": alignment.missing_in_archive,
            "extra_in_archive": alignment.extra_in_archive,
        },
        "warnings": extraction.warnings + alignment.warnings,
    }
