# CIF Feature Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a dedicated CIF feature-engineering tool group that trains and reuses local `pipeline_id`s from `zip + metadata table` inputs aligned by CIF filename.

**Architecture:** Build CIF support as a parallel pipeline beside the existing composition/SMILES pipeline. New archive helpers handle secure ZIP reading and filename alignment, a new CIF featurizer turns `pymatgen.Structure` objects into deterministic structure/symmetry/composition features, and a new CIF runner owns fit/transform/inspect/list orchestration under `data/cif_pipelines/<pipeline_id>/`.

**Tech Stack:** Python, pandas, pymatgen, matminer, scikit-learn RFECV, FastMCP, unittest.

---

## File Structure

- Create: `src/materials_feature_engineering_mcp/cif_archive.py`
  - Secure ZIP scanning/extraction.
  - Metadata table loading.
  - Filename-based CIF/metadata alignment.

- Create: `src/materials_feature_engineering_mcp/cif_featurizer.py`
  - `pymatgen.Structure.from_file` parsing.
  - Basic structure features.
  - `matminer` symmetry/density/complexity features.
  - Composition features derived from parsed structures.

- Create: `src/materials_feature_engineering_mcp/cif_pipeline_store.py`
  - CIF pipeline persistence rooted at `data/cif_pipelines`.
  - Metadata read/write, transform output saving, listing.

- Create: `src/materials_feature_engineering_mcp/cif_pipeline_runner.py`
  - Public functions:
    - `summarize_cif_archive`
    - `fit_cif_pipeline`
    - `transform_with_cif_pipeline`
    - `inspect_cif_pipeline`
    - `list_cif_pipelines`

- Modify: `src/materials_feature_engineering_mcp/mcp_tool.py`
  - Expose the 5 CIF MCP tools in addition to the existing 5 chemistry-string tools.

- Create: `tests/test_cif_pipeline.py`
  - Focused CIF unit and integration tests.

- Modify: `README.md`
  - Add CIF tool-group usage.

---

### Task 1: CIF Archive And Filename Alignment Helpers

**Files:**
- Create: `src/materials_feature_engineering_mcp/cif_archive.py`
- Test: `tests/test_cif_pipeline.py`

- [ ] **Step 1: Write failing tests for archive summary and alignment**

Add this new test file:

```python
import os
import tempfile
import unittest
import zipfile
from pathlib import Path

import pandas as pd
from pymatgen.core import Lattice, Structure
from pymatgen.io.cif import CifWriter


def write_cif(path: Path, species=("Na", "Cl")) -> None:
    structure = Structure(
        Lattice.cubic(5.64),
        list(species),
        [[0, 0, 0], [0.5, 0.5, 0.5]],
    )
    CifWriter(structure).write_file(str(path))


def build_cif_zip(zip_path: Path, filenames: list[str]) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        with zipfile.ZipFile(zip_path, "w") as archive:
            for filename in filenames:
                cif_path = tmp_path / Path(filename).name
                write_cif(cif_path)
                archive.write(cif_path, arcname=filename)


class CifArchiveTests(unittest.TestCase):
    def test_summarize_archive_aligns_metadata_by_filename(self):
        from src.materials_feature_engineering_mcp.cif_archive import (
            summarize_cif_inputs,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            zip_path = tmp_path / "structures.zip"
            metadata_path = tmp_path / "metadata.csv"
            build_cif_zip(zip_path, ["nested/a.cif", "b.cif", "extra.cif"])
            pd.DataFrame(
                {
                    "cif_filename": ["a.cif", "b.cif", "missing.cif"],
                    "target": [1.0, 2.0, 3.0],
                }
            ).to_csv(metadata_path, index=False)

            summary = summarize_cif_inputs(
                structure_archive=str(zip_path),
                metadata_table=str(metadata_path),
                cif_filename_column="cif_filename",
                sample_files=10,
                sample_rows=3,
            )

            self.assertEqual(summary["archive_summary"]["cif_file_count"], 3)
            self.assertEqual(summary["metadata_summary"]["rows"], 3)
            self.assertEqual(summary["alignment_summary"]["matched_count"], 2)
            self.assertEqual(summary["alignment_summary"]["missing_in_archive"], ["missing.cif"])
            self.assertEqual(summary["alignment_summary"]["extra_in_archive"], ["extra.cif"])
            self.assertTrue(summary["warnings"])

    def test_extract_archive_rejects_duplicate_cif_basenames(self):
        from src.materials_feature_engineering_mcp.cif_archive import extract_cif_archive

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            zip_path = tmp_path / "dup.zip"
            build_cif_zip(zip_path, ["one/a.cif", "two/a.cif"])

            with self.assertRaisesRegex(ValueError, "Duplicate CIF filename"):
                extract_cif_archive(str(zip_path), tmp_path / "extract")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
./.venv/bin/python -m unittest tests.test_cif_pipeline.CifArchiveTests -v
```

Expected:

- Fails with `ModuleNotFoundError: No module named 'src.materials_feature_engineering_mcp.cif_archive'`.

- [ ] **Step 3: Implement `cif_archive.py`**

Create `src/materials_feature_engineering_mcp/cif_archive.py`:

```python
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
```

- [ ] **Step 4: Run tests and verify they pass**

Run:

```bash
./.venv/bin/python -m unittest tests.test_cif_pipeline.CifArchiveTests -v
```

Expected:

- `Ran 2 tests ... OK`

- [ ] **Step 5: Commit**

```bash
git add src/materials_feature_engineering_mcp/cif_archive.py tests/test_cif_pipeline.py
git commit -m "feat: add cif archive alignment helpers"
```

---

### Task 2: CIF Featurizer

**Files:**
- Create: `src/materials_feature_engineering_mcp/cif_featurizer.py`
- Modify: `tests/test_cif_pipeline.py`

- [ ] **Step 1: Add failing tests for CIF featurization**

Append to `tests/test_cif_pipeline.py`:

```python
class CifFeaturizerTests(unittest.TestCase):
    def test_featurizer_generates_structure_and_composition_features(self):
        from src.materials_feature_engineering_mcp.cif_featurizer import CifFeaturizer

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            cif_path = tmp_path / "nacl.cif"
            write_cif(cif_path)

            featurizer = CifFeaturizer(
                structure_feature_types=["basic", "symmetry"],
                composition_feature_types=["element_amount"],
            )
            features, summary, warnings = featurizer.featurize_files({"nacl.cif": str(cif_path)})

            self.assertEqual(features.shape[0], 1)
            self.assertIn("structure__num_sites", features.columns)
            self.assertIn("structure__volume", features.columns)
            self.assertIn("symmetry__spacegroup_number", features.columns)
            self.assertTrue(any(column.startswith("composition__") for column in features.columns))
            self.assertEqual(summary["parsed_count"], 1)
            self.assertEqual(warnings, [])

    def test_featurizer_surfaces_invalid_cif_warning(self):
        from src.materials_feature_engineering_mcp.cif_featurizer import CifFeaturizer

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            bad_path = tmp_path / "bad.cif"
            bad_path.write_text("not a valid cif", encoding="utf-8")

            featurizer = CifFeaturizer(
                structure_feature_types=["basic"],
                composition_feature_types=["element_amount"],
            )
            features, summary, warnings = featurizer.featurize_files({"bad.cif": str(bad_path)})

            self.assertEqual(features.shape[0], 1)
            self.assertTrue((features.iloc[0] == 0).all())
            self.assertEqual(summary["failed_count"], 1)
            self.assertTrue(any("bad.cif" in warning for warning in warnings))
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
./.venv/bin/python -m unittest tests.test_cif_pipeline.CifFeaturizerTests -v
```

Expected:

- Fails with `ModuleNotFoundError: No module named 'src.materials_feature_engineering_mcp.cif_featurizer'`.

- [ ] **Step 3: Implement `cif_featurizer.py`**

Create `src/materials_feature_engineering_mcp/cif_featurizer.py`:

```python
"""
CIF structure featurization based on pymatgen and matminer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from matminer.featurizers.structure import DensityFeatures, GlobalSymmetryFeatures, StructuralComplexity
from pymatgen.core import Structure

from .feature_generator import MaterialsFeatureGenerator


DEFAULT_STRUCTURE_FEATURE_TYPES = ["basic", "symmetry", "density", "complexity"]
DEFAULT_CIF_COMPOSITION_FEATURE_TYPES = ["element_property", "stoichiometry", "valence_orbital", "element_amount"]


@dataclass
class CifFeaturizer:
    structure_feature_types: Sequence[str] = tuple(DEFAULT_STRUCTURE_FEATURE_TYPES)
    composition_feature_types: Sequence[str] = tuple(DEFAULT_CIF_COMPOSITION_FEATURE_TYPES)

    def __post_init__(self) -> None:
        supported_structure_types = {"basic", "symmetry", "density", "complexity"}
        unsupported = sorted(set(self.structure_feature_types) - supported_structure_types)
        if unsupported:
            raise ValueError(f"Unsupported CIF structure feature types: {unsupported}")

    def featurize_files(
        self,
        files_by_name: Dict[str, str],
        ordered_filenames: List[str] | None = None,
    ) -> Tuple[pd.DataFrame, Dict[str, Any], List[str]]:
        filenames = ordered_filenames or sorted(files_by_name)
        structures: List[Structure | None] = []
        warnings: List[str] = []

        for filename in filenames:
            try:
                structures.append(Structure.from_file(files_by_name[filename]))
            except Exception as exc:
                structures.append(None)
                warnings.append(f"Failed to parse CIF file '{filename}': {exc}")

        frames: List[pd.DataFrame] = []
        if "basic" in self.structure_feature_types:
            frames.append(self._basic_features(structures))
        if "symmetry" in self.structure_feature_types:
            frames.append(self._matminer_features("symmetry", GlobalSymmetryFeatures(), structures, warnings))
        if "density" in self.structure_feature_types:
            frames.append(self._matminer_features("density", DensityFeatures(), structures, warnings))
        if "complexity" in self.structure_feature_types:
            frames.append(self._matminer_features("complexity", StructuralComplexity(), structures, warnings))
        if self.composition_feature_types:
            frames.append(self._composition_features(structures, filenames, warnings))

        features = pd.concat(frames, axis=1) if frames else pd.DataFrame(index=range(len(filenames)))
        features.index = filenames
        features = features.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)

        summary = {
            "input_count": len(filenames),
            "parsed_count": sum(structure is not None for structure in structures),
            "failed_count": sum(structure is None for structure in structures),
            "structure_feature_types": list(self.structure_feature_types),
            "composition_feature_types": list(self.composition_feature_types),
            "feature_count": int(features.shape[1]),
        }
        return features, summary, warnings

    def _basic_features(self, structures: List[Structure | None]) -> pd.DataFrame:
        rows: List[Dict[str, float]] = []
        for structure in structures:
            if structure is None:
                rows.append({
                    "structure__num_sites": 0.0,
                    "structure__density": 0.0,
                    "structure__volume": 0.0,
                    "structure__volume_per_atom": 0.0,
                    "structure__a": 0.0,
                    "structure__b": 0.0,
                    "structure__c": 0.0,
                    "structure__alpha": 0.0,
                    "structure__beta": 0.0,
                    "structure__gamma": 0.0,
                })
                continue

            lattice = structure.lattice
            num_sites = float(len(structure))
            rows.append({
                "structure__num_sites": num_sites,
                "structure__density": float(structure.density),
                "structure__volume": float(structure.volume),
                "structure__volume_per_atom": float(structure.volume / num_sites) if num_sites else 0.0,
                "structure__a": float(lattice.a),
                "structure__b": float(lattice.b),
                "structure__c": float(lattice.c),
                "structure__alpha": float(lattice.alpha),
                "structure__beta": float(lattice.beta),
                "structure__gamma": float(lattice.gamma),
            })
        return pd.DataFrame(rows)

    def _matminer_features(
        self,
        prefix: str,
        featurizer: Any,
        structures: List[Structure | None],
        warnings: List[str],
    ) -> pd.DataFrame:
        labels = [f"{prefix}__{label}" for label in featurizer.feature_labels()]
        rows: List[List[float]] = []
        for index, structure in enumerate(structures):
            if structure is None:
                rows.append([np.nan] * len(labels))
                continue
            try:
                values = featurizer.featurize(structure)
            except Exception as exc:
                warnings.append(f"{prefix} features failed for row {index}: {exc}")
                values = [np.nan] * len(labels)
            rows.append(values)
        return pd.DataFrame(rows, columns=labels)

    def _composition_features(
        self,
        structures: List[Structure | None],
        filenames: List[str],
        warnings: List[str],
    ) -> pd.DataFrame:
        formulas = [
            structure.composition.reduced_formula if structure is not None else None
            for structure in structures
        ]
        formula_df = pd.DataFrame({"composition_formula": formulas})
        generator = MaterialsFeatureGenerator()
        generator.data = formula_df
        try:
            composition_features = generator.generate_composition_features(
                "composition_formula",
                feature_types=list(self.composition_feature_types),
            )
        except Exception as exc:
            warnings.append(f"Composition features failed for CIF-derived formulas: {exc}")
            composition_features = pd.DataFrame(index=range(len(filenames)))
        return composition_features.add_prefix("composition__")
```

- [ ] **Step 4: Run tests and verify they pass**

Run:

```bash
./.venv/bin/python -m unittest tests.test_cif_pipeline.CifFeaturizerTests -v
```

Expected:

- `Ran 2 tests ... OK`

- [ ] **Step 5: Commit**

```bash
git add src/materials_feature_engineering_mcp/cif_featurizer.py tests/test_cif_pipeline.py
git commit -m "feat: add cif featurizer"
```

---

### Task 3: CIF Pipeline Store And Runner

**Files:**
- Create: `src/materials_feature_engineering_mcp/cif_pipeline_store.py`
- Create: `src/materials_feature_engineering_mcp/cif_pipeline_runner.py`
- Modify: `tests/test_cif_pipeline.py`

- [ ] **Step 1: Add failing integration tests for fit/transform/inspect/list**

Append to `tests/test_cif_pipeline.py`:

```python
class CifPipelineRunnerTests(unittest.TestCase):
    def test_fit_and_transform_cif_pipeline_persists_schema(self):
        from src.materials_feature_engineering_mcp.cif_pipeline_runner import (
            fit_cif_pipeline,
            inspect_cif_pipeline,
            list_cif_pipelines,
            transform_with_cif_pipeline,
        )
        from src.materials_feature_engineering_mcp.cif_pipeline_store import get_cif_pipeline_dir

        with tempfile.TemporaryDirectory() as tmp_dir:
            previous_cwd = os.getcwd()
            os.chdir(tmp_dir)
            try:
                tmp_path = Path(tmp_dir)
                train_zip = tmp_path / "train.zip"
                predict_zip = tmp_path / "predict.zip"
                train_metadata = tmp_path / "train.csv"
                predict_metadata = tmp_path / "predict.csv"

                build_cif_zip(train_zip, ["a.cif", "b.cif", "extra.cif"])
                build_cif_zip(predict_zip, ["a.cif", "b.cif"])
                pd.DataFrame({"cif_filename": ["a.cif", "b.cif"], "target": [1.0, 2.0]}).to_csv(train_metadata, index=False)
                pd.DataFrame({"cif_filename": ["a.cif", "b.cif"]}).to_csv(predict_metadata, index=False)

                fit_result = fit_cif_pipeline(
                    structure_archive=str(train_zip),
                    metadata_table=str(train_metadata),
                    cif_filename_column="cif_filename",
                    target_column="target",
                    task_type="regression",
                    pipeline_config={
                        "featurization": {
                            "structure_feature_types": ["basic", "symmetry"],
                            "composition_feature_types": ["element_amount"],
                        },
                        "selection": {"method": "none"},
                    },
                )

                pipeline_id = fit_result["pipeline_id"]
                pipeline_dir = get_cif_pipeline_dir(pipeline_id)
                self.assertTrue((pipeline_dir / "metadata.json").exists())
                self.assertTrue((pipeline_dir / "training_features.csv").exists())
                self.assertGreater(len(fit_result["feature_columns"]), 0)
                self.assertTrue(any("not referenced" in warning for warning in fit_result["warnings"]))

                inspect_result = inspect_cif_pipeline(pipeline_id)
                self.assertEqual(inspect_result["pipeline_type"], "cif")
                self.assertEqual(inspect_result["cif_filename_column"], "cif_filename")

                listed = list_cif_pipelines()["pipelines"]
                self.assertTrue(any(item["pipeline_id"] == pipeline_id for item in listed))

                transform_result = transform_with_cif_pipeline(
                    pipeline_id=pipeline_id,
                    structure_archive=str(predict_zip),
                    metadata_table=str(predict_metadata),
                    cif_filename_column="cif_filename",
                )
                transform_output = next((pipeline_dir / "transforms").glob("*.csv"))
                transformed_df = pd.read_csv(transform_output)

                self.assertEqual(transform_result["feature_columns"], fit_result["feature_columns"])
                self.assertEqual(transformed_df.columns.tolist(), fit_result["feature_columns"])
            finally:
                os.chdir(previous_cwd)

    def test_fit_cif_pipeline_requires_target_column(self):
        from src.materials_feature_engineering_mcp.cif_pipeline_runner import fit_cif_pipeline

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            zip_path = tmp_path / "train.zip"
            metadata_path = tmp_path / "metadata.csv"
            build_cif_zip(zip_path, ["a.cif"])
            pd.DataFrame({"cif_filename": ["a.cif"]}).to_csv(metadata_path, index=False)

            with self.assertRaisesRegex(ValueError, "target_column not found"):
                fit_cif_pipeline(
                    structure_archive=str(zip_path),
                    metadata_table=str(metadata_path),
                    cif_filename_column="cif_filename",
                    target_column="target",
                    task_type="regression",
                    pipeline_config={"selection": {"method": "none"}},
                )
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
./.venv/bin/python -m unittest tests.test_cif_pipeline.CifPipelineRunnerTests -v
```

Expected:

- Fails with missing `cif_pipeline_runner` / `cif_pipeline_store`.

- [ ] **Step 3: Implement `cif_pipeline_store.py`**

Create `src/materials_feature_engineering_mcp/cif_pipeline_store.py`:

```python
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
```

- [ ] **Step 4: Implement `cif_pipeline_runner.py`**

Create `src/materials_feature_engineering_mcp/cif_pipeline_runner.py`:

```python
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

from .cif_archive import align_cif_metadata, extract_cif_archive, load_cif_metadata as load_cif_metadata_table, summarize_cif_inputs
from .cif_featurizer import DEFAULT_CIF_COMPOSITION_FEATURE_TYPES, DEFAULT_STRUCTURE_FEATURE_TYPES, CifFeaturizer
from .cif_pipeline_store import (
    create_cif_pipeline_dir,
    list_cif_pipelines as list_saved_cif_pipelines,
    load_cif_metadata,
    save_cif_metadata,
    save_cif_transform_output,
)
from .config import get_download_url, get_static_url
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
```

- [ ] **Step 5: Run tests and verify they pass**

Run:

```bash
./.venv/bin/python -m unittest tests.test_cif_pipeline.CifPipelineRunnerTests -v
```

Expected:

- `Ran 2 tests ... OK`

- [ ] **Step 6: Commit**

```bash
git add src/materials_feature_engineering_mcp/cif_pipeline_store.py src/materials_feature_engineering_mcp/cif_pipeline_runner.py tests/test_cif_pipeline.py
git commit -m "feat: add cif pipeline runner"
```

---

### Task 4: Expose CIF Tools Through MCP

**Files:**
- Modify: `src/materials_feature_engineering_mcp/mcp_tool.py`
- Modify: `tests/test_cif_pipeline.py`
- Modify: `README.md`

- [ ] **Step 1: Add failing MCP registration test**

Append to `tests/test_cif_pipeline.py`:

```python
class CifMcpToolTests(unittest.TestCase):
    def test_mcp_registers_cif_tools(self):
        from src.materials_feature_engineering_mcp.mcp_tool import mcp

        tool_names = set(mcp._tool_manager._tools.keys())
        self.assertIn("summarize_cif_archive", tool_names)
        self.assertIn("fit_cif_pipeline", tool_names)
        self.assertIn("transform_with_cif_pipeline", tool_names)
        self.assertIn("inspect_cif_pipeline", tool_names)
        self.assertIn("list_cif_pipelines", tool_names)
```

- [ ] **Step 2: Run test and verify it fails**

Run:

```bash
./.venv/bin/python -m unittest tests.test_cif_pipeline.CifMcpToolTests -v
```

Expected:

- Fails because CIF tool names are missing.

- [ ] **Step 3: Modify `mcp_tool.py` imports**

Add these imports near the existing runner imports:

```python
from .cif_pipeline_runner import (
    fit_cif_pipeline as fit_cif_pipeline_runner,
    inspect_cif_pipeline as inspect_cif_pipeline_runner,
    list_cif_pipelines as list_cif_pipelines_runner,
    summarize_cif_archive as summarize_cif_archive_runner,
    transform_with_cif_pipeline as transform_with_cif_pipeline_runner,
)
```

- [ ] **Step 4: Add CIF MCP wrappers**

Add these functions after `list_pipelines`:

```python
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
```

- [ ] **Step 5: Update README**

Add a short CIF section:

```markdown
### CIF 专用工具组

当前还提供独立的 CIF pipeline 工具：

- `summarize_cif_archive`
- `fit_cif_pipeline`
- `transform_with_cif_pipeline`
- `inspect_cif_pipeline`
- `list_cif_pipelines`

训练输入是 `zip + metadata table`。`zip` 内放 `.cif` 文件，metadata 表用 `cif_filename` 列和文件名精确对齐。

```python
result = fit_cif_pipeline(
    structure_archive="structures.zip",
    metadata_table="labels.csv",
    cif_filename_column="cif_filename",
    target_column="band_gap",
    task_type="regression",
    pipeline_config={
        "featurization": {
            "structure_feature_types": ["basic", "symmetry", "density", "complexity"],
            "composition_feature_types": ["element_amount"],
        },
        "selection": {"method": "rfecv"},
    },
)
pipeline_id = result["pipeline_id"]
```
```

- [ ] **Step 6: Run MCP test and verify it passes**

Run:

```bash
./.venv/bin/python -m unittest tests.test_cif_pipeline.CifMcpToolTests -v
```

Expected:

- `Ran 1 test ... OK`

- [ ] **Step 7: Commit**

```bash
git add src/materials_feature_engineering_mcp/mcp_tool.py tests/test_cif_pipeline.py README.md
git commit -m "feat: expose cif pipeline mcp tools"
```

---

### Task 5: Full Verification And Realistic Smoke Test

**Files:**
- Modify: `REAL_WORLD_DATASET_TEST_REPORT.md`

- [ ] **Step 1: Run focused CIF tests**

Run:

```bash
./.venv/bin/python -m unittest tests.test_cif_pipeline -v
```

Expected:

- All CIF tests pass.

- [ ] **Step 2: Run all regression tests**

Run:

```bash
./.venv/bin/python -m unittest tests.test_regressions tests.test_cif_pipeline -v
```

Expected:

- Existing chemistry-string tests still pass.
- New CIF tests pass.

- [ ] **Step 3: Run compile check**

Run:

```bash
./.venv/bin/python -m compileall src single_port_server.py test_html_report.py tests/test_regressions.py tests/test_cif_pipeline.py
```

Expected:

- Exit code 0.

- [ ] **Step 4: Run an ad hoc CIF smoke test**

Run:

```bash
./.venv/bin/python - <<'PY'
import os
import tempfile
import zipfile
from pathlib import Path

import pandas as pd
from pymatgen.core import Lattice, Structure
from pymatgen.io.cif import CifWriter

from src.materials_feature_engineering_mcp.cif_pipeline_runner import (
    fit_cif_pipeline,
    summarize_cif_archive,
    transform_with_cif_pipeline,
)

def write_cif(path: Path, species):
    structure = Structure(
        Lattice.cubic(5.64),
        list(species),
        [[0, 0, 0], [0.5, 0.5, 0.5]],
    )
    CifWriter(structure).write_file(str(path))

def build_zip(zip_path: Path):
    with tempfile.TemporaryDirectory() as inner:
        inner_path = Path(inner)
        a = inner_path / "a.cif"
        b = inner_path / "b.cif"
        write_cif(a, ("Na", "Cl"))
        write_cif(b, ("K", "Cl"))
        with zipfile.ZipFile(zip_path, "w") as archive:
            archive.write(a, "a.cif")
            archive.write(b, "b.cif")

with tempfile.TemporaryDirectory() as tmp:
    previous = os.getcwd()
    os.chdir(tmp)
    try:
        tmp_path = Path(tmp)
        train_zip = tmp_path / "train.zip"
        predict_zip = tmp_path / "predict.zip"
        train_csv = tmp_path / "train.csv"
        predict_csv = tmp_path / "predict.csv"
        build_zip(train_zip)
        build_zip(predict_zip)
        pd.DataFrame({"cif_filename": ["a.cif", "b.cif"], "target": [1.0, 2.0]}).to_csv(train_csv, index=False)
        pd.DataFrame({"cif_filename": ["a.cif", "b.cif"]}).to_csv(predict_csv, index=False)

        summary = summarize_cif_archive(str(train_zip), str(train_csv), "cif_filename")
        fit = fit_cif_pipeline(
            structure_archive=str(train_zip),
            metadata_table=str(train_csv),
            cif_filename_column="cif_filename",
            target_column="target",
            task_type="regression",
            pipeline_config={
                "featurization": {
                    "structure_feature_types": ["basic", "symmetry"],
                    "composition_feature_types": ["element_amount"],
                },
                "selection": {"method": "none"},
            },
        )
        transformed = transform_with_cif_pipeline(
            pipeline_id=fit["pipeline_id"],
            structure_archive=str(predict_zip),
            metadata_table=str(predict_csv),
            cif_filename_column="cif_filename",
        )
        print("matched", summary["alignment_summary"]["matched_count"])
        print("pipeline_id", fit["pipeline_id"])
        print("feature_count", len(fit["feature_columns"]))
        print("transform_feature_count", len(transformed["feature_columns"]))
    finally:
        os.chdir(previous)
PY
```

Expected:

- Prints `matched 2`.
- Prints a `pipeline_id`.
- Feature counts are greater than 0.

- [ ] **Step 5: Update test report**

Append to `REAL_WORLD_DATASET_TEST_REPORT.md`:

```markdown
## CIF Pipeline Smoke Test

Date: 2026-04-20

Added a dedicated CIF pipeline tool group and verified:

- `summarize_cif_archive`
- `fit_cif_pipeline`
- `transform_with_cif_pipeline`
- `inspect_cif_pipeline`
- `list_cif_pipelines`

Smoke test used synthetic NaCl/KCl CIF files generated by `pymatgen`, zipped into an archive, and aligned with metadata via `cif_filename`.
```

- [ ] **Step 6: Clean generated `__pycache__` changes if tracked**

Run:

```bash
git status --short
```

If tracked `__pycache__/*.pyc` files show as modified, restore them from `HEAD` without touching user source changes:

```bash
git show HEAD:src/materials_feature_engineering_mcp/__pycache__/__init__.cpython-312.pyc > src/materials_feature_engineering_mcp/__pycache__/__init__.cpython-312.pyc
git show HEAD:src/materials_feature_engineering_mcp/__pycache__/config.cpython-312.pyc > src/materials_feature_engineering_mcp/__pycache__/config.cpython-312.pyc
git show HEAD:src/materials_feature_engineering_mcp/__pycache__/data_explorer.cpython-312.pyc > src/materials_feature_engineering_mcp/__pycache__/data_explorer.cpython-312.pyc
git show HEAD:src/materials_feature_engineering_mcp/__pycache__/feature_generator.cpython-312.pyc > src/materials_feature_engineering_mcp/__pycache__/feature_generator.cpython-312.pyc
git show HEAD:src/materials_feature_engineering_mcp/__pycache__/feature_selector.cpython-312.pyc > src/materials_feature_engineering_mcp/__pycache__/feature_selector.cpython-312.pyc
git show HEAD:src/materials_feature_engineering_mcp/__pycache__/mcp_tool.cpython-312.pyc > src/materials_feature_engineering_mcp/__pycache__/mcp_tool.cpython-312.pyc
git show HEAD:src/materials_feature_engineering_mcp/__pycache__/report_generator.cpython-312.pyc > src/materials_feature_engineering_mcp/__pycache__/report_generator.cpython-312.pyc
```

- [ ] **Step 7: Commit**

```bash
git add REAL_WORLD_DATASET_TEST_REPORT.md
git commit -m "test: verify cif pipeline smoke flow"
```

---

## Self-Review Checklist

- Spec coverage:
  - Dedicated CIF tool group: Task 4.
  - ZIP + metadata alignment by filename: Task 1.
  - Structure + symmetry + composition features: Task 2.
  - Long-lived `pipeline_id` under `data/cif_pipelines`: Task 3.
  - Fit and transform support: Task 3.
  - Structured warnings and errors: Tasks 1, 2, and 3.
  - Realistic smoke test: Task 5.

- Completeness scan:
  - No incomplete markers or unspecified follow-up work is required to implement first version.

- Type consistency:
  - Public runner names match MCP wrapper names.
  - Store names use `cif_pipeline` consistently.
  - Test helper names are defined before use.
