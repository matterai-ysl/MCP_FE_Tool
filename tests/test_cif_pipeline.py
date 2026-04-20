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
            self.assertIn("symmetry__spacegroup_num", features.columns)
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


class CifMcpToolTests(unittest.TestCase):
    def test_mcp_registers_cif_tools(self):
        from src.materials_feature_engineering_mcp.mcp_tool import mcp

        tool_names = set(mcp._tool_manager._tools.keys())
        self.assertIn("summarize_cif_archive", tool_names)
        self.assertIn("fit_cif_pipeline", tool_names)
        self.assertIn("transform_with_cif_pipeline", tool_names)
        self.assertIn("inspect_cif_pipeline", tool_names)
        self.assertIn("list_cif_pipelines", tool_names)


if __name__ == "__main__":
    unittest.main()
