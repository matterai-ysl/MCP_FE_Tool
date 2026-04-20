import ast
import os
import runpy
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURE_GENERATOR_PATH = PROJECT_ROOT / "src" / "materials_feature_engineering_mcp" / "feature_generator.py"


class RegressionTests(unittest.TestCase):
    def test_feature_generator_source_is_compatible_with_python_310_parser(self):
        source = FEATURE_GENERATOR_PATH.read_text(encoding="utf-8")
        ast.parse(source, filename=str(FEATURE_GENERATOR_PATH), feature_version=(3, 10))

    def test_user_output_dir_stays_under_data_root(self):
        from src.materials_feature_engineering_mcp.mcp_tool import _create_user_output_dir

        with tempfile.TemporaryDirectory() as tmp_dir:
            previous_cwd = os.getcwd()
            os.chdir(tmp_dir)
            try:
                output_dir, _ = _create_user_output_dir("../../tmp/pwn")
                resolved_output_dir = Path(output_dir).resolve()
                data_root = (Path(tmp_dir) / "data").resolve()

                self.assertTrue(str(resolved_output_dir).startswith(str(data_root)))
                self.assertNotIn("..", Path(output_dir).parts)
            finally:
                os.chdir(previous_cwd)

    def test_static_models_route_exists(self):
        from single_port_server import SinglePortMCPServer

        routes = SinglePortMCPServer()._create_file_service_routes()
        route_paths = {getattr(route, "path", None) for route in routes}

        self.assertIn("/static/models", route_paths)

    def test_distribution_analysis_skips_shapiro_for_small_samples(self):
        from src.materials_feature_engineering_mcp.data_explorer import DataExplorer

        explorer = DataExplorer()
        result = explorer._test_distributions(pd.Series([1.0, 2.0]))

        self.assertEqual(result["正态性检验"]["状态"], "skipped")

    def test_load_data_rejects_target_dims_without_feature_columns(self):
        from src.materials_feature_engineering_mcp.data_explorer import DataExplorer

        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "sample.csv"
            pd.DataFrame({"feature": [1, 2], "target": [3, 4]}).to_csv(csv_path, index=False)

            explorer = DataExplorer()
            with self.assertRaises(ValueError):
                explorer.load_data(str(csv_path), "regression", 2)

    def test_test_html_report_has_no_import_side_effects(self):
        script_path = PROJECT_ROOT / "test_html_report.py"

        with tempfile.TemporaryDirectory() as tmp_dir:
            previous_cwd = os.getcwd()
            previous_sys_path = list(sys.path)
            os.chdir(tmp_dir)
            sys.path.insert(0, str(PROJECT_ROOT))
            try:
                runpy.run_path(str(script_path), run_name="not_main")
                self.assertFalse((Path(tmp_dir) / "test_sample.csv").exists())
            finally:
                sys.path[:] = previous_sys_path
                os.chdir(previous_cwd)

    def test_feature_generator_uses_explicit_constructor_configuration(self):
        from src.materials_feature_engineering_mcp.feature_generator import MaterialsFeatureGenerator

        generator = MaterialsFeatureGenerator(api_key="demo-key", api_base="https://example.com")

        self.assertEqual(generator.api_key, "demo-key")
        self.assertEqual(generator.api_base, "https://example.com")

    def test_summarize_dataset_exposes_examples_for_agent_decision(self):
        from src.materials_feature_engineering_mcp.pipeline_runner import summarize_dataset

        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "summary.csv"
            pd.DataFrame(
                {
                    "composition": ["Fe2O3", "Al2O3", "SiO2"],
                    "smiles": ["CCO", "CCN", "CCC"],
                    "target": [1.0, 2.0, 3.0],
                }
            ).to_csv(csv_path, index=False)

            summary = summarize_dataset(str(csv_path), sample_rows=2, example_values=2)

            self.assertEqual(summary["shape"], {"rows": 3, "columns": 3})
            self.assertEqual(len(summary["preview_rows"]), 2)
            self.assertEqual(summary["columns"][0]["name"], "composition")
            self.assertEqual(summary["columns"][0]["examples"], ["Fe2O3", "Al2O3"])

    def test_fit_and_transform_pipeline_persists_feature_schema(self):
        from src.materials_feature_engineering_mcp.pipeline_runner import (
            fit_feature_pipeline,
            inspect_pipeline,
            list_pipelines,
            transform_with_pipeline,
        )
        from src.materials_feature_engineering_mcp.pipeline_store import get_pipeline_dir

        train_df = pd.DataFrame(
            {
                "composition": ["Fe2O3", "Al2O3", "SiO2", "TiO2", "MgO"],
                "smiles": ["CCO", "CCN", "CCC", "CCCl", "CCBr"],
                "temperature": [600, 700, 800, 900, 1000],
                "target": [1.2, 2.4, 3.8, 4.1, 5.0],
            }
        )
        predict_df = pd.DataFrame(
            {
                "composition": ["FeO", "CaO"],
                "smiles": ["CCO", "CCF"],
                "temperature": [650, 720],
            }
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            previous_cwd = os.getcwd()
            os.chdir(tmp_dir)
            try:
                train_path = Path(tmp_dir) / "train.csv"
                predict_path = Path(tmp_dir) / "predict.csv"
                train_df.to_csv(train_path, index=False)
                predict_df.to_csv(predict_path, index=False)

                fit_result = fit_feature_pipeline(
                    data_path=str(train_path),
                    target_column="target",
                    task_type="regression",
                    composition_columns=["composition"],
                    smiles_columns=["smiles"],
                    passthrough_columns=["temperature"],
                    pipeline_config={
                        "featurization": {
                            "composition_feature_types": ["element_amount"],
                            "smiles_feature_types": ["descriptors"],
                            "descriptor_names": ["MolWt", "TPSA"],
                        },
                        "selection": {"method": "none"},
                    },
                )

                pipeline_id = fit_result["pipeline_id"]
                pipeline_dir = get_pipeline_dir(pipeline_id)
                self.assertTrue((pipeline_dir / "metadata.json").exists())
                self.assertTrue((pipeline_dir / "training_features.csv").exists())
                self.assertGreater(len(fit_result["feature_columns"]), 0)

                inspect_result = inspect_pipeline(pipeline_id)
                self.assertEqual(inspect_result["column_spec"]["composition_columns"], ["composition"])
                self.assertEqual(inspect_result["selection_config"]["method"], "none")
                self.assertEqual(inspect_result["feature_order"], fit_result["feature_columns"])

                listed = list_pipelines()["pipelines"]
                self.assertTrue(any(item["pipeline_id"] == pipeline_id for item in listed))

                transform_result = transform_with_pipeline(pipeline_id, str(predict_path))
                transform_output = next((pipeline_dir / "transforms").glob("*.csv"))
                transformed_df = pd.read_csv(transform_output)

                self.assertEqual(transform_result["feature_columns"], fit_result["feature_columns"])
                self.assertEqual(transformed_df.columns.tolist(), fit_result["feature_columns"])
                self.assertGreater(len(transform_result["missing_columns"]), 0)
                for column in transform_result["missing_columns"]:
                    self.assertIn(column, transformed_df.columns)
                    self.assertTrue((transformed_df[column] == 0).all())
            finally:
                os.chdir(previous_cwd)

    def test_feature_selection_report_handles_rfecv_step_greater_than_one(self):
        from src.materials_feature_engineering_mcp.feature_selector import FeatureSelector

        rng = np.random.default_rng(42)
        X = pd.DataFrame(
            rng.normal(size=(30, 8)),
            columns=[f"feature_{idx}" for idx in range(8)],
        )
        y = pd.Series(X["feature_0"] * 2 - X["feature_1"], name="target")

        selector = FeatureSelector(task_type="regression", cv_folds=3, step=3, min_features=1)
        result = selector.select_features(X, y)

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "selected.csv"
            report_path = selector.generate_report(result, X, str(output_path))
            self.assertTrue(Path(report_path).exists())

    def test_transform_with_pipeline_reports_missing_required_source_columns(self):
        from src.materials_feature_engineering_mcp.pipeline_runner import (
            fit_feature_pipeline,
            transform_with_pipeline,
        )

        train_df = pd.DataFrame(
            {
                "smiles": ["CCO", "CCN", "CCC"],
                "target": [1.0, 2.0, 3.0],
            }
        )
        predict_df = pd.DataFrame({"other": [1, 2]})

        with tempfile.TemporaryDirectory() as tmp_dir:
            previous_cwd = os.getcwd()
            os.chdir(tmp_dir)
            try:
                train_path = Path(tmp_dir) / "train.csv"
                predict_path = Path(tmp_dir) / "predict.csv"
                train_df.to_csv(train_path, index=False)
                predict_df.to_csv(predict_path, index=False)

                fit_result = fit_feature_pipeline(
                    data_path=str(train_path),
                    target_column="target",
                    task_type="regression",
                    smiles_columns=["smiles"],
                    pipeline_config={
                        "featurization": {
                            "smiles_feature_types": ["descriptors"],
                            "descriptor_names": ["MolWt", "TPSA"],
                        },
                        "selection": {"method": "none"},
                    },
                )

                with self.assertRaisesRegex(ValueError, "Required pipeline input column not found: smiles"):
                    transform_with_pipeline(fit_result["pipeline_id"], str(predict_path))
            finally:
                os.chdir(previous_cwd)

    def test_fit_feature_pipeline_surfaces_invalid_chemical_input_warnings(self):
        from src.materials_feature_engineering_mcp.pipeline_runner import fit_feature_pipeline

        train_df = pd.DataFrame(
            {
                "formula": ["Fe2O3", "NOT_A_FORMULA", "Al2O3"],
                "smiles": ["CCO", "NOT_A_SMILES", "CCC"],
                "target": [1.0, 2.0, 3.0],
            }
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            previous_cwd = os.getcwd()
            os.chdir(tmp_dir)
            try:
                train_path = Path(tmp_dir) / "train.csv"
                train_df.to_csv(train_path, index=False)

                result = fit_feature_pipeline(
                    data_path=str(train_path),
                    target_column="target",
                    task_type="regression",
                    composition_columns=["formula"],
                    smiles_columns=["smiles"],
                    pipeline_config={
                        "featurization": {
                            "composition_feature_types": ["element_amount"],
                            "smiles_feature_types": ["descriptors"],
                            "descriptor_names": ["MolWt", "TPSA"],
                        },
                        "selection": {"method": "none"},
                    },
                )

                warnings = "\n".join(result.get("warnings", []))
                self.assertIn("formula", warnings)
                self.assertIn("smiles", warnings)
                self.assertIn("invalid", warnings.lower())
            finally:
                os.chdir(previous_cwd)

    def test_composition_featurizers_run_without_multiprocessing(self):
        from pymatgen.core import Composition

        from src.materials_feature_engineering_mcp.feature_generator import MaterialsFeatureGenerator

        calls = []

        class FakeFeaturizer:
            def set_n_jobs(self, value):
                calls.append(("set_n_jobs", value))

            def featurize_dataframe(self, df, col_id, **kwargs):
                calls.append(("featurize_dataframe", col_id, kwargs))
                output = df.copy()
                output["fake_feature"] = output[col_id].map(
                    lambda composition: composition.num_atoms if isinstance(composition, Composition) else 0
                )
                return output

        generator = MaterialsFeatureGenerator()
        generator.data = pd.DataFrame({"formula": ["Fe2O3", "Al2O3"]})

        with patch(
            "src.materials_feature_engineering_mcp.feature_generator.ElementProperty.from_preset",
            return_value=FakeFeaturizer(),
        ):
            features = generator.generate_composition_features("formula", feature_types=["element_property"])

        self.assertEqual(features.columns.tolist(), ["fake_feature"])
        self.assertIn(("set_n_jobs", 1), calls)
        featurize_calls = [call for call in calls if call[0] == "featurize_dataframe"]
        self.assertEqual(featurize_calls[0][2]["pbar"], False)
        self.assertEqual(featurize_calls[0][2]["ignore_errors"], True)


if __name__ == "__main__":
    unittest.main()
