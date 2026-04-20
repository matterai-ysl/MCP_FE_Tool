# Real-World Dataset Stress Test Report

Date: 2026-04-19

## Scope

This round focused on the new chemistry pipeline toolset:

- `summarize_dataset`
- `fit_feature_pipeline`
- `transform_with_pipeline`
- `inspect_pipeline`
- `list_pipelines`

The goal was to download real public datasets and stress the pipeline with:

- real composition regression data
- real SMILES regression data
- real SMILES classification data
- RFECV enabled runs
- invalid chemical string inputs
- train/predict schema mismatch

## Downloaded Datasets

Saved under `data/external_tests/raw/`.

1. `bandgaps.csv`
   Source: `https://raw.githubusercontent.com/cburdine/materials-ml-workshop/main/MaterialsML/regression/bandgaps.csv`
   Shape: `29195 x 11`
   Main fields used: `formula`, `volume`, `density`, `formation_energy_per_atom`, `band_gap`

2. `delaney-processed.csv`
   Source: `https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv`
   Shape: `1128 x 10`
   Main fields used: `smiles`, `Molecular Weight`, `Polar Surface Area`, `measured log solubility in mols per litre`

3. `BBBP.csv`
   Source: `https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv`
   Shape: `2050 x 4`
   Main fields used: `smiles`, `p_np`

## What Was Run

### Smoke Tests

1. `bandgaps_regression`
   - composition: `formula`
   - passthrough: `volume`, `density`, `formation_energy_per_atom`
   - target: `band_gap`
   - selection: `none`
   - result: fit and transform succeeded
   - note: transform emitted large alignment warnings:
     - `Filled 49 missing feature columns with 0.`
     - `Dropped 4 columns not present in the trained pipeline.`

2. `delaney_regression`
   - smiles: `smiles`
   - passthrough: `Molecular Weight`, `Polar Surface Area`
   - target: `measured log solubility in mols per litre`
   - selection: `none`
   - result: fit and transform succeeded

3. `bbbp_classification`
   - smiles: `smiles`
   - target: `p_np`
   - selection: `none`
   - result: fit and transform succeeded

### RFECV Stress Tests

1. `bandgaps_rfecv`
   - input feature count before selection: `74`
   - RFECV config: `cv_folds=3`, `step=5`
   - result: feature selection itself completed, but report generation crashed
   - error:
     - `ValueError: x and y must have same first dimension, but have shapes (15,) and (16,)`

2. `delaney_rfecv`
   - input feature count before selection: `7`
   - RFECV config: `cv_folds=3`, `step=1`
   - result: succeeded

3. `bbbp_rfecv`
   - input feature count before selection: `5`
   - RFECV config: `cv_folds=3`, `step=1`
   - result: succeeded

### Invalid Input Tests

1. Invalid composition string
   - mutation: first `formula` changed to `NOT_A_FORMULA`
   - result: fit still succeeded
   - issue: no structured warning returned to caller

2. Invalid SMILES string
   - mutation: first `smiles` changed to `NOT_A_SMILES`
   - result: fit still succeeded
   - issue: RDKit printed parse errors to stderr, but the tool returned success without structured warning

3. Missing source column during transform
   - mutation: removed `smiles` column from prediction file
   - result: transform failed with raw exception
   - error:
     - `KeyError: 'smiles'`

## Findings

### P1. RFECV report generation breaks when `step > 1`

Evidence:

- Real failure on `bandgaps_rfecv`
- Error:
  - `ValueError: x and y must have same first dimension, but have shapes (15,) and (16,)`

Root cause:

- [feature_selector.py](/Users/songlin/Desktop/Code/MCP_FE_Tool/src/materials_feature_engineering_mcp/feature_selector.py:149) builds `n_features_range` with:
  - `range(self.min_features, len(X.columns) + 1, self.step)`
- But `RFECV.cv_results_` can return a score vector whose length does not match that manually constructed range when feature count and `step` do not align cleanly.
- The plotting code at [feature_selector.py](/Users/songlin/Desktop/Code/MCP_FE_Tool/src/materials_feature_engineering_mcp/feature_selector.py:153) and [feature_selector.py](/Users/songlin/Desktop/Code/MCP_FE_Tool/src/materials_feature_engineering_mcp/feature_selector.py:154) assumes equal lengths and crashes.

Impact:

- `fit_feature_pipeline(..., selection.method='rfecv')` can fail on real datasets even after selection finishes successfully.
- This blocks pipeline persistence because the crash happens inside report generation during save.

### P1. `transform_with_pipeline` returns a raw `KeyError` when required source columns are missing

Evidence:

- Removing `smiles` from a prediction file caused:
  - `KeyError: 'smiles'`

Root cause:

- [pipeline_runner.py](/Users/songlin/Desktop/Code/MCP_FE_Tool/src/materials_feature_engineering_mcp/pipeline_runner.py:157) loads pipeline metadata and immediately calls `_build_feature_frame(...)`.
- Inside `_build_feature_frame`, SMILES columns are accessed directly with `df[column]` at [pipeline_runner.py](/Users/songlin/Desktop/Code/MCP_FE_Tool/src/materials_feature_engineering_mcp/pipeline_runner.py:325).
- Unlike fit-time column normalization, transform-time execution does not validate that required source columns are still present before dereferencing them.

Impact:

- Prediction-time failures are not user-friendly.
- API callers get a low-level pandas exception instead of a structured validation message like `smiles column not found in dataset: smiles`.

### P2. Invalid chemical strings are silently degraded instead of being surfaced as warnings

Evidence:

- `NOT_A_FORMULA` in the composition dataset still allowed the fit to succeed.
- `NOT_A_SMILES` in the SMILES dataset still allowed the fit to succeed.
- RDKit emitted parse errors to stderr, but the API result did not contain structured invalid-row warnings.

Root cause:

- Composition path:
  - [feature_generator.py](/Users/songlin/Desktop/Code/MCP_FE_Tool/src/materials_feature_engineering_mcp/feature_generator.py:173) converts bad formulas to `None`
  - [feature_generator.py](/Users/songlin/Desktop/Code/MCP_FE_Tool/src/materials_feature_engineering_mcp/feature_generator.py:207) later collapses invalid compositions to empty element dicts
  - no invalid count or invalid row list is returned
- SMILES path:
  - [smiles_featurizer.py](/Users/songlin/Desktop/Code/MCP_FE_Tool/src/materials_feature_engineering_mcp/smiles_featurizer.py:49) does compute `invalid_count` and `invalid_indices`
  - [pipeline_runner.py](/Users/songlin/Desktop/Code/MCP_FE_Tool/src/materials_feature_engineering_mcp/pipeline_runner.py:328) stores only internal summary metadata
  - [pipeline_runner.py](/Users/songlin/Desktop/Code/MCP_FE_Tool/src/materials_feature_engineering_mcp/pipeline_runner.py:139) does not expose these invalid-row warnings in the public return payload

Impact:

- Bad chemistry strings are easy to miss in automated workflows.
- Server logs become noisy, but the API result looks successful.
- Users may train on silently zeroed-out chemical features without realizing data quality degraded.

### P3. Composition feature alignment can drift heavily between train and predict splits

Evidence:

- In the real `bandgaps.csv` smoke test, transform produced:
  - `Filled 49 missing feature columns with 0.`
  - `Dropped 4 columns not present in the trained pipeline.`

Interpretation:

- This is not necessarily a code bug.
- It is an operational risk of using `element_amount`-style features on composition data where the training split does not cover the same element set as the prediction split.

Impact:

- On small or biased training splits, prediction-time feature coverage can drift substantially.
- Performance may degrade even though the pipeline behaves as designed.

## Recommended Next Fixes

1. Fix RFECV plotting/report generation so feature-count axes derive directly from `RFECV.cv_results_`, not from a manually reconstructed range.
2. Validate required input columns during `transform_with_pipeline` before calling `_build_feature_frame`.
3. Expose invalid composition / invalid SMILES counts and row indices as structured warnings in both fit and transform results.
4. Consider adding a train/predict chemistry-coverage warning when composition feature alignment drops or fills a large fraction of columns.

## Verification Commands Run

```bash
./.venv/bin/python -m unittest tests.test_regressions -v
./.venv/bin/python -m compileall src single_port_server.py test_html_report.py tests/test_regressions.py
```

Additional real-world checks were executed with ad hoc Python invocations against the downloaded public datasets above.

## Update After Fixes

Date: 2026-04-19

The following items from this report have now been fixed and rechecked:

1. `RFECV + step > 1` report generation
   - Re-ran the previous failing `bandgaps_rfecv` scenario.
   - Result: pipeline fit completed, PNG/HTML/detail reports were generated successfully.

2. Missing required source column at transform time
   - Behavior now returns a structured `ValueError` instead of raw pandas `KeyError`.

3. Invalid composition / SMILES visibility
   - `fit_feature_pipeline` now returns structured warnings for invalid chemistry strings.
   - RDKit parse spam is suppressed from stderr in normal operation.

The remaining train/predict feature drift note for composition-derived element columns is still a modeling/data coverage concern rather than a code correctness bug.

## CIF Pipeline Smoke Test

Date: 2026-04-20

Added a dedicated CIF pipeline tool group and verified:

- `summarize_cif_archive`
- `fit_cif_pipeline`
- `transform_with_cif_pipeline`
- `inspect_cif_pipeline`
- `list_cif_pipelines`

Smoke test used synthetic NaCl/KCl CIF files generated by `pymatgen`, zipped into an archive, and aligned with metadata via `cif_filename`.

Result:

- matched rows: `2`
- feature count: `18`
- transform feature count: `18`
- fit warnings: none
- transform warnings: none

## OBELiX Real CIF Validation

Date: 2026-04-20

Downloaded OBELiX from the official repository:

- Source: `https://github.com/NRC-Mila/OBELiX`
- Train CSV: `https://raw.githubusercontent.com/NRC-Mila/OBELiX/main/data/downloads/train.csv`
- Train CIF archive: `https://raw.githubusercontent.com/NRC-Mila/OBELiX/main/data/downloads/train_cifs.zip`
- Test CSV: `https://raw.githubusercontent.com/NRC-Mila/OBELiX/main/data/downloads/test.csv`
- Test CIF archive: `https://raw.githubusercontent.com/NRC-Mila/OBELiX/main/data/downloads/test_cifs.zip`

Local files:

- `data/external_tests/cif_obelix/train.csv`
- `data/external_tests/cif_obelix/train_cifs.zip`
- `data/external_tests/cif_obelix/test.csv`
- `data/external_tests/cif_obelix/test_cifs.zip`
- `data/external_tests/cif_obelix/train_metadata_for_cif_pipeline.csv`
- `data/external_tests/cif_obelix/test_metadata_for_cif_pipeline.csv`

Metadata preparation:

- Added `cif_filename = ID + ".cif"` for filename alignment.
- Added `ionic_conductivity_numeric` by parsing values such as `<1E-10`.
- Added `target_log10_ionic_conductivity` for regression validation.

Input summaries:

- Train CSV rows: `478`
- Train CIF files: `254`
- Train matched rows: `254`
- Train warning: `224 metadata rows reference CIF files missing from archive.`
- Test CSV rows: `121`
- Test CIF files: `67`
- Test matched rows: `67`
- Test warning: `54 metadata rows reference CIF files missing from archive.`

Full fit/transform validation:

- Pipeline ID: `05a3f2cd-d7dd-4b4c-8210-9a5bc7890e98`
- Structure features: `basic`, `symmetry`, `density`, `complexity`
- Composition features: `element_property`, `stoichiometry`, `valence_orbital`, `element_amount`
- Selection: `none`
- Train matched rows: `254`
- Generated feature count: `217`
- Transform rows: `67`
- Transform feature count: `217`
- Transform filled missing train-schema columns: `16`
- Transform dropped extra test-only columns: `1`

RFECV validation on a real OBELiX subset:

- Pipeline ID: `1f3628b6-9817-4486-8e99-1d571c6144b0`
- Subset rows: `45`
- Generated feature count before selection: `47`
- Selected feature count: `12`
- RFECV config: `cv_folds=3`, `min_features=2`, `step=5`
- Artifacts generated: training CSV, PNG report, HTML report, details TXT
- Transform rows: `67`
- Transform feature count: `12`
- Transform filled missing train-schema columns: `1`
- Transform dropped extra test-only columns: `40`

### OBELiX Finding Fixed

Issue:

- Full real CIF validation initially triggered repeated multiprocessing failures during matminer composition feature generation.
- Error pattern:
  - `FileNotFoundError: [Errno 2] No such file or directory: '/Users/songlin/Desktop/Code/MCP_FE_Tool/<stdin>'`

Root cause:

- `MaterialsFeatureGenerator.generate_composition_features` used matminer composition featurizers with their default `n_jobs=10`.
- On macOS spawn mode, stdin-driven scripts and MCP-style process entrypoints can cause worker processes to try to reload `<stdin>`, which does not exist as a file.

Fix:

- Set matminer composition featurizers to single-process execution with `set_n_jobs(1)`.
- Call `featurize_dataframe(..., ignore_errors=True, pbar=False)` to keep service execution stable and quiet.
- Added regression coverage in `test_composition_featurizers_run_without_multiprocessing`.

Residual notes:

- Many OBELiX structures are disordered, so matminer `DensityFeatures` reports `Disordered structure support not built yet.` for affected rows. The pipeline handles this by warning and filling those failed feature values with `0.0`.
- Element-amount features naturally differ between train and test when the element coverage differs. The persistent pipeline schema correctly fills missing trained columns with `0` and drops test-only columns.
