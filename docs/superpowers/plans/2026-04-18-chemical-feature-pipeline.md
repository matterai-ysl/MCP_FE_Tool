# Chemical Feature Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current mixed MCP surface with a chemistry-focused feature pipeline API that persists reusable local pipelines via `pipeline_id`.

**Architecture:** Add a small pipeline runtime layer on top of the existing composition generator and feature selector. Keep chemistry featurization deterministic, persist pipeline metadata under `data/pipelines/`, and expose only summary, fit, transform, inspect, and list tools from the MCP layer.

**Tech Stack:** FastMCP, pandas, matminer, RDKit, scikit-learn RFECV, JSON file storage

---

### Task 1: Add Pipeline Runtime Modules

**Files:**
- Create: `src/materials_feature_engineering_mcp/smiles_featurizer.py`
- Create: `src/materials_feature_engineering_mcp/pipeline_store.py`
- Create: `src/materials_feature_engineering_mcp/pipeline_runner.py`
- Modify: `pyproject.toml`

- [ ] Add deterministic SMILES featurization with RDKit descriptors and fingerprints.
- [ ] Add local pipeline storage helpers rooted at `data/pipelines/`.
- [ ] Add fit/transform orchestration that composes composition featurization, smiles featurization, passthrough columns, and feature selection.

### Task 2: Replace MCP Tool Surface

**Files:**
- Modify: `src/materials_feature_engineering_mcp/mcp_tool.py`

- [ ] Remove the old mixed-scope MCP tools from the public surface.
- [ ] Expose only `summarize_dataset`, `fit_feature_pipeline`, `transform_with_pipeline`, `inspect_pipeline`, and `list_pipelines`.
- [ ] Keep helper functions that are still needed for output directories and JSON-safe conversion.

### Task 3: Add Regression Coverage

**Files:**
- Modify: `tests/test_regressions.py`

- [ ] Add tests for pipeline persistence and listing.
- [ ] Add tests for transform-time feature alignment.
- [ ] Add tests for summarize output shape and examples.

### Task 4: Verify End-to-End Behavior

**Files:**
- Modify if needed: `uv.lock`

- [ ] Refresh lockfile and sync dependencies.
- [ ] Run regression tests.
- [ ] Run compile/import smoke checks for the new tool surface.
