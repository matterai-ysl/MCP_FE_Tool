# CIF Feature Pipeline Design

## Goal

为当前项目新增一组独立的 CIF 专用特征工程工具，支持：

- 用户上传一个包含大量 `.cif` 文件的 `zip`
- 用户额外上传一个 `csv/xlsx` 元数据表
- 通过 `cif_filename` 将结构文件与性能标签对齐
- 从 CIF 同时提取：
  - 结构特征
  - 对称性特征
  - 从结构派生的 composition 特征
- 在训练阶段保存长期复用的本地 `pipeline_id`
- 在测试集 / 预测集阶段复用同一个 `pipeline_id`

不包含：

- 归一化、标准化、正则化
- 模型训练与推理
- LLM 自动猜列
- 图神经网络或 learned embedding
- DScribe / SOAP / MBTR 等更重的结构描述符

## Why a Separate CIF Tool Group

本次采用独立 CIF 工具组，而不是把 CIF 参数塞进现有 chemistry-string pipeline。

原因：

- CIF 的输入形态是 `zip + metadata table`，和表格列中的 `composition` / `smiles` 明显不同
- CIF 需要文件解压、清单构建、文件名对齐、结构解析，这些步骤本身就构成独立工作流
- 独立工具组能保持现有 `summarize_dataset / fit_feature_pipeline / transform_with_pipeline` 的接口简单
- 后续如果要扩展更强结构描述符，也更容易在 CIF 工具组内演进，而不污染已有字符串工具

## Technical Basis

第一版采用：

- `pymatgen`
  - 负责读取 CIF
  - `Structure.from_file(...)`
- `matminer`
  - 负责结构与对称性特征
  - 以及从 structure 派生 composition 特征

不引入：

- `DScribe`
- `mofdscribe`

这不是否定这些库，而是为了让第一版优先复用项目中已有的 `pymatgen + matminer` 依赖与心智模型。

## Final Tool Set

### 1. `summarize_cif_archive`

职责：

- 读取 `zip`
- 构建其中 `.cif` 文件清单
- 读取 `metadata_table`
- 基于 `cif_filename_column` 做对齐预检查
- 返回用于智能体判断和后续工具调用的事实视图

输入：

- `structure_archive`
- `metadata_table`
- `cif_filename_column`

输出：

- `archive_summary`
  - `.cif` 文件总数
  - 文件名列表采样
- `metadata_summary`
  - 行数、列数、列名、dtype、预览行
- `alignment_summary`
  - matched count
  - missing in archive
  - extra files in archive
  - duplicate filenames in metadata
- `warnings`

边界：

- 不做特征提取
- 不保存 pipeline
- 不猜目标列

### 2. `fit_cif_pipeline`

职责：

- 在训练集上执行 CIF 特征工程
- 保存长期复用的本地 pipeline
- 返回 `pipeline_id`

输入：

- `structure_archive`
- `metadata_table`
- `cif_filename_column`
- `target_column`
- `task_type`
- `pipeline_config`

执行内容：

- 解压 `zip`
- 构建 `cif_filename -> local_path` 映射
- 用 metadata 表做精确 join
- 读取每个 CIF 为 `pymatgen.Structure`
- 生成结构特征
- 生成对称性特征
- 从结构派生 composition 特征
- 执行特征选择
- 保存 pipeline metadata / selected features / artifacts

输出：

- `pipeline_id`
- `output_file`
- `feature_columns`
- `pipeline_summary`
- `warnings`
- `artifacts`

### 3. `transform_with_cif_pipeline`

职责：

- 使用已有 `pipeline_id` 处理测试集或预测集
- 保证与训练集相同的特征空间和列顺序

输入：

- `pipeline_id`
- `structure_archive`
- `metadata_table`
- `cif_filename_column`

执行内容：

- 解压 `zip`
- 复用 pipeline 中保存的结构特征配置
- 复用 pipeline 中保存的 composition 特征配置
- 对齐到训练阶段最终保留特征列
- 缺失列补 0，多余列丢弃并告警

输出：

- `output_file`
- `feature_columns`
- `missing_columns`
- `dropped_columns`
- `warnings`

### 4. `inspect_cif_pipeline`

职责：

- 查看单个本地 CIF pipeline 的摘要和配置

输出：

- `pipeline_id`
- `created_at`
- `source_data`
- `cif_filename_column`
- `structure_config`
- `composition_config`
- `selection_config`
- `selected_features`
- `artifacts`

### 5. `list_cif_pipelines`

职责：

- 列出本地长期保存的 CIF pipelines

输出：

- pipeline 摘要列表

## Input Contract

### Archive

用户上传一个 `.zip`，内部主要包含 `.cif` 文件。

第一版约束：

- 允许 zip 内有子目录
- 但对齐时只使用最终文件名，例如 `foo/bar/mp-1005.cif` 对齐键为 `mp-1005.cif`
- 如果 zip 内出现重名文件，视为错误并返回 warning / failure

### Metadata Table

用户上传 `csv/xlsx`。

最少需要：

- 一个 `cif_filename` 列

训练场景还需要：

- 一个目标列，例如 `band_gap`

示例：

```csv
cif_filename,band_gap,formation_energy
mp-1005.cif,0.0,-0.5984
mp-1006367.cif,0.0,-1.9414
```

预测场景可以只有：

```csv
cif_filename
mp-1005.cif
mp-1006367.cif
```

## Alignment Rules

以 `metadata_table[cif_filename_column]` 为主键，与 zip 解压后的文件名精确匹配。

行为规则：

- metadata 中有、archive 中没有：记录 warning
- archive 中有、metadata 中没有：记录 warning
- metadata 中有重复文件名：报错
- archive 中有重复文件名：报错
- 成功对齐的样本才进入特征工程

第一版不支持：

- 模糊匹配
- 自动去扩展名再匹配
- 通过目录层级做额外路由

## Feature Scope

### A. Structure-Derived Basic Features

第一版结构基础特征包括：

- `num_sites`
- `density`
- `volume`
- `volume_per_atom`
- 晶格参数：
  - `a`
  - `b`
  - `c`
  - `alpha`
  - `beta`
  - `gamma`

### B. Symmetry Features

第一版对称性特征包括：

- `spacegroup_number`
- `crystal_system`
- `is_centrosymmetric`
- `n_symmetry_ops`

优先采用 `matminer.featurizers.structure.symmetry.GlobalSymmetryFeatures`。

### C. Structure Complexity / Packing Features

第一版加入这些低风险、确定性的结构特征：

- `DensityFeatures`
- `StructuralComplexity`

若部分结构不满足前置条件：

- 不让整批任务失败
- 将对应特征记为缺失后再统一数值化处理
- 同时返回 warning

### D. Composition Features From CIF

从每个 `Structure` 派生 composition，再复用当前 composition 特征链。

第一版支持：

- `element_property`
- `stoichiometry`
- `valence_orbital`
- `element_amount`

这一步的目标是保证：

- CIF pipeline 不只是“结构几何量”
- 还能共享现有材料 composition 表示能力

## Pipeline Storage Model

独立存储在：

`data/cif_pipelines/<pipeline_id>/`

目录内容：

- `metadata.json`
  - pipeline 基本信息
  - `cif_filename_column`
  - structure feature config
  - composition feature config
  - selection config
  - selected features
  - feature order
  - training data fingerprint
- `training_features.csv`
- `feature_selection_report.png`
- `feature_selection_report.html`
- `feature_selection_details.txt`
- `transforms/`
  - 预测 / 测试输出
- `extracted_archive/`
  - 可选，仅在需要缓存时保留

## Error Handling and Warnings

以下情况应该返回结构化 warnings，而不是沉默处理：

- CIF 解析失败
- CIF 文件缺失
- 元数据文件名未匹配
- 结构特征 precheck 失败
- 某些结构无法计算特定 featurizer
- 训练 / 预测阶段特征对齐存在较大漂移

以下情况应该直接报错：

- `cif_filename_column` 不存在
- metadata 中存在重复文件名
- archive 中存在重复文件名
- 没有任何成功匹配的样本
- 训练场景缺失目标列

## File Boundaries

### Existing files to reuse

- `src/materials_feature_engineering_mcp/feature_generator.py`
  - 继续复用 composition 特征逻辑
- `src/materials_feature_engineering_mcp/feature_selector.py`
  - 继续复用 RFECV 逻辑
- `src/materials_feature_engineering_mcp/pipeline_store.py`
  - 可抽象复用部分持久化逻辑
- `src/materials_feature_engineering_mcp/utils.py`
  - 路径、安全、解压、数据加载 helper

### New files

- `src/materials_feature_engineering_mcp/cif_archive.py`
  - 处理 zip 解压、CIF 文件清单、文件名对齐
- `src/materials_feature_engineering_mcp/cif_featurizer.py`
  - 读取 CIF，生成 structure + symmetry + composition features
- `src/materials_feature_engineering_mcp/cif_pipeline_runner.py`
  - `summarize_cif_archive / fit_cif_pipeline / transform_with_cif_pipeline / inspect / list`

### Existing file to extend

- `src/materials_feature_engineering_mcp/mcp_tool.py`
  - 新增 CIF 专用 MCP tools
  - 现有 chemistry-string tool group 保持不变

## Success Criteria

- 用户可以上传 `zip + metadata table` 训练 CIF pipeline
- 用户可以通过 `pipeline_id` 对新的 `zip + metadata table` 做 transform
- CIF 文件与性能标签按文件名稳定对齐
- 结构特征和 composition 特征能同时生成
- 真实数据集上的 CIF 管道可通过回归测试与 smoke test
- 错误和 warning 是结构化返回，而不是底层库直接泄漏
