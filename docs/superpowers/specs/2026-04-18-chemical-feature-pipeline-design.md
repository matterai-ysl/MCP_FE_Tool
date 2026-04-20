# Chemical Feature Pipeline Design

## Goal

将当前混合型的材料特征工程 MCP 工具重构为一个边界清晰的“化学表示到特征列”工具集，聚焦：

- 无机组成式（composition）特征化
- 有机 `SMILES` 特征化
- 基于训练集的特征选择
- 本地长期保存的特征工程 pipeline，并通过 `pipeline_id` 对测试集/预测集复用

不包含：

- 归一化、标准化、正则化
- 模型训练与推理
- LLM 自动猜列

## Final Tool Set

### 1. `summarize_dataset`

职责：
- 读取 CSV / Excel / URL 数据
- 返回列名、dtype、缺失率、唯一值数、示例值、预览行、warnings

边界：
- 不猜目标列
- 不猜 composition / smiles 列
- 不落盘

### 2. `fit_feature_pipeline`

职责：
- 在训练集上执行确定性的化学特征工程
- 保存本地长期复用的 pipeline
- 返回 `pipeline_id`

输入：
- `data_path`
- `target_column`
- `task_type`
- `composition_columns`
- `smiles_columns`
- `passthrough_columns`
- `pipeline_config`

执行内容：
- composition 特征化
- smiles 特征化
- passthrough 列拼接
- 基于训练集的特征选择
- 保存 pipeline metadata / selected feature schema / artifacts

输出：
- `pipeline_id`
- `output_file`
- `feature_columns`
- `pipeline_summary`

### 3. `transform_with_pipeline`

职责：
- 使用已有 `pipeline_id` 处理测试集或预测集
- 保证与训练集相同的特征空间和列顺序

执行内容：
- 使用 pipeline 中保存的 composition 配置生成特征
- 使用 pipeline 中保存的 smiles 配置生成特征
- 对齐到训练阶段的最终保留特征列
- 缺失列补 0，多余列丢弃并告警

输出：
- `output_file`
- `feature_columns`
- `missing_columns`
- `dropped_columns`
- `warnings`

### 4. `inspect_pipeline`

职责：
- 查看单个本地 pipeline 的摘要和配置

输出：
- `pipeline_id`
- `created_at`
- `source_data`
- `column_spec`
- `featurization_config`
- `selection_config`
- `selected_features`

### 5. `list_pipelines`

职责：
- 列出本地长期保存的 pipeline

输出：
- pipeline 摘要列表

## Local Storage Model

Pipeline 统一存储在：

`data/pipelines/<pipeline_id>/`

目录内容：

- `metadata.json`
  - pipeline 基本信息
  - column spec
  - feature config
  - selection config
  - selected features
  - feature order
  - training data fingerprint
- `training_features.csv`
  - 训练集转换后的最终特征文件
- `feature_selection_report.png`
- `feature_selection_report.html`
- `feature_selection_details.txt`
- `transforms/`
  - 使用该 pipeline 处理测试集/预测集得到的输出

## Feature Scope

### Composition

基于现有 `matminer` 能力，支持：
- `element_property`
- `stoichiometry`
- `valence_orbital`
- `element_amount`

### SMILES

第一版只支持经典、确定性的化学特征：
- RDKit descriptors
- Morgan fingerprint / ECFP
- MACCS keys（可选）

不支持：
- learned embedding
- 图神经网络表示
- LLM 表示

## Alignment Rules

训练集和预测集特征一致性由 pipeline 保证：

- 使用完全相同的 composition / smiles 特征化配置
- 最终特征列严格按训练阶段 `selected_features` 对齐
- 预测阶段缺失列补 0
- 预测阶段新增但训练中不存在的列丢弃并记录 warning

## File Boundaries

### Existing files to keep

- `src/materials_feature_engineering_mcp/feature_generator.py`
  - 继续承载 composition 特征化能力
- `src/materials_feature_engineering_mcp/feature_selector.py`
  - 继续承载 RFECV 特征选择能力
- `src/materials_feature_engineering_mcp/utils.py`
  - 继续承载路径、安全、数据加载 helper

### New files

- `src/materials_feature_engineering_mcp/smiles_featurizer.py`
  - `SMILES` 到 RDKit 描述符 / 指纹
- `src/materials_feature_engineering_mcp/pipeline_store.py`
  - pipeline 本地存储、读取、列举
- `src/materials_feature_engineering_mcp/pipeline_runner.py`
  - 训练集 fit 与预测集 transform 主流程

### File to simplify

- `src/materials_feature_engineering_mcp/mcp_tool.py`
  - 改为只暴露 5 个新 MCP tools
  - 不再承载旧的“大一统流程”

## Migration Decision

本次实现采用“新接口替换旧接口”的方向：
- MCP 暴露层以新工具集合为准
- 旧的内部类能力可保留供新 runner 复用

## Success Criteria

- MCP 实际暴露的工具只剩下 5 个新接口
- `fit_feature_pipeline` 可以生成长期保存的 `pipeline_id`
- `transform_with_pipeline` 可以基于同一个 `pipeline_id` 稳定复用
- 训练集 / 预测集特征列顺序一致
- `summarize_dataset` 足够让智能体自己判断 composition / smiles / target
