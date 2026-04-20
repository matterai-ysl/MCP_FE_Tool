# 材料科学机器学习特征工程MCP工具

这是一个专为材料科学领域设计的机器学习特征工程 MCP（Model Context Protocol）工具，当前聚焦“化学表示到特征列”的确定性转换能力。

## 功能特性

### 🔍 数据理解
- **数据摘要**：返回列名、dtype、缺失率、唯一值数、示例值和预览行
- **Agent 友好**：给智能体提供足够上下文，自行判断 `composition`、`smiles`、`target`

### 🧬 化学特征工程
- **Composition 特征化**：基于 `matminer` 生成组成式特征
- **SMILES 特征化**：基于 RDKit 生成确定性的描述符与指纹
- **数值透传**：支持保留指定数值列拼接进最终特征空间

### 🎯 特征选择与复用
- **训练集选择**：支持基于训练集执行 RFECV 特征选择
- **长期本地 pipeline**：本地保存可复用 pipeline，并返回 `pipeline_id`
- **预测集对齐**：测试集 / 预测集按同一 `pipeline_id` 复用并严格对齐列顺序

### 🚫 当前明确不做
- 不做归一化、标准化、正则化
- 不做模型训练与推理
- 不在 MCP 工具层自动猜列或自动编排流程

### 📊 材料科学专门优化
- 适用于原子特征、能量特征、机械特征、结构特征等
- 支持热导率、形成能、带隙等材料属性分析
- 晶体结构、空间群等分类特征处理
- 材料稳定性分类任务支持
- **化学组成解析**：支持复杂化学式的自动解析和特征提取

## 安装和环境设置

### 前提条件
- Python >= 3.10
- uv 包管理器

### 安装依赖
```bash
# 在项目目录下运行
uv sync
```

## 使用方法

### 1. 作为MCP工具使用

工具当前提供两组 MCP 函数：一组处理表格里的 composition / SMILES 字符串，一组处理 `CIF zip + metadata table`。

#### `summarize_dataset`
读取数据并返回给智能体判断列语义所需的事实视图。
```python
summary = summarize_dataset(
    data_path="data.csv",
    sample_rows=5,
    example_values=5,
)
```

#### `fit_feature_pipeline`
在训练集上执行 composition / SMILES 特征化与特征选择，并长期保存本地 pipeline。
```python
result = fit_feature_pipeline(
    data_path="train.csv",
    target_column="target",
    task_type="regression",
    composition_columns=["composition"],
    smiles_columns=["smiles"],
    passthrough_columns=["temperature"],
    pipeline_config={
        "featurization": {
            "composition_feature_types": ["element_amount"],
            "smiles_feature_types": ["descriptors", "morgan"],
        },
        "selection": {"method": "rfecv"},
    },
)
pipeline_id = result["pipeline_id"]
```

#### `transform_with_pipeline`
使用已有 `pipeline_id` 对测试集或预测集执行同一套特征工程。
```python
result = transform_with_pipeline(
    pipeline_id=pipeline_id,
    data_path="predict.csv",
)
```

#### `inspect_pipeline`
查看本地持久化 pipeline 的配置、产物和最终特征列。
```python
pipeline_info = inspect_pipeline(pipeline_id)
```

#### `list_pipelines`
列出本地已有的长期保存 pipeline。
```python
pipelines = list_pipelines()
```

**推荐工作流**：
1. 先用 `summarize_dataset` 让智能体识别 `composition` / `smiles` / `target`
2. 用 `fit_feature_pipeline` 在训练集上生成并保存 `pipeline_id`
3. 用 `transform_with_pipeline` 处理测试集或预测集
4. 需要回看历史配置时用 `inspect_pipeline` / `list_pipelines`

**持久化目录**：
- `data/pipelines/<pipeline_id>/metadata.json`
- `data/pipelines/<pipeline_id>/training_features.csv`
- `data/pipelines/<pipeline_id>/transforms/*.csv`

**说明**：
- 旧的“数据探索 / 自动预处理 / OpenFE MCP 工具”不再作为对外 MCP 接口暴露
- 底层类仍保留在代码库中，供新的 pipeline runner 复用

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

predict_result = transform_with_cif_pipeline(
    pipeline_id=pipeline_id,
    structure_archive="predict_structures.zip",
    metadata_table="predict_metadata.csv",
    cif_filename_column="cif_filename",
)
```

CIF pipeline 会长期保存到：

- `data/cif_pipelines/<pipeline_id>/metadata.json`
- `data/cif_pipelines/<pipeline_id>/training_features.csv`
- `data/cif_pipelines/<pipeline_id>/transforms/*.csv`

### 2. 直接使用DataExplorer类

```python
from src.materials_feature_engineering_mcp.data_explorer import DataExplorer

# 创建探索器
explorer = DataExplorer()

# 加载数据
explorer.load_data("data.csv", "regression", 1)

# 执行各种分析
summary = explorer.data_summary()
missing_analysis = explorer.missing_value_analysis()
distribution_analysis = explorer.distribution_analysis()
target_analysis = explorer.target_analysis_and_preprocessing()

# 生成预处理数据
result = explorer.generate_processed_data("processed_data.csv")
```

### 3. 直接使用MaterialsFeatureGenerator类（新增）

```python
from src.materials_feature_engineering_mcp.feature_generator import MaterialsFeatureGenerator

# 创建特征生成器
generator = MaterialsFeatureGenerator(llm_model="gpt-3.5-turbo")

# 加载数据
generator.load_data("materials_data.csv")

# LLM分析列
analysis = generator.analyze_columns_with_llm(sample_rows=5)

# 生成化学组成特征
features = generator.generate_composition_features(
    composition_column="chemical_formula",
    feature_types=["stoichiometry", "element_property"]
)

# 创建增强数据集
enhanced_data = generator.create_enhanced_dataset(
    output_path="enhanced_materials.csv",
    selected_columns={"chemical_formula": ["stoichiometry"]}
)
```

## 数据格式要求

### 输入数据格式
- 支持CSV和Excel格式 (.csv, .xlsx, .xls)
- 表格数据，每行为一个样本，每列为一个特征
- **重要**：目标变量必须在表格的最后几列

### 数据布局示例
```
特征1, 特征2, 特征3, ..., 目标变量1, 目标变量2
1.2,   3.4,   5.6,   ..., 0.8,      1
2.1,   4.3,   6.5,   ..., 0.9,      0
...
```

### 参数说明
- `task_type`: "regression" 或 "classification"
- `target_dims`: 目标变量的维度数量（从表格末尾算起）

## 典型使用场景

### 场景1：材料热导率预测（回归）
```python
# 数据包含原子特征、结构特征等，最后一列是热导率
result = explore_materials_data(
    data_path="thermal_conductivity_data.csv",
    task_type="regression",
    target_dims=1,
    output_path="processed_thermal_data.csv"
)
```

### 场景2：材料稳定性分类
```python
# 数据包含形成能、带隙等特征，最后一列是稳定性类别
result = explore_materials_data(
    data_path="stability_data.csv",
    task_type="classification",
    target_dims=1,
    output_path="processed_stability_data.csv"
)
```

### 场景3：多目标预测
```python
# 预测多个材料属性（如热导率和电导率）
result = explore_materials_data(
    data_path="multi_property_data.csv",
    task_type="regression",
    target_dims=2,  # 最后两列是目标变量
    output_path="processed_multi_data.csv"
)
```

## 输出结果

### 分析报告包含
1. **数据摘要**：基本信息、数据类型分布、统计摘要
2. **缺失值分析**：缺失统计、模式分析、处理策略
3. **分布分析**：每个特征的统计信息、正态性检验、异常值检测
4. **目标变量分析**：任务相关的专门分析和预处理建议
5. **数据预处理**：处理后的数据文件和处理日志

### 预处理输出
- 处理后的数据文件（CSV格式）
- 详细的处理日志文件
- 包含所有预处理步骤的记录

## 示例和测试

### 运行示例

```bash
# 数据探索示例
python example_usage.py

# 特征生成示例
python example_feature_generation.py

# OpenFE特征工程示例
python example_openfe_usage.py
```

### 运行测试

```bash
# 测试完整功能
python test_preprocessing.py

# 测试特征生成
python test_feature_generation.py

# 测试基本功能
python test_data.py
```

## 技术特点

- **智能分析**：基于数据特征自动选择最佳预处理策略
- **材料科学优化**：针对材料特征和属性的专门优化
- **robust处理**：自动处理各种数据质量问题
- **详细报告**：提供完整的分析报告和处理建议
- **灵活使用**：支持MCP协议和直接Python调用

## 技术栈

- **核心**：pandas, numpy, scipy, scikit-learn
- **可视化**：matplotlib, seaborn
- **MCP框架**：fastmcp
- **统计分析**：scipy.stats
- **材料特征**：matminer, pymatgen
- **LLM集成**：litellm
- **自动特征工程**：openfe

## 扩展建议

未来可以扩展的功能：
- 特征选择工具
- 特征工程工具（多项式特征、交互特征等）
- 材料描述符计算
- 高级可视化功能
- 更多材料科学专门的预处理方法

---

🔬 专为材料科学研究设计，提供智能、高效的数据预处理解决方案！
