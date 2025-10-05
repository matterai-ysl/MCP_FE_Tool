# 材料科学机器学习特征工程MCP工具

这是一个专为材料科学领域设计的机器学习特征工程MCP（Model Context Protocol）工具，提供数据探索、分析和预处理功能。

## 功能特性

### 🔍 数据探索工具
- **数据摘要**：基本统计信息、数据类型分析、内存使用情况
- **缺失值分析**：缺失模式识别、相关性分析、智能处理策略推荐
- **分布检查**：正态性检验、偏度分析、异常值检测
- **目标变量分析**：回归和分类任务的专门分析和预处理建议

### 🛠️ 智能预处理
- 基于数据特征的自动预处理策略选择
- 缺失值智能填充（均值、中位数、模式、KNN等）
- 数值特征标准化（Standard、MinMax、Robust Scaler）
- 分类特征编码建议

### 🧬 智能特征生成（新增）
- **LLM智能列分析**：使用大语言模型自动识别化学组成列
- **matminer特征生成**：基于化学组成自动生成材料特征
- **化学计量特征**：元素比例、计量统计等特征
- **元素属性特征**：原子质量、电负性等物理特征
- **自动特征工程**：端到端的特征生成和数据增强

### 🚀 OpenFE自动特征工程（新增）
- **高维数据处理**：自动筛选和处理上百维度的特征数据
- **智能特征组合**：使用OpenFE自动生成有价值的特征组合
- **两阶段筛选**：基于方差和统计显著性的特征预筛选
- **任务自适应**：支持回归和分类任务的特征生成
- **灵活配置**：可自定义筛选强度和生成特征数量
- **📊 HTML可视化报告**：自动生成精美的HTML报告，包含特征统计和构造详情

### 🎯 RFE-CV特征选择（新增）
- **自动特征选择**：使用递归特征消除（RFE）+ 交叉验证自动选择最佳特征组合
- **最优特征数**：自动确定最优的特征数量，无需手动尝试
- **稳定可靠**：通过交叉验证确保选择结果的稳定性和泛化能力
- **特征排名**：提供每个特征的重要性排名和贡献度分析
- **双任务支持**：支持回归和分类任务，自动选择合适的评分指标
- **📊 可视化报告**：生成包含6个分析图表的完整可视化报告

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

工具提供7个主要的MCP函数：

#### `explore_materials_data`
完整的数据分析和预处理工具
```python
result = explore_materials_data(
    data_path="materials_data.csv",
    task_type="regression",  # 或 "classification"
    target_dims=1,           # 目标变量维度
    output_path="processed_data.csv"  # 可选
)
```

#### `quick_data_summary`
快速数据概览
```python
summary = quick_data_summary("data.csv")
```

#### `analyze_missing_values`
专门的缺失值分析
```python
missing_analysis = analyze_missing_values(
    data_path="data.csv",
    task_type="regression",
    target_dims=1
)
```

#### `analyze_target_variables`
目标变量专门分析
```python
target_analysis = analyze_target_variables(
    data_path="data.csv",
    task_type="classification",
    target_dims=1
)
```

#### `preprocess_data_only`
仅执行数据预处理
```python
result = preprocess_data_only(
    data_path="input.csv",
    task_type="regression",
    target_dims=1,
    output_path="processed.csv"
)
```

#### `analyze_columns_for_feature_generation` （新增）
使用LLM分析表格列，识别可用于特征生成的列
```python
result = analyze_columns_for_feature_generation(
    data_path="data.csv",
    llm_model="gpt-3.5-turbo",
    api_key="your_api_key",  # 可选
    sample_rows=5
)
```

#### `generate_material_features` （新增）
基于化学组成列生成材料特征
```python
result = generate_material_features(
    data_path="data.csv",
    composition_column="chemical_formula",
    feature_types=["element_property", "stoichiometry"],
    output_path="features.csv"  # 可选
)
```

#### `auto_generate_features_with_llm` （新增）
使用LLM自动分析并生成材料特征
```python
result = auto_generate_features_with_llm(
    data_path="data.csv",
    sample_rows=5,
    target_dims=1
)
```

#### `auto_feature_engineering_with_openfe` （新增）
使用OpenFE进行自动化特征工程，适用于高维特征数据
```python
result = auto_feature_engineering_with_openfe(
    data_path="data.csv",
    target_dims=1,                        # 目标变量维度（默认最后1列）
    task_type="regression",               # 任务类型：regression 或 classification
    n_features_before_openfe=50,          # 初步筛选后保留的特征数
    n_new_features=10,                    # OpenFE生成的新特征数量
    output_path=None                      # 可选，不提供则自动生成
)
```

**使用场景**：
- 处理上百维度的高维特征数据
- 需要自动生成有价值的特征组合
- 传统特征工程效果不佳时的解决方案

**工作流程**：
1. 自动识别目标变量（默认最后n列）
2. 基于方差和统计显著性进行初步特征筛选
3. 使用OpenFE生成新的组合特征
4. 保存增强后的数据集
5. **自动生成HTML可视化报告和文本报告**

**生成的报告**：
- 📊 `*_report.html` - 精美的HTML可视化报告（推荐）
- 📄 `*_feature_descriptions.txt` - 文本格式详细报告
- 📈 包含特征统计、构造方式、进度可视化等

#### `select_optimal_features` （新增）
使用递归特征消除（RFE）结合交叉验证自动选择最佳特征组合
```python
result = select_optimal_features(
    data_path="data.csv",
    target_dims=1,                       # 目标变量维度（最后n列），默认1
    task_type="regression",              # 任务类型：regression 或 classification
    cv_folds=5,                          # 交叉验证折数
    min_features=1,                      # 最少保留的特征数
    step=1                               # 每次迭代移除的特征数
)
```

**核心优势**：
- ✅ 自动确定最优特征数量
- ✅ 通过交叉验证确保结果稳定性
- ✅ 提供特征重要性排名
- ✅ 支持回归和分类任务
- ✅ 生成详细的可视化报告

**使用场景**：
- 从大量特征中筛选关键特征
- 降低模型复杂度，防止过拟合
- 提高模型可解释性
- 减少计算成本

**工作流程**：
1. 读取数据并分离特征和目标变量
2. 使用RFE-CV迭代评估特征组合
3. 自动选择交叉验证评分最高的特征集
4. 生成可视化报告和详细统计
5. 保存选中特征的数据集

**生成的输出**：
- 📊 `*_feature_selection_report.png` - 6个子图的可视化报告
  - CV评分曲线
  - 特征重要性排名
  - 特征排名分布
  - 选择结果饼图
  - 评分稳定性分析
  - 关键指标汇总
- 📄 `*_feature_selection_details.txt` - 详细文本报告
  - 选中特征列表（带重要性）
  - 拒绝特征列表（带排名）
  - 完整的CV评分历史
- 💾 `*_selected_features.csv` - 包含选中特征的数据集

**参数调优建议**：
- `cv_folds`: 小数据集用5-10折，大数据集用3-5折
- `min_features`: 建议设为原始特征数的10-20%
- `step`: 小数据集用1-2（精确），大数据集用5-10（快速）

**与其他工具配合**：
```python
# 推荐工作流
# 1. 数据探索
explore_data(data_path="raw_data.csv")

# 2. 自动特征工程
openfe_result = auto_feature_engineering_with_openfe(
    data_path="raw_data.csv",
    n_new_features=20
)

# 3. 特征选择 ⭐
selection_result = select_optimal_features(
    data_path=openfe_result['output_file'],  # 使用OpenFE的输出
    target_dims=1,                       # 目标在最后1列
    task_type="regression"
)
# 得到优化后的高质量特征集！
```

**详细使用指南**：参见 `FEATURE_SELECTION_GUIDE.md`

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