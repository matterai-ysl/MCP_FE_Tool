# OpenFE 自动特征工程功能说明

## 📋 功能概述

新增了 `auto_feature_engineering_with_openfe` MCP工具，专门用于处理高维特征数据的自动化特征工程。

## 🎯 适用场景

- **高维数据处理**：当输入特征数量达到上百维时
- **特征工程瓶颈**：传统特征工程方法效果不佳
- **自动化需求**：需要自动生成有价值的特征组合
- **快速迭代**：希望快速探索不同的特征组合

## 🔧 核心功能

### 1. 自动目标变量识别
- 默认将最后 n 列作为目标变量
- 支持单目标和多目标任务
- 自动分离特征和目标

### 2. 两阶段特征筛选

#### 第一阶段：方差筛选
- 移除低方差特征（默认阈值：0.01）
- 保留信息量大的特征
- 避免常数或近常数特征干扰

#### 第二阶段：统计显著性筛选
- 回归任务：使用F检验（f_regression）
- 分类任务：使用F检验（f_classif）
- 选择与目标变量最相关的特征
- 将特征数量控制在可管理范围

### 3. OpenFE特征生成
- 自动探索特征之间的组合关系
- 生成二元运算特征（加、减、乘、除等）
- 基于梯度提升树评估特征重要性
- 选择最有价值的新特征

### 4. 数据保存和报告
- 自动保存增强后的数据集
- 生成详细的特征工程报告
- 记录所有处理步骤和统计信息

## 📊 参数说明

```python
auto_feature_engineering_with_openfe(
    data_path: str,                    # 数据文件路径
    target_dims: int = 1,              # 目标变量维度（最后n列）
    task_type: str = "regression",     # 任务类型：regression 或 classification
    n_features_before_openfe: int = 50,# 预筛选后保留的特征数
    n_new_features: int = 10,          # OpenFE生成的新特征数量
    output_path: Optional[str] = None  # 输出路径（可选）
)
```

### 参数详解

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `data_path` | str | 必需 | CSV或Excel格式的数据文件路径 |
| `target_dims` | int | 1 | 目标变量维度，从表格末尾计算 |
| `task_type` | str | "regression" | 任务类型，影响特征筛选策略 |
| `n_features_before_openfe` | int | 50 | 进入OpenFE前的特征数量上限 |
| `n_new_features` | int | 10 | OpenFE生成的新特征数量 |
| `output_path` | str | None | 输出文件路径，不提供则自动生成 |

## 💡 使用示例

### 示例1：基本用法（回归任务）

```python
from src.materials_feature_engineering_mcp.mcp_tool import auto_feature_engineering_with_openfe

result = auto_feature_engineering_with_openfe(
    data_path="materials_data.csv",
    target_dims=1,
    task_type="regression"
)

print(f"状态: {result['状态']}")
print(f"输出文件: {result['输出文件']}")
print(f"新增特征数: {result['OpenFE特征生成']['新增特征数']}")
```

### 示例2：高维数据处理

```python
# 假设有200个特征，先筛选到30个再用OpenFE
result = auto_feature_engineering_with_openfe(
    data_path="high_dimensional_data.csv",
    target_dims=1,
    task_type="regression",
    n_features_before_openfe=30,  # 先筛选到30个特征
    n_new_features=5,              # 生成5个新特征
    output_path="outputs/enhanced.csv"
)
```

### 示例3：分类任务

```python
result = auto_feature_engineering_with_openfe(
    data_path="classification_data.csv",
    target_dims=1,
    task_type="classification",    # 分类任务
    n_features_before_openfe=40,
    n_new_features=8
)
```

## 📈 返回结果

成功执行后返回的字典包含以下信息：

```python
{
    "状态": "成功",
    "输入文件": "data.csv",
    "输出文件": "outputs/data_openfe_20231004_123456.csv",
    "原始数据形状": (1000, 105),  # (样本数, 总列数)
    "目标变量": ["target"],
    "任务类型": "regression",
    "初步筛选": {
        "原始特征数": 104,
        "筛选后特征数": 50,
        "筛选率": "48.1%"
    },
    "OpenFE特征生成": {
        "输入特征数": 50,
        "输出特征数": 58,
        "新增特征数": 8,
        "目标生成数": 10
    },
    "最终数据形状": (1000, 59),
    "特征列表": ["feature1", "feature2", ..., "autoFE_f_0", "autoFE_f_1", ..., "target"],
    "特征构造说明": {
        "描述": "每个新特征的构造方式",
        "详细报告": "outputs/data_openfe_20231004_123456_feature_descriptions.txt",
        "特征详情": {
            "autoFE_f_0": "feat_A + feat_B",
            "autoFE_f_1": "feat_C * feat_D",
            ...
        }
    }
}
```

### 🔍 特征报告（双格式）

工具会**自动生成两种格式的报告**：

#### 1. HTML可视化报告 (`*_report.html`) ⭐推荐

美观的交互式HTML报告，包含：
- 📊 **可视化统计图表**：特征数量变化、筛选进度条
- 📋 **详细特征列表**：每个新特征的构造方式
- 🎨 **响应式设计**：支持打印和移动设备查看
- 💡 **一目了然**：渐变色彩、卡片布局、易读表格

只需在浏览器中打开 HTML 文件即可查看精美报告！

#### 2. 文本报告 (`*_feature_descriptions.txt`)

纯文本格式的传统报告，详细说明每个新特征的构造方式：

```
================================================================================
OpenFE 自动特征工程报告
================================================================================

数据文件: data.csv
生成时间: 2023-10-04 12:34:56
任务类型: regression
生成特征数: 8

================================================================================
特征构造详情
================================================================================

【autoFE_f_0】
构造方式: feat_temperature + feat_pressure
--------------------------------------------------------------------------------

【autoFE_f_1】
构造方式: feat_density * feat_volume
--------------------------------------------------------------------------------

...

说明:
- 特征名称格式: autoFE_f_N (N为特征序号)
- 构造方式显示了该特征由哪些原始特征通过何种运算生成
- 常见运算: +（加）, -（减）, *（乘）, /（除）, ^（幂）等
================================================================================
```

## ⚙️ 工作流程图

```
输入数据 (100+ 特征)
    ↓
识别目标变量 (最后n列)
    ↓
特征-目标分离
    ↓
【第一阶段】方差筛选
    ↓ (移除低方差特征)
【第二阶段】统计显著性筛选
    ↓ (选择top-k特征)
筛选后特征 (50个)
    ↓
【OpenFE】特征生成
    ↓ (生成组合特征)
增强特征 (58个)
    ↓
合并目标变量
    ↓
保存结果 + 生成报告
```

## ⚡ 性能优化建议

### 1. 特征筛选策略

| 原始特征数 | 建议 n_features_before_openfe | 理由 |
|-----------|------------------------------|------|
| < 30 | 不筛选，直接用OpenFE | 特征数少，无需筛选 |
| 30-100 | 30-50 | 平衡性能和信息保留 |
| 100-300 | 40-60 | 保留主要信息，提升速度 |
| > 300 | 50-80 | 大幅筛选，显著加速 |

### 2. 新特征生成数量

| 数据规模 | 建议 n_new_features | 理由 |
|---------|---------------------|------|
| 小数据集 (< 1000样本) | 5-10 | 避免过拟合 |
| 中等数据集 (1000-10000) | 10-20 | 平衡性能和泛化 |
| 大数据集 (> 10000) | 20-50 | 可生成更多特征 |

### 3. 任务类型选择

- **回归任务**：预测连续值（如热导率、形成能）
- **分类任务**：预测类别（如材料稳定性等级）

## 🔍 常见问题

### Q1: OpenFE运行时间很长怎么办？
**A**: 减少 `n_features_before_openfe` 参数，例如从50减到30。

### Q2: 生成的特征质量如何评估？
**A**: 
1. 查看返回结果中的统计信息
2. 使用生成的数据训练模型，对比原始特征的效果
3. 分析特征重要性

### Q3: 是否支持URL数据源？
**A**: 支持，本地文件和URL都可以。

### Q4: 输出文件格式是什么？
**A**: 始终输出CSV格式，保存在 `outputs/` 目录下。

### Q5: 如何处理缺失值？
**A**: 工具会自动填充缺失值：
- 数值型特征：使用均值填充
- 分类型特征：使用众数填充

### Q6: 如何理解 OpenFE 生成的特征？
**A**: 工具会自动生成特征描述报告（**双格式**）：

1. **HTML可视化报告** ⭐推荐：
   ```bash
   # 自动生成到 outputs 目录
   outputs/xxx_report.html
   
   # 用浏览器打开即可查看精美报告
   open outputs/xxx_report.html
   ```
   
2. **文本报告**：
   ```bash
   outputs/xxx_feature_descriptions.txt
   ```

3. **MCP响应中**：包含 `特征构造说明` 字段
   ```python
   # 在返回结果中查看
   print(result["特征构造说明"]["HTML报告"])
   print(result["特征构造说明"]["特征详情"])
   ```

4. **特征格式**：
   - `autoFE_f_0`: OpenFE生成的第1个特征
   - 构造示例：`feat_A + feat_B` 表示特征A与特征B相加
   - 常见运算：`+`, `-`, `*`, `/`, `^`（幂）, `log`, `sqrt` 等

## 🚀 最佳实践

### 1. 数据准备
```python
# 确保目标变量在最后
# 正确格式：feature1, feature2, ..., target
# 错误格式：target, feature1, feature2, ...
```

### 2. 参数调优
```python
# 从保守设置开始
result = auto_feature_engineering_with_openfe(
    data_path="data.csv",
    n_features_before_openfe=30,  # 较少特征
    n_new_features=5               # 较少新特征
)

# 根据结果逐步增加
```

### 3. 迭代优化
```python
# 第一次：快速探索
result1 = auto_feature_engineering_with_openfe(
    data_path="data.csv",
    n_features_before_openfe=20,
    n_new_features=3
)

# 第二次：增加特征
result2 = auto_feature_engineering_with_openfe(
    data_path="data.csv",
    n_features_before_openfe=50,
    n_new_features=10
)
```

## 📦 依赖项

新增依赖已添加到 `pyproject.toml`：
- `openfe`：核心特征工程库
- `scikit-learn<1.4.0`：特征选择（**重要**：必须 <1.4.0 版本以兼容 OpenFE）
- `pandas`, `numpy`：数据处理（已有）

### ⚠️ 版本兼容性说明

OpenFE 0.0.12 与 scikit-learn 1.4.0+ 不兼容，原因：
- OpenFE 使用了 `mean_squared_error(..., squared=False)` 参数
- scikit-learn 1.4.0+ 移除了此参数，改用 `root_mean_squared_error()`

**解决方案**：项目已锁定 `scikit-learn<1.4.0` 版本。

## 🔗 相关资源

- [OpenFE GitHub](https://github.com/IIIS-Li-Group/OpenFE)
- [OpenFE 文档](https://openfe.readthedocs.io/)
- [特征工程最佳实践](https://scikit-learn.org/stable/modules/feature_selection.html)

---

**注意事项**：
- OpenFE需要一定的计算时间，请根据数据规模合理设置参数
- 建议在使用前先运行小规模测试
- 生成的特征可能需要进一步的特征选择和验证

