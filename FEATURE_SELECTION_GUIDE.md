# 特征选择工具使用指南

## 概述

特征选择工具使用**递归特征消除（Recursive Feature Elimination, RFE）**结合**交叉验证（Cross-Validation）**来自动选择最佳的特征组合。

## 核心优势

✅ **自动化** - 无需手动尝试不同的特征组合  
✅ **可靠性** - 通过交叉验证确保结果稳定性  
✅ **可解释** - 提供每个特征的重要性排名  
✅ **灵活性** - 支持回归和分类任务  
✅ **可视化** - 生成详细的图表和报告  

## 工作原理

1. **初始化**: 使用所有特征训练模型
2. **评估**: 计算每个特征的重要性
3. **消除**: 移除最不重要的特征
4. **交叉验证**: 评估新特征集的性能
5. **迭代**: 重复步骤2-4，直到达到最小特征数
6. **选择**: 选择交叉验证评分最高的特征集

## 工具参数

### 必需参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `data_path` | string | 数据文件路径（CSV格式） |
| `target_dims` | int | 目标变量维度（最后n列），默认1 |

### 可选参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `task_type` | string | "regression" | 任务类型：'regression' 或 'classification' |
| `cv_folds` | int | 5 | 交叉验证折数 |
| `min_features` | int | 1 | 最少保留的特征数 |
| `step` | int | 1 | 每次迭代移除的特征数 |

## 使用示例

### 1. 回归任务（默认）

```python
result = select_optimal_features(
    data_path="/path/to/your/data.csv",
    target_dims=1  # 目标在最后1列
)
```

### 2. 分类任务

```python
result = select_optimal_features(
    data_path="/path/to/your/data.csv",
    target_dims=1,  # 目标在最后1列
    task_type="classification"
)
```

### 3. 自定义参数

```python
result = select_optimal_features(
    data_path="/path/to/your/data.csv",
    target_dims=1,         # 目标在最后1列
    task_type="regression",
    cv_folds=10,           # 使用10折交叉验证
    min_features=5,        # 至少保留5个特征
    step=2                 # 每次移除2个特征（加快速度）
)
```

## 输出结果

工具会生成以下文件：

### 1. 数据文件 (`*_selected_features.csv`)
包含选中特征的数据集，可直接用于建模

### 2. 可视化报告 (`*_feature_selection_report.png`)
包含6个子图：
- **CV评分曲线**: 显示特征数量 vs 模型性能
- **特征重要性**: Top 20 特征的重要性排名
- **特征排名分布**: 所有特征的排名直方图
- **选择结果饼图**: 选中/拒绝特征的比例
- **评分稳定性**: 每次迭代的评分和标准差
- **信息汇总**: 关键指标概览

### 3. 详细报告 (`*_feature_selection_details.txt`)
包含：
- 选中特征列表（带重要性评分）
- 拒绝特征列表（带排名）
- 完整的CV评分历史

## 返回值

工具返回一个字典，包含：

```python
{
    'selected_features': [...],        # 选中的特征列表
    'rejected_features': [...],        # 拒绝的特征列表
    'n_selected_features': 15,         # 选中特征数量
    'n_original_features': 50,         # 原始特征数量
    'retention_rate': "30.00%",        # 特征保留率
    'best_cv_score': 0.8523,           # 最佳CV评分
    'output_file': "path/to/output.csv",
    'report_file': "path/to/report.png",
    'details_file': "path/to/details.txt"
}
```

## 使用建议

### 1. 参数调优

**cv_folds (交叉验证折数)**
- 小数据集：使用5-10折
- 大数据集：使用3-5折
- 类别不平衡：使用StratifiedKFold（自动应用于分类任务）

**min_features (最小特征数)**
- 从业务角度：考虑实际应用中能接受的最少特征数
- 从性能角度：建议设置为原始特征数的10-20%
- 防止过拟合：较少特征通常泛化能力更强

**step (步长)**
- 小步长（1-2）：精确但慢，适合小数据集（<50特征）
- 大步长（5-10）：快速但粗糙，适合大数据集（>100特征）
- 自适应：先用大步长筛选，再用小步长精调

### 2. 任务类型选择

**回归任务 (regression)**
- 目标变量是连续值：价格、温度、评分等
- 评分指标：MSE (Mean Squared Error)
- 模型：RandomForestRegressor

**分类任务 (classification)**
- 目标变量是离散类别：类别、标签等
- 评分指标：F1 Score (weighted)
- 模型：RandomForestClassifier

### 3. 数据预处理

工具会自动处理：
- ✅ 缺失值（使用均值填充）
- ✅ 非数值列（自动过滤）
- ✅ 特征标准化（内部使用StandardScaler）

建议在使用前：
- 检查异常值
- 处理类别变量（需要提前编码）
- 确保数据质量

### 4. 结果解读

**CV评分曲线**
- 寻找"拐点"：评分不再显著提升的点
- 比较最优点和拐点：权衡性能 vs 简洁性

**特征重要性**
- 关注Top特征：这些对模型贡献最大
- 业务验证：重要特征是否符合领域知识

**保留率**
- 20-40%：通常是理想范围
- <20%：可能过于激进，检查数据质量
- >60%：可能需要更强的特征工程

## 实际案例

### 案例1：材料性能预测（回归）

```python
# 问题：从100个材料特征中选择关键特征预测硬度（目标在最后1列）
result = select_optimal_features(
    data_path="materials_dataset.csv",
    target_dims=1,
    task_type="regression",
    cv_folds=5,
    min_features=10,  # 至少保留10个特征
    step=3            # 每次移除3个特征
)

# 结果：从100个特征中选出25个，保留率25%，R² = 0.91
```

### 案例2：故障分类（分类）

```python
# 问题：从传感器数据（200特征）预测设备故障类型（目标在最后1列）
result = select_optimal_features(
    data_path="sensor_data.csv",
    target_dims=1,
    task_type="classification",
    cv_folds=10,      # 数据量大，使用10折CV
    min_features=15,
    step=5            # 快速筛选
)

# 结果：选出42个特征，F1 Score = 0.87
```

## 与其他工具的配合

### 工作流建议

```
1. 数据探索
   └─> explore_data()
       了解数据分布和相关性

2. 特征工程
   └─> auto_feature_engineering_with_openfe()
       生成新特征

3. 特征选择 ⭐ 
   └─> select_optimal_features()
       选择最佳特征组合

4. 模型训练
   └─> 使用选中的特征训练最终模型
```

### 示例工作流

```python
# 步骤1：探索原始数据
explore_result = explore_data(
    data_path="raw_data.csv"
)

# 步骤2：自动特征工程
openfe_result = auto_feature_engineering_with_openfe(
    data_path="raw_data.csv",
    target_dims=1,
    n_new_features=20
)

# 步骤3：特征选择（从扩展后的特征集中选择最佳特征）
selection_result = select_optimal_features(
    data_path=openfe_result['output_file'],  # 使用OpenFE的输出
    target_dims=1,                       # 目标在最后1列
    task_type="regression",
    cv_folds=5,
    min_features=15
)

# 最终得到的是经过特征工程和优化选择的高质量特征集
```

## 性能考虑

### 计算时间估算

| 数据规模 | 特征数 | CV折数 | Step | 预估时间 |
|---------|--------|--------|------|---------|
| 1K 样本 | 50特征 | 5折 | 1 | ~2分钟 |
| 10K样本 | 100特征 | 5折 | 3 | ~10分钟 |
| 100K样本 | 200特征 | 5折 | 5 | ~1小时 |

### 加速技巧

1. **增加step**: 从1改为3-5可以显著加速
2. **减少cv_folds**: 从10折改为5折可以减半时间
3. **设置合理的min_features**: 避免评估过多特征组合
4. **使用更快的estimator**: 可以修改源码使用LinearRegression替代RandomForest

## 常见问题

### Q1: 如何处理类别变量？

A: 工具会自动过滤非数值列。请在使用前进行编码：
```python
# 方法1：One-Hot编码
df = pd.get_dummies(df, columns=['category_col'])

# 方法2：Label编码
from sklearn.preprocessing import LabelEncoder
df['category_encoded'] = LabelEncoder().fit_transform(df['category_col'])
```

### Q2: CV评分一直下降怎么办？

A: 可能原因：
- 特征质量差：进行特征工程
- 样本量太小：收集更多数据
- 模型不合适：尝试其他模型

### Q3: 选中的特征太少怎么办？

A: 
- 增加`min_features`参数
- 检查特征间的多重共线性
- 考虑特征组合而非单个特征

### Q4: 运行时间太长？

A: 
- 增大`step`参数（如改为5或10）
- 减少`cv_folds`（如改为3）
- 先用大step粗选，再用小step精选

## 技术细节

### 使用的算法
- **特征选择**: RFECV (Recursive Feature Elimination with Cross-Validation)
- **基础模型**: RandomForestRegressor / RandomForestClassifier
- **交叉验证**: KFold / StratifiedKFold
- **特征标准化**: StandardScaler

### 评分指标
- **回归**: Negative MSE (越接近0越好)
- **分类**: Weighted F1 Score (0-1之间，越大越好)

## 更新日志

### Version 1.0.0 (2025-10-04)
- ✨ 初始版本发布
- ✅ 支持RFE-CV特征选择
- ✅ 支持回归和分类任务
- ✅ 生成可视化报告
- ✅ 集成到MCP工具链

## 反馈与支持

如遇问题或有改进建议，请查看项目README或联系开发团队。

