# 特征选择工具快速开始

## 5分钟快速上手

### 步骤1: 准备数据

确保你的数据是CSV格式，并且：
- ✅ 有一个明确的目标变量列
- ✅ 其他列是数值型特征
- ✅ 数据已经过基本清洗

```csv
# 示例数据格式 (materials_data.csv)
feature_1,feature_2,feature_3,...,feature_50,target
0.123,4.56,7.89,...,1.23,42.5
0.456,7.89,1.23,...,4.56,38.2
...
```

### 步骤2: 使用MCP工具调用

通过Claude Desktop或其他MCP客户端调用：

```python
# 最简单的用法（回归任务，目标在最后1列）
result = select_optimal_features(
    data_path="/path/to/materials_data.csv",
    target_dims=1
)

# 或者分类任务
result = select_optimal_features(
    data_path="/path/to/materials_data.csv",
    target_dims=1,
    task_type="classification"
)
```

### 步骤3: 查看结果

工具会输出：

```
================================================================================
特征选择工具 (RFE-CV)
================================================================================

开始特征选择 (RFE-CV)...
任务类型: regression
原始特征数: 50
样本数: 1000
交叉验证折数: 5

执行递归特征消除 + 交叉验证...
Fitting estimator with 50 features.
Fitting estimator with 49 features.
...

✓ 特征选择完成！
  最优特征数: 15
  选择的特征数: 15
  拒绝的特征数: 35
  最佳CV评分: 0.8523

================================================================================
✓ 特征选择完成！
================================================================================

输出文件:
  - 数据: /path/to/materials_data_selected_features.csv
  - 报告: /path/to/materials_data_feature_selection_report.png
  - 详情: /path/to/materials_data_feature_selection_details.txt

特征统计:
  - 原始特征: 50
  - 选中特征: 15
  - 保留率: 30.00%
  - 最佳评分: 0.8523
```

### 步骤4: 查看可视化报告

打开生成的PNG报告，你会看到：

1. **CV评分曲线** - 显示不同特征数量下的模型性能
2. **特征重要性** - Top 20特征的重要性排名
3. **特征排名分布** - 所有特征的排名直方图  
4. **选择结果** - 选中vs拒绝特征的比例
5. **评分稳定性** - 每次迭代的评分和标准差
6. **信息汇总** - 关键指标一览

### 步骤5: 使用选中的特征

```python
# 读取选中特征的数据集
import pandas as pd
df_selected = pd.read_csv("materials_data_selected_features.csv")

# 直接用于建模
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

X = df_selected.drop(columns=['target'])
y = df_selected['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestRegressor()
model.fit(X_train, y_train)
score = model.score(X_test, y_test)
print(f"模型R²评分: {score:.4f}")
```

## 常见用例

### 用例1: 材料性能预测（回归）

```python
# 从100个材料特征中选择关键特征预测硬度（目标在最后1列）
result = select_optimal_features(
    data_path="materials_properties.csv",
    target_dims=1,
    task_type="regression",
    cv_folds=5,
    min_features=10,  # 至少保留10个特征
    step=3            # 每次移除3个特征，加快速度
)
```

### 用例2: 材料分类（分类）

```python
# 预测材料的稳定性类别（目标在最后1列）
result = select_optimal_features(
    data_path="materials_stability.csv",
    target_dims=1,
    task_type="classification",
    cv_folds=10,      # 使用10折交叉验证
    min_features=5
)
```

### 用例3: 大规模特征数据

```python
# 处理200+特征的数据（目标在最后1列）
result = select_optimal_features(
    data_path="large_feature_set.csv",
    target_dims=1,
    task_type="regression",
    cv_folds=5,
    min_features=20,  # 保留至少20个特征
    step=10           # 大步长，快速筛选
)
```

## 完整工作流示例

```python
# 推荐的端到端特征工程流程

# 步骤1: 数据探索
explore_result = explore_data(
    data_path="raw_materials_data.csv"
)

# 步骤2: 自动特征工程（使用OpenFE）
openfe_result = auto_feature_engineering_with_openfe(
    data_path="raw_materials_data.csv",
    target_dims=1,
    n_features_before_openfe=50,  # 初步筛选到50个特征
    n_new_features=20              # 生成20个新特征
)
# 现在有 50 + 20 = 70 个特征

# 步骤3: 特征选择（使用RFE-CV） ⭐
selection_result = select_optimal_features(
    data_path=openfe_result['output_file'],  # 使用OpenFE的输出
    target_dims=1,                       # 目标在最后1列
    task_type="regression",
    cv_folds=5,
    min_features=15,
    step=2
)
# 最终得到15个最优特征

# 步骤4: 使用优化后的特征集训练模型
final_data = pd.read_csv(selection_result['output_file'])
# ... 训练你的最终模型
```

## 参数调优技巧

### cv_folds（交叉验证折数）

```python
# 小数据集（< 1000样本）
cv_folds=10

# 中等数据集（1K-10K样本）
cv_folds=5  # 推荐

# 大数据集（> 10K样本）
cv_folds=3
```

### min_features（最小特征数）

```python
# 保守策略（保留更多特征）
min_features=int(n_features * 0.3)  # 30%

# 激进策略（大幅降维）
min_features=int(n_features * 0.1)  # 10%

# 平衡策略
min_features=int(n_features * 0.2)  # 20% (推荐)
```

### step（步长）

```python
# 精确模式（特征数 < 50）
step=1  # 每次移除1个特征

# 平衡模式（特征数 50-100）
step=3  # 每次移除3个特征

# 快速模式（特征数 > 100）
step=10  # 每次移除10个特征

# 自适应策略
if n_features < 50:
    step = 1
elif n_features < 100:
    step = 3
else:
    step = max(1, n_features // 20)  # 约5%
```

## 常见问题解答

**Q: 运行时间太长怎么办？**

A: 
1. 增大`step`参数（如改为5或10）
2. 减少`cv_folds`（如改为3）
3. 先用大step粗选，再用小step精选

**Q: 选出的特征太少了？**

A: 
1. 增加`min_features`参数
2. 检查特征间是否存在多重共线性
3. 尝试先进行特征工程再选择

**Q: CV评分一直很低？**

A: 
1. 检查数据质量和预处理
2. 尝试不同的特征工程方法
3. 考虑是否选择了合适的任务类型

**Q: 如何处理类别变量？**

A: 
工具会自动过滤非数值列。使用前请先编码：
```python
# One-Hot编码
df = pd.get_dummies(df, columns=['category_col'])

# Label编码
from sklearn.preprocessing import LabelEncoder
df['category_encoded'] = LabelEncoder().fit_transform(df['category_col'])
```

## 下一步

- 📖 查看详细文档：`FEATURE_SELECTION_GUIDE.md`
- 🔍 了解RFE原理：[Scikit-learn RFE文档](https://scikit-learn.org/stable/modules/feature_selection.html#rfe)
- 🚀 结合OpenFE使用：`OPENFE_FEATURE.md`

## 性能参考

| 数据规模 | 特征数 | Step | CV折数 | 预估时间 |
|---------|--------|------|--------|---------|
| 1K样本 | 50特征 | 1 | 5折 | ~2分钟 |
| 1K样本 | 50特征 | 3 | 5折 | ~1分钟 |
| 10K样本 | 100特征 | 3 | 5折 | ~10分钟 |
| 10K样本 | 100特征 | 10 | 3折 | ~3分钟 |
| 100K样本 | 200特征 | 10 | 3折 | ~30分钟 |

*注：时间估算基于随机森林模型，实际时间可能因硬件而异*

