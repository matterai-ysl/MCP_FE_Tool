# OpenFE 快速开始指南

## 🚀 5分钟快速上手

### 第一步：安装依赖

```bash
# 在项目目录下运行
uv sync
```

⚠️ **重要提示**：项目会自动安装 `scikit-learn<1.4.0` 版本以兼容 OpenFE。如果遇到版本冲突，请运行：
```bash
uv pip install "scikit-learn<1.4.0"
```

### 第二步：准备数据

确保你的CSV数据格式如下：
```
feature1, feature2, feature3, ..., target
1.2,      3.4,      5.6,      ..., 0.8
2.1,      4.3,      6.5,      ..., 0.9
```

⚠️ **重要**：目标变量必须在最后一列（或最后n列）

### 第三步：运行示例

创建一个Python文件 `my_openfe_test.py`：

```python
from src.materials_feature_engineering_mcp.mcp_tool import auto_feature_engineering_with_openfe

# 最简单的用法
result = auto_feature_engineering_with_openfe(
    data_path="your_data.csv",  # 替换为你的数据文件
    target_dims=1                # 最后1列是目标变量
)

# 查看结果
print(f"✅ 状态: {result['状态']}")
if result['状态'] == '成功':
    print(f"📁 输出文件: {result['输出文件']}")
    print(f"📊 原始特征数: {result['初步筛选']['原始特征数']}")
    print(f"📊 最终特征数: {result['最终数据形状'][1] - 1}")
    print(f"✨ 新增特征: {result['OpenFE特征生成']['新增特征数']}")
```

运行：
```bash
python my_openfe_test.py
```

### 第四步：查看结果

输出文件保存在 `outputs/` 目录下，文件名格式：
```
<原文件名>_openfe_<时间戳>.csv
```

## 📚 更多示例

### 高维数据（100+特征）

```python
result = auto_feature_engineering_with_openfe(
    data_path="high_dim_data.csv",
    target_dims=1,
    n_features_before_openfe=30,  # 先筛选到30个特征
    n_new_features=5               # 生成5个新特征
)
```

### 分类任务

```python
result = auto_feature_engineering_with_openfe(
    data_path="classification_data.csv",
    target_dims=1,
    task_type="classification"  # 指定为分类任务
)
```

### 多目标预测

```python
result = auto_feature_engineering_with_openfe(
    data_path="multi_target_data.csv",
    target_dims=2  # 最后2列是目标变量
)
```

## 🎯 参数速查表

| 参数 | 默认值 | 推荐值 | 说明 |
|------|--------|--------|------|
| `target_dims` | 1 | 1-3 | 目标变量列数 |
| `task_type` | "regression" | "regression" 或 "classification" | 任务类型 |
| `n_features_before_openfe` | 50 | 20-80 | 筛选后特征数 |
| `n_new_features` | 10 | 5-20 | 新增特征数 |

## 💡 常用配置

### 快速探索（速度优先）
```python
result = auto_feature_engineering_with_openfe(
    data_path="data.csv",
    n_features_before_openfe=20,
    n_new_features=3
)
```

### 标准配置（平衡）
```python
result = auto_feature_engineering_with_openfe(
    data_path="data.csv",
    n_features_before_openfe=50,
    n_new_features=10
)
```

### 深度探索（效果优先）
```python
result = auto_feature_engineering_with_openfe(
    data_path="data.csv",
    n_features_before_openfe=80,
    n_new_features=20
)
```

## 🔧 故障排除

### 问题1：运行太慢
**解决方案**：减少 `n_features_before_openfe` 参数
```python
n_features_before_openfe=20  # 从50减到20
```

### 问题2：内存不足
**解决方案**：同时减少两个参数
```python
n_features_before_openfe=15
n_new_features=3
```

### 问题3：目标变量位置不对
**解决方案**：重新整理CSV文件，将目标变量移到最后

### 问题4：openfe未安装
**解决方案**：
```bash
uv sync  # 重新同步依赖
# 或
pip install openfe
```

## 📖 完整文档

查看 `OPENFE_FEATURE.md` 获取详细文档。

## 🆘 获取帮助

遇到问题？查看：
1. `OPENFE_FEATURE.md` - 完整功能文档
2. `example_openfe_usage.py` - 详细示例代码
3. `README.md` - 项目整体说明

---

**祝你使用愉快！** 🎉

