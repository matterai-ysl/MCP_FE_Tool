# 特征选择工具接口统一更新

## 📝 更新内容

为了与其他MCP工具保持一致，特征选择工具的接口参数已统一。

### 主要变更

**之前（旧版）：**
```python
result = select_optimal_features(
    data_path="data.csv",
    target_column="target",  # ❌ 使用列名指定目标
    task_type="regression"
)
```

**现在（新版）：**
```python
result = select_optimal_features(
    data_path="data.csv",
    target_dims=1,           # ✅ 使用维度指定目标（最后n列）
    task_type="regression"
)
```

## 🎯 统一的设计原则

所有MCP工具现在遵循相同的约定：

1. **目标变量位置**: 始终在数据的**最后n列**
2. **参数命名**: 统一使用 `target_dims` 表示目标维度
3. **默认值**: `target_dims=1` (最后1列是目标)
4. **多目标支持**: `target_dims=2` 表示最后2列是目标变量

## 📚 工具对比

### 所有工具的统一接口

| 工具 | 参数 | 说明 |
|------|------|------|
| `explore_materials_data` | `target_dims=1` | 数据探索和预处理 |
| `auto_feature_engineering_with_openfe` | `target_dims=1` | OpenFE自动特征工程 |
| `select_optimal_features` | `target_dims=1` | RFE-CV特征选择 ✨ 已更新 |

## 🔄 迁移指南

如果您之前使用了旧版本的接口，请按以下方式更新：

### 单个目标变量

```python
# 旧版
select_optimal_features(
    data_path="data.csv",
    target_column="price"
)

# 新版
select_optimal_features(
    data_path="data.csv",
    target_dims=1  # 最后1列是目标
)
```

### 多个目标变量

```python
# 新版支持多目标（使用第一个进行特征选择）
select_optimal_features(
    data_path="data.csv",
    target_dims=2  # 最后2列是目标变量
)
```

## 💡 使用示例

### 示例1: 基本用法

```python
# 数据格式：
# feature_1, feature_2, ..., feature_n, target
# 
# 只需指定target_dims=1
result = select_optimal_features(
    data_path="materials_data.csv",
    target_dims=1  # 最后1列是目标
)
```

### 示例2: 与OpenFE配合

```python
# 完整工作流
# 1. OpenFE生成新特征
openfe_result = auto_feature_engineering_with_openfe(
    data_path="raw_data.csv",
    target_dims=1,
    n_new_features=20
)

# 2. RFE-CV选择最佳特征
selection_result = select_optimal_features(
    data_path=openfe_result['output_file'],
    target_dims=1,  # ✅ 接口统一
    task_type="regression"
)
```

### 示例3: 多目标场景

```python
# 数据格式：
# feature_1, feature_2, ..., target_1, target_2
#
# 使用第一个目标（target_1）进行特征选择
result = select_optimal_features(
    data_path="multi_target_data.csv",
    target_dims=2  # 最后2列是目标
)

# 工具会自动：
# 1. 使用target_1进行特征选择
# 2. 保存所有选中特征 + 所有目标变量
```

## 🔧 技术实现

### 核心改进

1. **参数变更**
   ```python
   def select_best_features(
       data_path: str,
       target_dims: int = 1,  # ✅ 新参数
       # target_column: str,   # ❌ 已移除
       ...
   )
   ```

2. **自动识别目标列**
   ```python
   # 自动从最后n列提取目标
   target_columns = df.columns[-target_dims:].tolist()
   feature_columns = df.columns[:-target_dims].tolist()
   ```

3. **保持所有目标**
   ```python
   # 结果中包含所有目标列
   result_df = pd.concat([X_selected, df[target_columns]], axis=1)
   ```

## 📄 更新的文档

以下文档已更新：

- ✅ `README.md` - 主文档
- ✅ `FEATURE_SELECTION_GUIDE.md` - 详细指南
- ✅ `QUICKSTART_FEATURE_SELECTION.md` - 快速开始
- ✅ `src/materials_feature_engineering_mcp/feature_selector.py` - 核心实现
- ✅ `src/materials_feature_engineering_mcp/mcp_tool.py` - MCP工具接口

## ⚠️ 重要提示

**使用更新后的工具前，必须重启MCP服务器！**

1. 完全退出Claude Desktop（或您使用的MCP客户端）
2. 重新启动应用
3. 新接口即可生效

## 🎉 优势

### 1. 一致性
所有工具使用相同的接口设计，学习成本更低

### 2. 灵活性
支持单目标和多目标场景

### 3. 简洁性
无需指定列名，自动从最后n列获取

### 4. 兼容性
与OpenFE等其他工具完美配合

### 5. 可维护性
统一的代码模式，更易维护

## 📊 功能对比

| 特性 | 旧版 | 新版 |
|------|------|------|
| 接口统一性 | ❌ 不同工具不同参数 | ✅ 所有工具使用target_dims |
| 多目标支持 | ❌ 不支持 | ✅ 支持多目标（使用第一个） |
| 列名依赖 | ❌ 需要知道列名 | ✅ 无需指定列名 |
| 与OpenFE配合 | ⚠️ 需要手动对齐 | ✅ 无缝对接 |
| 代码简洁度 | 一般 | ✅ 更简洁 |

## 🔍 常见问题

### Q1: 如果我的目标列不在最后怎么办？

A: 在使用工具前，先调整DataFrame的列顺序：
```python
import pandas as pd
df = pd.read_csv("data.csv")

# 移动目标列到最后
target_col = 'my_target'
cols = [c for c in df.columns if c != target_col] + [target_col]
df = df[cols]

# 保存
df.to_csv("data_reordered.csv", index=False)

# 然后使用工具
result = select_optimal_features(
    data_path="data_reordered.csv",
    target_dims=1
)
```

### Q2: 多目标时如何选择用哪个进行特征选择？

A: 目前工具会自动使用第一个目标变量。如果需要使用其他目标：
```python
# 方法1: 调整列顺序，把想用的目标放在最前面
# 方法2: 暂时只保留想用的目标列
```

### Q3: 会影响已有的工作流吗？

A: 如果您使用的是 `auto_feature_engineering_with_openfe` + `select_optimal_features` 工作流，现在会更流畅：

```python
# 现在所有工具都使用target_dims！
openfe_result = auto_feature_engineering_with_openfe(
    data_path="data.csv",
    target_dims=1  # ✅
)

selection_result = select_optimal_features(
    data_path=openfe_result['output_file'],
    target_dims=1  # ✅ 完美对接
)
```

## 📝 更新日志

### Version 1.1.0 (2025-10-05)

**变更**:
- 🔄 将 `target_column` 参数改为 `target_dims`
- ✨ 支持多目标变量场景
- 📄 更新所有相关文档
- ✅ 与其他MCP工具接口统一

**兼容性**:
- ⚠️ 不向后兼容（需要更新调用代码）
- ✅ 新老数据格式兼容

## 🚀 下一步

1. **重启MCP服务器**
2. **使用新接口**: `target_dims=1`
3. **享受统一的体验**！

---

如有任何问题或建议，请参考相关文档或提供反馈。
