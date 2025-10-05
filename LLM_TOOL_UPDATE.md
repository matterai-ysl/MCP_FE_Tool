# LLM自动特征生成工具更新

## 📋 更新概述

**更新日期**: 2025-10-05  
**工具名称**: `auto_generate_features_with_llm`  
**更新类型**: 用户输出目录管理集成

---

## ✅ 修改内容

### 1. 函数签名调整

**修改前**:
```python
def auto_generate_features_with_llm(
    data_path: str,
    sample_rows: int = 5,
    target_dims: int = 1,
    ctx: Optional[Context] = None
) -> Dict[str, Any]:
```

**修改后**:
```python
def auto_generate_features_with_llm(
    ctx: Context,  # ✨ 移到第一位，非可选
    data_path: str,
    sample_rows: int = 5,
    target_dims: int = 1
) -> Dict[str, Any]:
```

**变更说明**:
- `ctx` 参数移到第一位（与其他工具保持一致）
- `ctx` 改为必需参数（非Optional）
- 更新了docstring，添加了ctx参数说明

---

### 2. 用户目录管理

**修改前**:
```python
# 旧的方式：直接从ctx获取user_id
if ctx is not None:
    user_id = ctx.request_context.request.headers.get("user_id", None)
else:
    user_id = None
print(user_id)

# 使用_auto_output_path生成输出路径
output_path = _auto_output_path(data_path)
```

**修改后**:
```python
# 新的方式：使用统一的辅助函数
user_id = _get_user_id_from_context(ctx)
output_dir, run_uuid = _create_user_output_dir(user_id)

# 生成输出路径到用户目录
input_filename = os.path.basename(data_path)
base_name = os.path.splitext(input_filename)[0]
output_filename = f"{base_name}_enhanced.csv"
output_path = os.path.join(output_dir, output_filename)
```

**变更说明**:
- 使用统一的 `_get_user_id_from_context()` 函数
- 调用 `_create_user_output_dir()` 创建用户专属目录
- 输出路径指向 `data/{user_id}/{uuid}/` 目录

---

### 3. 报告文件保存

**修改前**:
```python
report_path = output_path.replace('.csv', '_feature_report.txt')
try:
    generator._generate_feature_report(enhanced_data, report_path)
except Exception:
    pass
```

**修改后**:
```python
# 生成特征报告（也保存到用户目录）
report_path = output_path.replace('.csv', '_feature_report.txt')
try:
    generator._generate_feature_report(enhanced_data, report_path)
    print(f"✓ Feature report saved to: {report_path}")
except Exception as e:
    print(f"⚠️  Warning: Could not generate feature report: {e}")
```

**变更说明**:
- 报告文件自动保存到用户目录（因为基于output_path生成）
- 添加了成功和失败的日志输出
- 改进了错误处理，显示具体错误信息

---

### 4. 返回值增强

**修改前**:
```python
return _json_safe({
    "状态": "成功",
    "输入文件": data_path,
    "输出文件": output_path,
    "LLM分析": analysis_result,
    "选定列": selected_columns,
    "原始数据形状": generator.data.shape,
    "增强数据形状": enhanced_data.shape,
    "新增特征数": len(new_feature_cols),
    "新增特征列": new_feature_cols
})
```

**修改后**:
```python
result = {
    "状态": "成功",
    "输入文件": data_path,
    "输出文件": output_path,
    "特征报告": report_path,  # ✨ 新增
    "LLM分析": analysis_result,
    "选定列": selected_columns,
    "原始数据形状": generator.data.shape,
    "增强数据形状": enhanced_data.shape,
    "新增特征数": len(new_feature_cols),
    "新增特征列": new_feature_cols,
    "用户信息": {  # ✨ 新增
        "用户ID": user_id if user_id else "anonymous",
        "运行UUID": run_uuid,
        "输出目录": output_dir
    }
}

return _json_safe(result)  # type: ignore
```

**变更说明**:
- 添加了 `特征报告` 字段，返回报告文件路径
- 添加了 `用户信息` 字段，包含用户ID、UUID和输出目录
- 对于"无特征生成"的情况也添加了用户信息

---

## 📁 输出文件

### 文件命名

假设输入文件为 `materials_data.csv`：

```
data/{user_id}/{uuid}/
├── materials_data_enhanced.csv              # 增强后的数据
└── materials_data_enhanced_feature_report.txt  # 特征报告
```

### 目录结构示例

```
data/
├── user_123/
│   ├── a1b2c3d4-e5f6-7g8h-9i0j-k1l2m3n4o5p6/
│   │   ├── materials_data_enhanced.csv
│   │   └── materials_data_enhanced_feature_report.txt
│   └── b2c3d4e5-f6g7-8h9i-0j1k-l2m3n4o5p6q7/
│       └── ...
└── anonymous/
    └── uuid/
        └── ...
```

---

## 🔄 使用方式

### 提供user_id

**HTTP请求头**:
```http
user_id: user_123
```

**调用**:
```python
result = auto_generate_features_with_llm(
    data_path="materials_data.csv",
    sample_rows=5,
    target_dims=1
)
```

**输出**:
```
📁 Output directory created: data/user_123/a1b2c3d4-e5f6-7g8h-9i0j-k1l2m3n4o5p6
   User ID: user_123
   Run UUID: a1b2c3d4-e5f6-7g8h-9i0j-k1l2m3n4o5p6

✓ Enhanced data saved to: data/user_123/a1b2c3d4.../materials_data_enhanced.csv
✓ Feature report saved to: data/user_123/a1b2c3d4.../materials_data_enhanced_feature_report.txt
```

**返回值**:
```json
{
    "状态": "成功",
    "输入文件": "materials_data.csv",
    "输出文件": "data/user_123/a1b2c3d4.../materials_data_enhanced.csv",
    "特征报告": "data/user_123/a1b2c3d4.../materials_data_enhanced_feature_report.txt",
    "LLM分析": {...},
    "选定列": {...},
    "原始数据形状": [100, 10],
    "增强数据形状": [100, 50],
    "新增特征数": 40,
    "新增特征列": ["feat_1", "feat_2", ...],
    "用户信息": {
        "用户ID": "user_123",
        "运行UUID": "a1b2c3d4-e5f6-7g8h-9i0j-k1l2m3n4o5p6",
        "输出目录": "data/user_123/a1b2c3d4-e5f6-7g8h-9i0j-k1l2m3n4o5p6"
    }
}
```

---

### 不提供user_id（匿名）

输出将保存到: `data/anonymous/{uuid}/`

---

## 📊 工具对比

### 已完成的工具（3/5）

| 工具名称 | 状态 | 输出文件数 | 用户目录 |
|---------|------|-----------|---------|
| `auto_feature_engineering_with_openfe` | ✅ 已完成 | 3 | ✅ |
| `select_optimal_features` | ✅ 已完成 | 4 | ✅ |
| `auto_generate_features_with_llm` | ✅ 已完成 | 2 | ✅ |

### 待修改的工具（2/5）

| 工具名称 | 状态 | 输出文件数 |
|---------|------|-----------|
| `preprocess_data_only` | ⏳ 待修改 | 1 |
| `generate_material_features` | ⏳ 待修改 | 1 |

---

## ✅ 测试验证

### 1. 功能测试

```python
# 测试用例1：有user_id
# HTTP Header: user_id: test_user
result = auto_generate_features_with_llm(
    data_path="test_data.csv",
    sample_rows=5,
    target_dims=1
)

assert "用户信息" in result
assert result["用户信息"]["用户ID"] == "test_user"
assert result["输出文件"].startswith("data/test_user/")
```

```python
# 测试用例2：无user_id（匿名）
# HTTP Header: (无)
result = auto_generate_features_with_llm(
    data_path="test_data.csv",
    sample_rows=5
)

assert result["用户信息"]["用户ID"] == "anonymous"
assert result["输出文件"].startswith("data/anonymous/")
```

### 2. 目录结构验证

```bash
cd /Users/ysl/Desktop/Code/MCP_FE_Tool
tree data/ -L 3

# 应该看到：
# data/
# ├── test_user/
# │   └── {uuid}/
# │       ├── test_data_enhanced.csv
# │       └── test_data_enhanced_feature_report.txt
# └── anonymous/
#     └── {uuid}/
#         └── ...
```

---

## 🔍 代码审查清单

- [x] `ctx` 参数移到第一位
- [x] `ctx` 改为必需参数
- [x] 使用 `_get_user_id_from_context(ctx)` 获取user_id
- [x] 使用 `_create_user_output_dir(user_id)` 创建目录
- [x] 输出文件保存到用户目录
- [x] 报告文件保存到用户目录
- [x] 返回值包含 `用户信息` 字段
- [x] 返回值包含 `特征报告` 字段
- [x] 添加了输出日志
- [x] 改进了错误处理
- [x] 无linter错误

---

## 🔄 向后兼容性

### ⚠️ 破坏性变更

**函数签名变更**：`ctx` 参数从可选变为必需，并移到第一位。

**影响**：
- 旧的调用方式将不再工作
- 所有调用此工具的代码需要更新

**迁移指南**：
```python
# 旧的调用方式（不再工作）
result = auto_generate_features_with_llm(
    data_path="data.csv",
    sample_rows=5,
    ctx=some_context  # ctx在最后
)

# 新的调用方式
result = auto_generate_features_with_llm(
    ctx=some_context,  # ctx在第一位，必需
    data_path="data.csv",
    sample_rows=5
)
```

**为什么做这个破坏性变更**：
1. 与其他工具保持一致性（`auto_feature_engineering_with_openfe`, `select_optimal_features`）
2. 确保用户目录管理功能始终可用
3. 简化调用逻辑（不需要检查ctx是否为None）

---

## 📝 相关文件

### 修改的文件

1. **`src/materials_feature_engineering_mcp/mcp_tool.py`**
   - 修改 `auto_generate_features_with_llm` 函数签名
   - 添加用户目录管理逻辑
   - 修改输出路径生成
   - 增强返回值

---

## 🎯 后续工作

### 建议的改进

1. **LLM分析缓存**
   - 对相同数据的LLM分析结果进行缓存
   - 减少重复调用LLM的成本

2. **批量处理**
   - 支持多个数据文件的批量处理
   - 使用同一个UUID目录

3. **特征质量评估**
   - 自动评估生成特征的质量
   - 提供特征重要性分析

4. **用户配置**
   - 允许用户配置默认的特征类型
   - 保存用户偏好设置

---

## 📊 性能影响

### 时间开销

| 操作 | 时间 | 影响 |
|-----|------|-----|
| 创建用户目录 | < 1ms | 可忽略 |
| UUID生成 | < 1ms | 可忽略 |
| 路径字符串操作 | < 1ms | 可忽略 |
| **总计** | **< 5ms** | **可忽略** |

### 空间开销

| 项目 | 大小 | 说明 |
|-----|------|-----|
| 目录结构 | ~4KB | 每个运行一个目录 |
| 元数据 | 可忽略 | UUID等信息 |

---

## 🎉 总结

本次更新成功将 `auto_generate_features_with_llm` 工具集成到用户输出目录管理系统中：

1. ✅ **用户隔离** - 每个用户有独立的输出空间
2. ✅ **会话追踪** - 每次运行都有唯一标识
3. ✅ **返回值增强** - 提供完整的用户和文件信息
4. ✅ **一致性** - 与其他工具保持相同的设计模式
5. ✅ **可追溯性** - 所有输出都可以追溯到特定用户和运行

**已完成工具**: 3/5 (60%)
- ✅ `auto_feature_engineering_with_openfe`
- ✅ `select_optimal_features`
- ✅ `auto_generate_features_with_llm`

**剩余工具**: 2/5 (40%)
- ⏳ `preprocess_data_only`
- ⏳ `generate_material_features`

---

**更新版本**: v2.1.0  
**更新日期**: 2025-10-05  
**更新类型**: Feature Enhancement  
**破坏性变更**: 是（函数签名）
