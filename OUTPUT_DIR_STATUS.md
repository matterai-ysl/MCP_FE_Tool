# 输出目录管理实现状态

## ✅ 已完成的工具

### 1. `auto_feature_engineering_with_openfe`
- ✅ 添加 `ctx: Context` 参数
- ✅ 获取 `user_id` 并创建用户目录
- ✅ 所有输出保存到 `data/{user_id}/{uuid}/`
- ✅ 返回值包含用户信息

**输出文件**:
- `{basename}_openfe.csv`
- `{basename}_openfe_feature_descriptions.txt`
- `{basename}_openfe_report.html`

---

### 2. `select_optimal_features`
- ✅ 添加 `ctx: Context` 参数
- ✅ 获取 `user_id` 并创建用户目录
- ✅ 所有输出保存到 `data/{user_id}/{uuid}/`
- ✅ 返回值包含用户信息
- ✅ 修改 `feature_selector.py` 支持 `output_dir` 参数

**输出文件**:
- `{basename}_selected_features.csv`
- `{basename}_selected_features_feature_selection_report.png`
- `{basename}_selected_features_feature_selection_report.html`
- `{basename}_selected_features_feature_selection_details.txt`

---

## ⏳ 待修改的工具

### 3. `preprocess_data_only`
**当前状态**: 使用用户提供的 `output_path`

**需要修改**:
```python
@mcp.tool()
def preprocess_data_only(
    ctx: Context,  # ✨ 添加
    data_path: str,
    task_type: str,
    target_dims: int,
    output_path: Optional[str] = None  # ✨ 改为可选
) -> Dict[str, Any]:
    # 获取用户ID并创建输出目录
    user_id = _get_user_id_from_context(ctx)
    output_dir, run_uuid = _create_user_output_dir(user_id)
    
    # 生成输出路径
    if output_path is None:
        input_filename = os.path.basename(data_path)
        base_name = os.path.splitext(input_filename)[0]
        output_filename = f"{base_name}_preprocessed.csv"
        output_path = os.path.join(output_dir, output_filename)
    else:
        output_filename = os.path.basename(output_path)
        output_path = os.path.join(output_dir, output_filename)
    
    # ... 其余代码 ...
```

**输出文件**:
- 用户指定的文件名或 `{basename}_preprocessed.csv`

---

### 4. `generate_material_features`
**当前状态**: 可选输出，使用用户提供的 `output_path`

**需要修改**:
```python
@mcp.tool()
def generate_material_features(
    ctx: Context,  # ✨ 添加
    data_path: str,
    composition_column: str,
    feature_types: list = None,
    output_path: Optional[str] = None,
    target_dims: int = 0
) -> Dict[str, Any]:
    # 获取用户ID并创建输出目录
    user_id = _get_user_id_from_context(ctx)
    output_dir, run_uuid = _create_user_output_dir(user_id)
    
    # ... 特征生成逻辑 ...
    
    # 如果指定了输出路径，保存到用户目录
    if output_path:
        output_filename = os.path.basename(output_path)
        output_path = os.path.join(output_dir, output_filename)
    else:
        # 默认输出
        input_filename = os.path.basename(data_path)
        base_name = os.path.splitext(input_filename)[0]
        output_filename = f"{base_name}_with_features.csv"
        output_path = os.path.join(output_dir, output_filename)
    
    # ... 保存逻辑 ...
```

**输出文件**:
- 用户指定的文件名或 `{basename}_with_features.csv`

---

### 5. `auto_generate_features_with_llm`
**当前状态**: ✅ 已完成

**完成时间**: 2025-10-05

**修改内容**:
- ✅ ctx参数移到第一位并改为必需
- ✅ 使用 `_get_user_id_from_context()` 和 `_create_user_output_dir()`
- ✅ 所有输出文件保存到用户目录
- ✅ 返回值包含用户信息
- ✅ 改进了日志输出和错误处理

**输出文件**:
- `{basename}_enhanced.csv` - 增强后的数据
- `{basename}_enhanced_feature_report.txt` - 特征生成报告

**详细文档**: 见 `LLM_TOOL_UPDATE.md`

---

## 📋 修改优先级

### 高优先级 ⭐⭐⭐
- ✅ `auto_feature_engineering_with_openfe` - **已完成**
- ✅ `select_optimal_features` - **已完成**
- ✅ `auto_generate_features_with_llm` - **已完成**

### 中优先级 ⭐⭐
- ⏳ `generate_material_features` - 常用功能

### 低优先级 ⭐
- ⏳ `preprocess_data_only` - 工具函数，使用频率较低

---

## 🛠️ 修改模板

所有待修改的工具都遵循相同的模式：

```python
@mcp.tool()
def tool_name(
    ctx: Context,  # 1. 添加ctx参数（如果没有）
    # ... 其他参数 ...
) -> Dict[str, Any]:
    # 2. 获取用户ID并创建输出目录
    user_id = _get_user_id_from_context(ctx)
    output_dir, run_uuid = _create_user_output_dir(user_id)
    
    try:
        # 3. 工具逻辑...
        
        # 4. 修改输出路径生成
        input_filename = os.path.basename(data_path)
        base_name = os.path.splitext(input_filename)[0]
        output_filename = f"{base_name}_suffix.csv"
        output_path = os.path.join(output_dir, output_filename)
        
        # 5. 保存文件到用户目录
        # ... 保存逻辑 ...
        
        # 6. 添加用户信息到返回值
        result["用户信息"] = {
            "用户ID": user_id if user_id else "anonymous",
            "运行UUID": run_uuid,
            "输出目录": output_dir
        }
        
        return result
        
    except Exception as e:
        # ... 错误处理 ...
```

---

## 📊 工具对比表

| 工具名称 | 状态 | 输出文件数 | Context参数 | user_id获取 | 用户目录 | 返回值更新 |
|---------|------|-----------|------------|-----------|---------|-----------|
| `auto_feature_engineering_with_openfe` | ✅ 已完成 | 3 | ✅ | ✅ | ✅ | ✅ |
| `select_optimal_features` | ✅ 已完成 | 4 | ✅ | ✅ | ✅ | ✅ |
| `auto_generate_features_with_llm` | ✅ 已完成 | 2 | ✅ | ✅ | ✅ | ✅ |
| `preprocess_data_only` | ⏳ 待修改 | 1 | ❌ | ❌ | ❌ | ❌ |
| `generate_material_features` | ⏳ 待修改 | 1 | ❌ | ❌ | ❌ | ❌ |

---

## 🔍 检查清单

修改每个工具时，确保：

- [ ] 添加或验证 `ctx: Context` 参数
- [ ] 调用 `_get_user_id_from_context(ctx)` 获取user_id
- [ ] 调用 `_create_user_output_dir(user_id)` 创建目录
- [ ] 所有输出文件都保存到 `output_dir`
- [ ] 返回值包含 `user_id`, `run_uuid`, `output_dir`
- [ ] 测试无user_id的情况（应使用anonymous）
- [ ] 测试有user_id的情况
- [ ] 验证输出目录结构正确

---

## ✅ 测试验证

### 1. 测试已完成的工具

```bash
# 运行特征工程工具
# 应在 data/user_id/uuid/ 下看到输出

# 运行特征选择工具
# 应在 data/user_id/uuid/ 下看到输出
```

### 2. 检查目录结构

```bash
cd /Users/ysl/Desktop/Code/MCP_FE_Tool
tree data/ -L 3
```

应该看到：
```
data/
├── user_id_1/
│   └── uuid_1/
│       ├── *.csv
│       └── *.html
```

---

## 📝 注意事项

1. **向后兼容**: 所有修改都保持向后兼容
2. **错误处理**: 如果获取user_id失败，自动使用anonymous
3. **权限**: 确保data/目录有写权限
4. **磁盘空间**: 考虑实现定期清理机制
5. **用户配额**: 未来可能需要实现配额限制

---

**最后更新**: 2025-10-05  
**状态**: 3/5 工具已完成 (60%)

**已完成**:
- ✅ `auto_feature_engineering_with_openfe`
- ✅ `select_optimal_features`
- ✅ `auto_generate_features_with_llm`

**剩余**:
- ⏳ `preprocess_data_only`
- ⏳ `generate_material_features`
