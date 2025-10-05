# 用户输出目录管理系统

## 📋 概述

**更新日期**: 2025-10-05  
**版本**: v2.0.0

本次更新实现了基于用户ID和UUID的输出目录隔离系统，确保每个用户的每次运行都有独立的输出目录。

---

## 🎯 功能特点

### 1. 用户隔离
- 每个用户有独立的输出目录
- 基于HTTP请求头中的 `user_id` 进行区分
- 未提供 `user_id` 时使用 `anonymous` 作为默认用户

### 2. 会话管理
- 每次工具调用生成唯一的UUID
- UUID用于标识单次运行会话
- 便于追踪和管理历史运行记录

### 3. 目录结构
```
data/
├── user_id_1/
│   ├── uuid_1/
│   │   ├── data_openfe.csv
│   │   ├── data_openfe_feature_descriptions.txt
│   │   └── data_openfe_report.html
│   ├── uuid_2/
│   │   ├── data_selected_features.csv
│   │   ├── data_selected_features_feature_selection_report.png
│   │   ├── data_selected_features_feature_selection_report.html
│   │   └── data_selected_features_feature_selection_details.txt
│   └── uuid_3/
│       └── ...
├── user_id_2/
│   └── uuid_4/
│       └── ...
└── anonymous/
    └── uuid_5/
        └── ...
```

---

## 🔧 实现细节

### 1. 辅助函数

#### `_create_user_output_dir(user_id: Optional[str]) -> Tuple[str, str]`

创建用户输出目录并生成UUID。

**参数**:
- `user_id`: 用户ID，如果为None或空字符串则使用'anonymous'

**返回**:
- `(output_dir, run_uuid)`: 输出目录路径和运行UUID

**示例**:
```python
output_dir, run_uuid = _create_user_output_dir("user_123")
# output_dir: "data/user_123/a1b2c3d4-e5f6-7g8h-9i0j-k1l2m3n4o5p6"
# run_uuid: "a1b2c3d4-e5f6-7g8h-9i0j-k1l2m3n4o5p6"
```

#### `_get_user_id_from_context(ctx: Optional[Context]) -> Optional[str]`

从FastMCP的Context对象中提取user_id。

**参数**:
- `ctx`: FastMCP的Context对象

**返回**:
- 用户ID字符串或None

**示例**:
```python
user_id = _get_user_id_from_context(ctx)
# user_id: "user_123" 或 None
```

---

### 2. 修改的工具

#### `auto_feature_engineering_with_openfe`

**修改内容**:
1. 添加 `ctx: Context` 参数（第一个参数）
2. 在函数开头获取用户ID并创建输出目录
3. 修改输出路径生成逻辑，使用用户目录
4. 在返回值中添加用户信息

**代码示例**:
```python
@mcp.tool()
def auto_feature_engineering_with_openfe(
    ctx: Context,  # ✨ 新增
    data_path: str,
    target_dims: int = 1,
    task_type: str = "regression",
    n_features_before_openfe: int = 50,
    n_new_features: int = 10,
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    # 获取用户ID并创建输出目录
    user_id = _get_user_id_from_context(ctx)
    output_dir, run_uuid = _create_user_output_dir(user_id)
    
    # ... 工具逻辑 ...
    
    # 生成输出路径到用户目录
    if output_path is None:
        input_filename = os.path.basename(data_path)
        base_name = os.path.splitext(input_filename)[0]
        output_filename = f"{base_name}_openfe.csv"
        output_path = os.path.join(output_dir, output_filename)
    else:
        output_filename = os.path.basename(output_path)
        output_path = os.path.join(output_dir, output_filename)
    
    # ... 保存文件 ...
    
    # 添加用户和会话信息到返回值
    report["用户信息"] = {
        "用户ID": user_id if user_id else "anonymous",
        "运行UUID": run_uuid,
        "输出目录": output_dir
    }
    
    return report
```

#### `select_optimal_features`

**修改内容**:
1. 添加 `ctx: Context` 参数（第一个参数）
2. 在函数开头获取用户ID并创建输出目录
3. 传递 `output_dir` 参数给 `select_best_features`
4. 在返回值中添加用户信息

**代码示例**:
```python
@mcp.tool()
def select_optimal_features(
    ctx: Context,  # ✨ 新增
    data_path: str,
    target_dims: int = 1,
    task_type: str = "regression",
    cv_folds: int = 5,
    min_features: int = 1,
    step: int = 1
) -> Dict[str, Any]:
    # 获取用户ID并创建输出目录
    user_id = _get_user_id_from_context(ctx)
    output_dir, run_uuid = _create_user_output_dir(user_id)
    
    # 调用特征选择函数，传递output_dir
    result = select_best_features(
        data_path=data_path,
        target_dims=target_dims,
        task_type=task_type,
        cv_folds=cv_folds,
        min_features=min_features,
        step=step,
        output_dir=output_dir  # ✨ 新增
    )
    
    # ... 工具逻辑 ...
    
    # 添加用户信息到返回值
    return {
        # ... 其他字段 ...
        'user_id': user_id if user_id else "anonymous",
        'run_uuid': run_uuid,
        'output_dir': output_dir
    }
```

---

### 3. 底层函数修改

#### `select_best_features` (feature_selector.py)

**修改内容**:
1. 添加 `output_dir: Optional[str] = None` 参数
2. 修改输出路径生成逻辑，优先使用 `output_dir`
3. 兼容模式：如果没有提供 `output_dir`，使用原始行为

**代码示例**:
```python
def select_best_features(
    data_path: str,
    target_dims: int = 1,
    task_type: str = 'regression',
    cv_folds: int = 5,
    min_features: int = 1,
    step: int = 1,
    output_suffix: str = '_selected_features',
    output_dir: Optional[str] = None  # ✨ 新增
) -> Dict[str, Any]:
    # ... 函数逻辑 ...
    
    # 获取输入文件名
    if data_path.startswith(('http://', 'https://', 'ftp://', 's3://', 'gs://')):
        filename = os.path.basename(data_path)
        if not filename.endswith('.csv'):
            filename = 'feature_selection_output.csv'
    else:
        filename = os.path.basename(data_path)
    
    # 生成输出文件名
    base_name = os.path.splitext(filename)[0]
    output_filename = f"{base_name}{output_suffix}.csv"
    
    # 如果提供了output_dir（用户目录），使用它
    if output_dir:
        output_path = os.path.join(output_dir, output_filename)
        print(f"✓ 使用用户输出目录: {output_dir}")
    else:
        # 兼容模式：没有output_dir时，保存到原始位置
        if data_path.startswith(('http://', 'https://', 'ftp://', 's3://', 'gs://')):
            output_path = os.path.abspath(output_filename)
        else:
            output_path = os.path.join(os.path.dirname(os.path.abspath(data_path)), output_filename)
    
    # ... 保存文件 ...
```

---

## 📊 使用示例

### 1. 带用户ID的调用

**HTTP请求头**:
```
user_id: user_123
```

**调用**:
```python
result = auto_feature_engineering_with_openfe(
    data_path="materials_data.csv",
    target_dims=1,
    task_type="regression",
    n_features_before_openfe=50,
    n_new_features=10
)
```

**输出**:
```
📁 Output directory created: data/user_123/a1b2c3d4-e5f6-7g8h-9i0j-k1l2m3n4o5p6
   User ID: user_123
   Run UUID: a1b2c3d4-e5f6-7g8h-9i0j-k1l2m3n4o5p6

✓ Enhanced data saved to: data/user_123/a1b2c3d4-e5f6-7g8h-9i0j-k1l2m3n4o5p6/materials_data_openfe.csv
```

**返回值**:
```json
{
    "状态": "成功",
    "输出文件": "data/user_123/a1b2c3d4.../materials_data_openfe.csv",
    "用户信息": {
        "用户ID": "user_123",
        "运行UUID": "a1b2c3d4-e5f6-7g8h-9i0j-k1l2m3n4o5p6",
        "输出目录": "data/user_123/a1b2c3d4-e5f6-7g8h-9i0j-k1l2m3n4o5p6"
    }
}
```

### 2. 无用户ID的调用（匿名）

**HTTP请求头**:
```
(没有 user_id)
```

**输出**:
```
📁 Output directory created: data/anonymous/b2c3d4e5-f6g7-8h9i-0j1k-l2m3n4o5p6q7
   User ID: anonymous
   Run UUID: b2c3d4e5-f6g7-8h9i-0j1k-l2m3n4o5p6q7

✓ Enhanced data saved to: data/anonymous/b2c3d4e5.../materials_data_openfe.csv
```

**返回值**:
```json
{
    "状态": "成功",
    "输出文件": "data/anonymous/b2c3d4e5.../materials_data_openfe.csv",
    "用户信息": {
        "用户ID": "anonymous",
        "运行UUID": "b2c3d4e5-f6g7-8h9i-0j1k-l2m3n4o5p6q7",
        "输出目录": "data/anonymous/b2c3d4e5-f6g7-8h9i-0j1k-l2m3n4o5p6q7"
    }
}
```

---

## 🔒 安全性考虑

### 1. 路径遍历防护

虽然当前实现使用 `user_id` 作为目录名，但建议在生产环境中添加额外的验证：

```python
def _sanitize_user_id(user_id: str) -> str:
    """清理user_id，防止路径遍历攻击"""
    import re
    # 只允许字母、数字、下划线、连字符
    sanitized = re.sub(r'[^a-zA-Z0-9_-]', '_', user_id)
    # 防止 ../ 等路径遍历
    sanitized = sanitized.replace('..', '_')
    return sanitized

def _create_user_output_dir(user_id: Optional[str] = None) -> Tuple[str, str]:
    if not user_id or user_id.strip() == "":
        user_id = "anonymous"
    else:
        user_id = _sanitize_user_id(user_id)  # ✨ 添加清理
    
    # ... 其余代码 ...
```

### 2. 磁盘空间管理

建议实现以下功能：

#### 定期清理
```python
def cleanup_old_runs(max_age_days: int = 7):
    """清理超过指定天数的运行目录"""
    import time
    from pathlib import Path
    
    base_dir = Path("data")
    current_time = time.time()
    max_age_seconds = max_age_days * 24 * 3600
    
    for user_dir in base_dir.iterdir():
        if not user_dir.is_dir():
            continue
        for run_dir in user_dir.iterdir():
            if not run_dir.is_dir():
                continue
            # 检查目录年龄
            dir_age = current_time - run_dir.stat().st_mtime
            if dir_age > max_age_seconds:
                shutil.rmtree(run_dir)
                print(f"✓ Cleaned up old run: {run_dir}")
```

#### 用户配额
```python
def check_user_quota(user_id: str, max_size_mb: int = 1000) -> bool:
    """检查用户是否超过存储配额"""
    import shutil
    from pathlib import Path
    
    user_dir = Path("data") / user_id
    if not user_dir.exists():
        return True
    
    total_size = sum(
        f.stat().st_size 
        for f in user_dir.rglob('*') 
        if f.is_file()
    )
    total_size_mb = total_size / (1024 * 1024)
    
    return total_size_mb < max_size_mb
```

---

## 📈 统计和监控

### 1. 运行统计

可以添加统计功能来追踪用户使用情况：

```python
def log_run_info(user_id: str, run_uuid: str, tool_name: str, status: str):
    """记录运行信息"""
    import json
    from datetime import datetime
    
    log_file = Path("data") / ".run_log.jsonl"
    
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "user_id": user_id,
        "run_uuid": run_uuid,
        "tool_name": tool_name,
        "status": status
    }
    
    with open(log_file, 'a') as f:
        f.write(json.dumps(log_entry) + '\n')
```

### 2. 用户活跃度分析

```python
def get_user_stats(user_id: Optional[str] = None) -> Dict[str, Any]:
    """获取用户统计信息"""
    base_dir = Path("data")
    
    if user_id:
        user_dir = base_dir / user_id
        if not user_dir.exists():
            return {"error": "User not found"}
        
        runs = list(user_dir.iterdir())
        return {
            "user_id": user_id,
            "total_runs": len(runs),
            "total_size_mb": sum(
                f.stat().st_size for f in user_dir.rglob('*') if f.is_file()
            ) / (1024 * 1024)
        }
    else:
        # 所有用户统计
        all_users = {}
        for user_dir in base_dir.iterdir():
            if not user_dir.is_dir() or user_dir.name.startswith('.'):
                continue
            runs = list(user_dir.iterdir())
            all_users[user_dir.name] = {
                "total_runs": len(runs),
                "total_size_mb": sum(
                    f.stat().st_size for f in user_dir.rglob('*') if f.is_file()
                ) / (1024 * 1024)
            }
        return all_users
```

---

## 🔄 向后兼容性

### 1. 兼容模式

所有修改都保持向后兼容：
- 如果没有提供 `output_dir`，函数使用原始行为
- 如果没有 `user_id`，使用 `anonymous` 作为默认值
- 旧的调用方式仍然有效

### 2. 迁移指南

#### 从旧版本迁移

如果你有旧版本的输出文件，可以使用以下脚本迁移：

```python
def migrate_old_outputs():
    """将旧的输出文件迁移到新的目录结构"""
    import shutil
    from pathlib import Path
    import uuid
    
    # 假设旧文件在当前目录
    old_files = list(Path(".").glob("*_openfe.csv")) + \
                list(Path(".").glob("*_selected_features.csv"))
    
    for old_file in old_files:
        # 创建anonymous用户的运行目录
        run_uuid = str(uuid.uuid4())
        new_dir = Path("data") / "anonymous" / run_uuid
        new_dir.mkdir(parents=True, exist_ok=True)
        
        # 移动文件
        new_path = new_dir / old_file.name
        shutil.move(str(old_file), str(new_path))
        print(f"✓ Migrated: {old_file} -> {new_path}")
        
        # 移动相关的报告文件
        related_files = [
            old_file.parent / f"{old_file.stem}_report.html",
            old_file.parent / f"{old_file.stem}_feature_descriptions.txt",
            old_file.parent / f"{old_file.stem}_feature_selection_report.png",
            old_file.parent / f"{old_file.stem}_feature_selection_report.html",
            old_file.parent / f"{old_file.stem}_feature_selection_details.txt"
        ]
        
        for related in related_files:
            if related.exists():
                shutil.move(str(related), str(new_dir / related.name))
                print(f"✓ Migrated: {related} -> {new_dir / related.name}")
```

---

## 📝 修改的文件清单

### 1. `mcp_tool.py`

**新增**:
- `_create_user_output_dir()` - 创建用户输出目录
- `_get_user_id_from_context()` - 获取用户ID
- 导入: `uuid`, `Path`, `Tuple`

**修改**:
- `auto_feature_engineering_with_openfe()` - 添加ctx参数，使用用户目录
- `select_optimal_features()` - 添加ctx参数，传递output_dir

### 2. `feature_selector.py`

**新增**:
- 导入: `Optional`

**修改**:
- `select_best_features()` - 添加output_dir参数，修改输出路径逻辑

---

## ✅ 测试验证

### 1. 单元测试

```python
def test_create_user_output_dir():
    """测试用户目录创建"""
    # 测试有user_id
    output_dir, run_uuid = _create_user_output_dir("test_user")
    assert "data/test_user/" in output_dir
    assert len(run_uuid) == 36  # UUID长度
    
    # 测试无user_id
    output_dir, run_uuid = _create_user_output_dir(None)
    assert "data/anonymous/" in output_dir
    
    # 测试空user_id
    output_dir, run_uuid = _create_user_output_dir("")
    assert "data/anonymous/" in output_dir
```

### 2. 集成测试

```python
def test_feature_selection_with_user_dir():
    """测试特征选择工具使用用户目录"""
    # 模拟ctx
    class MockContext:
        class RequestContext:
            class Request:
                headers = {"user_id": "test_user_123"}
            request = Request()
        request_context = RequestContext()
    
    ctx = MockContext()
    
    # 调用工具
    result = select_optimal_features(
        ctx=ctx,
        data_path="test_data.csv",
        target_dims=1
    )
    
    # 验证输出
    assert "data/test_user_123/" in result['output_file']
    assert result['user_id'] == "test_user_123"
    assert 'run_uuid' in result
```

---

## 🎉 总结

本次更新实现了完整的用户输出目录管理系统，具有以下优势：

1. ✅ **用户隔离** - 每个用户有独立的输出空间
2. ✅ **会话追踪** - 每次运行都有唯一标识
3. ✅ **向后兼容** - 不影响现有代码
4. ✅ **安全性** - 可以添加路径验证和配额管理
5. ✅ **可扩展** - 易于添加统计和监控功能

**影响的工具**:
- `auto_feature_engineering_with_openfe`
- `select_optimal_features`

**未来扩展**:
- 添加更多工具的用户目录支持
- 实现自动清理功能
- 添加用户配额管理
- 实现运行历史查询API

---

**更新版本**: v2.0.0  
**更新日期**: 2025-10-05  
**更新类型**: Feature Enhancement  
**影响范围**: Output Management System
