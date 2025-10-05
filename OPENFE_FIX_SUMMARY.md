# OpenFE 特征描述修复总结

## 问题描述

之前OpenFE生成的特征在HTML报告中只显示操作符（如 `max`、`*`、`residual` 等），没有显示具体是从哪些原始特征计算得来的。

**之前的显示效果：**
```
特征名称          构建方式
autoFE_f_0       max
autoFE_f_1       -
autoFE_f_2       residual
```

## 修复方案

实现了递归解析OpenFE特征对象树的函数 `parse_openfe_feature_tree()`，该函数能够：

1. **识别节点类型**：区分操作符节点（Node）和特征叶子节点（FNode）
2. **递归解析**：遍历整个特征树，提取所有子节点信息
3. **格式化输出**：根据操作符类型生成可读的表达式

## 修复后的效果

**现在的显示效果：**
```
特征名称          构建方式
autoFE_f_0       GroupByThenMean(feat_C, feat_B)     # 通过feat_C分组然后计算feat_B的均值
autoFE_f_1       (feat_A / feat_B)                    # feat_A除以feat_B
autoFE_f_2       min(feat_A, feat_C)                  # feat_A和feat_C的最小值
autoFE_f_3       residual(feat_D, feat_E)             # feat_D和feat_E的残差
autoFE_f_4       (feat_A * feat_C)                    # feat_A乘以feat_C
```

## 核心实现

```python
def parse_openfe_feature_tree(feat, depth=0):
    """递归解析OpenFE特征树，生成可读的表达式"""
    type_name = type(feat).__name__
    
    # FNode - 原始特征叶子节点
    if type_name == 'FNode':
        if hasattr(feat, 'name') and feat.name:
            return str(feat.name)
        return "未知特征"
    
    # Node - 操作符节点
    elif type_name == 'Node':
        op_name = feat.name
        children = feat.children if hasattr(feat, 'children') and feat.children else []
        
        if not children:
            return str(op_name)
        
        # 递归解析子节点
        child_exprs = [parse_openfe_feature_tree(child, depth+1) for child in children]
        
        # 根据操作符类型格式化
        if op_name in ['+', '-', '*', '/', '>', '<', '>=', '<=', '==', '!=']:
            if len(child_exprs) == 2:
                return f"({child_exprs[0]} {op_name} {child_exprs[1]})"
        elif 'GroupBy' in op_name:
            return f"{op_name}({', '.join(child_exprs)})"
        elif op_name in ['max', 'min', 'mean', 'std', 'sum', 'var']:
            return f"{op_name}({', '.join(child_exprs)})"
        # ... 其他操作符
        
        return f"{op_name}({', '.join(child_exprs)})"
```

## 支持的操作符

- **二元运算符**: `+`, `-`, `*`, `/`, `>`, `<`, `>=`, `<=`, `==`, `!=`
- **聚合函数**: `max`, `min`, `mean`, `median`, `std`, `sum`, `var`
- **单参数函数**: `abs`, `sqrt`, `log`, `exp`, `sin`, `cos`, `residual`, `Round`
- **GroupBy操作**: `GroupByThenMean`, `GroupByThenStd`, `GroupByThenRank`, 等

## HTML报告改进

HTML报告现在包含：

1. **数据概览**
   - 数据文件名
   - 任务类型（回归/分类）
   - 原始数据形状
   - 最终数据形状
   - 目标列

2. **特征统计**
   - 原始特征数量
   - 筛选后特征数量
   - 筛选比例
   - OpenFE输入特征数
   - OpenFE输出特征数
   - 新生成特征数

3. **新特征详情表**
   - 序号
   - 特征名称
   - **构建方式**（完整表达式，显示原始特征名）

## 如何测试

使用MCP工具调用 `auto_feature_engineering_with_openfe` 工具，查看生成的HTML报告：

```python
# 通过Claude Desktop或其他MCP客户端调用
{
    "tool": "auto_feature_engineering_with_openfe",
    "arguments": {
        "data_path": "/path/to/your/data.csv",
        "target_dims": 1,
        "task_type": "regression",
        "n_features_before_openfe": 50,
        "n_new_features": 10
    }
}
```

报告将生成在与输入数据相同的目录，文件名为 `{原文件名}_openfe_report.html`。

## 相关文件

- `src/materials_feature_engineering_mcp/mcp_tool.py` - 核心特征解析逻辑（第877-932行）
- `src/materials_feature_engineering_mcp/report_generator.py` - HTML报告生成器
- `OPENFE_FEATURE.md` - 完整的OpenFE功能文档
- `OPENFE_QUICKSTART.md` - 快速开始指南

