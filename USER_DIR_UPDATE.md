# 用户目录管理更新 - 快速指南

## 🎯 更新内容

实现了基于 `user_id` 和 `uuid` 的输出目录隔离系统。

---

## 📁 目录结构

```
data/
├── user_id_1/
│   ├── uuid_1/
│   │   ├── data_openfe.csv
│   │   ├── data_openfe_report.html
│   │   └── ...
│   └── uuid_2/
│       ├── data_selected_features.csv
│       ├── data_selected_features_feature_selection_report.html
│       └── ...
├── user_id_2/
│   └── uuid_3/
│       └── ...
└── anonymous/  # 未提供user_id时使用
    └── uuid_4/
        └── ...
```

---

## 🔧 使用方式

### 1. 提供user_id（通过HTTP请求头）

```http
Headers:
  user_id: user_123
```

输出将保存到: `data/user_123/{uuid}/`

### 2. 不提供user_id

输出将保存到: `data/anonymous/{uuid}/`

---

## 📊 影响的工具

### 1. `auto_feature_engineering_with_openfe`
- ✅ 所有输出文件保存到 `data/{user_id}/{uuid}/`
- ✅ 返回值包含用户信息

### 2. `select_optimal_features`
- ✅ 所有输出文件保存到 `data/{user_id}/{uuid}/`
- ✅ 返回值包含用户信息

---

## 📝 返回值新增字段

```json
{
    "user_id": "user_123",
    "run_uuid": "a1b2c3d4-e5f6-7g8h-9i0j-k1l2m3n4o5p6",
    "output_dir": "data/user_123/a1b2c3d4-e5f6-7g8h-9i0j-k1l2m3n4o5p6"
}
```

---

## ✅ 验证

运行任意工具后，检查 `data/` 目录：

```bash
cd /Users/ysl/Desktop/Code/MCP_FE_Tool
ls -la data/
```

应该看到用户目录结构。

---

## 🔄 向后兼容

- ✅ 完全向后兼容
- ✅ 未使用新功能的代码仍然正常工作
- ✅ 如果没有user_id，自动使用anonymous

---

## 📚 详细文档

请查看 `USER_OUTPUT_MANAGEMENT.md` 了解完整实现细节。
