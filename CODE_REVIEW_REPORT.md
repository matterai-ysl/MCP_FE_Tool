# 代码审查报告

审查时间：2026-04-18

审查范围：
- `single_port_server.py`
- `__main__.py`
- `test_html_report.py`
- `src/materials_feature_engineering_mcp/*.py`

验证方式：
- 静态通读全部源码
- 关键路径检索（路由、路径拼接、异常处理、参数校验）
- 运行 `python3 -m compileall src single_port_server.py __main__.py test_html_report.py`

## 主要问题

### P0 - `feature_generator.py` 当前存在语法错误，整个 MCP 服务无法启动

- 位置：`src/materials_feature_engineering_mcp/feature_generator.py:600-617`
- 现象：HTML 模板里嵌套了多层 `f'''...{''.join([f'''...`，`compileall` 已直接报错：`SyntaxError: f-string: unmatched '['`
- 影响：
  - `mcp_tool.py` 在模块导入阶段就会 `from .feature_generator import MaterialsFeatureGenerator`
  - 因此 `mcp_tool.py`、`single_port_server.py` 和相关入口都会在 import 时失败
  - `generate_materials_basic_features` 整条链路不可用
- 建议：
  - 不要在一个超长 f-string 里再内嵌 list comprehension + 子 f-string
  - 改成先组装 `section_html` / `rows_html` 字符串，再插入模板

### P1 - `user_id` 直接参与路径拼接，存在目录穿越写文件风险

- 位置：`src/materials_feature_engineering_mcp/mcp_tool.py:30-59`
- 关联位置：`src/materials_feature_engineering_mcp/config.py:8-12`
- 现象：
  - `_create_user_output_dir()` 直接使用请求头里的 `user_id`
  - 例如 `user_id='../../tmp/pwn'` 时，生成路径会落到 `data` 目录之外
  - `get_download_url()` / `get_static_url()` 还会把 `../` 片段原样带进返回 URL
- 影响：
  - 服务端可能把输出文件写到项目外部目录
  - 返回给调用方的下载链接也会变成带穿越片段的异常 URL
- 建议：
  - 对 `user_id` 做白名单清洗，只允许字母、数字、`-`、`_`
  - 创建目录后再 `resolve()`，强校验最终路径必须位于 `data/` 下
  - URL 生成应基于已验证的相对路径，而不是直接对外反射输入

### P1 - 服务宣称支持 `/static/models/...`，但实际没有挂载这条路由

- 位置：`single_port_server.py:69-71`
- 关联位置：`single_port_server.py:106-107`, `single_port_server.py:168-178`, `single_port_server.py:258-259`
- 现象：
  - `/api/info` 和 `list_model_files()` 返回的静态模型地址都是 `/static/models/{path}`
  - 实际挂载的是 `/static`、`/static/model`、`/static/reports`
  - 没有 `/static/models`
- 影响：
  - API 返回的 `static_url` 与文档示例会 404
  - 前端/调用方按返回值访问静态模型文件时会失败
- 建议：
  - 统一成单一协议：要么真的挂载 `/static/models`，要么把所有返回值改成当前已存在的路由
  - `config.py`、`utils.py`、`single_port_server.py` 的 URL 规则应收口到一个地方维护

### P2 - 小样本数据会在分布检验阶段直接报错，中断整份分析报告

- 位置：`src/materials_feature_engineering_mcp/data_explorer.py:226-246`
- 现象：
  - `_test_distributions()` 无条件调用 `stats.shapiro(...)`
  - Shapiro-Wilk 要求样本数至少为 3
  - 只要某列有效值少于 3 个，`distribution_analysis()` 或目标分析就会抛异常
- 影响：
  - 小数据集、缺失严重的数据集会让 `explore_materials_data()` 直接失败
  - 当前上层又把异常包装成泛化错误，调用方很难知道是哪一列导致的
- 建议：
  - 在检验前判断 `len(data) >= 3`
  - 样本不足时返回“样本过少，跳过检验”而不是抛错
  - 顺手把常量列、全空列也一起兜底

### P2 - `test_html_report.py` 不是测试，而是带副作用的脚本；被 pytest 收集时会直接执行

- 位置：`test_html_report.py:4-40`
- 现象：
  - 文件名以 `test_` 开头，会被 pytest 收集
  - 顶层代码会直接写 `test_sample.csv`、调用特征生成逻辑、生成报告
  - 还可能触发 LLM 请求与外部依赖
- 影响：
  - CI/本地跑测试时会污染工作目录
  - 测试收集阶段就可能访问网络，导致用例极不稳定
  - 这类脚本也会掩盖真正的自动化测试缺口
- 建议：
  - 改成真正的 pytest 用例
  - 使用 `tmp_path`、mock LLM 响应、断言输出文件结构
  - 如果只是演示脚本，改名为 `scripts/generate_sample_report.py`

### P2 - `target_dims` 校验不一致，部分入口允许“没有任何特征列”的非法状态

- 位置：`src/materials_feature_engineering_mcp/mcp_tool.py:163-164`
- 关联位置：`src/materials_feature_engineering_mcp/data_explorer.py:64-66`, `src/materials_feature_engineering_mcp/mcp_tool.py:588-592`, `src/materials_feature_engineering_mcp/feature_selector.py:769-770`
- 现象：
  - `feature_selector.py` 已正确限制 `target_dims < len(df.columns)`
  - 但 `explore_materials_data()` 只校验 `target_dims > 0`
  - `auto_feature_engineering_with_openfe()` 也允许 `target_dims == len(df.columns)`
- 影响：
  - 会进入“全部列都是目标、零特征”的非法状态
  - 后续报错位置分散，表现成空特征、筛选失败、OpenFE 失败等晚期异常
- 建议：
  - 统一封装 `validate_target_dims(total_cols, target_dims)`
  - 所有入口都执行同一套校验并给出一致错误信息

### P3 - `MaterialsFeatureGenerator` 的公开参数和调用协议与实际实现不一致

- 位置：`src/materials_feature_engineering_mcp/feature_generator.py:35-45`
- 关联位置：`src/materials_feature_engineering_mcp/feature_generator.py:143-145`, `src/materials_feature_engineering_mcp/feature_generator.py:208-219`, `src/materials_feature_engineering_mcp/mcp_tool.py:276-286`
- 现象：
  - 构造函数暴露了 `api_key` / `api_base`，但实际完全忽略传入值，只读环境变量
  - `generate_composition_features(..., feature_types=...)` 暴露了 `feature_types`，但实现始终固定跑三类 featurizer
  - `selected_columns` 的 value 也没有真正参与控制逻辑
- 影响：
  - API 表面支持“依赖注入”和“选择特征类型”，实际并不支持
  - 调用方会以为自己能精确配置，但结果不可预测
- 建议：
  - 明确二选一：要么删掉这些参数，要么把它们真正接进实现
  - 这块最好补单测，保证调用协议和实际行为一致

## 需要重构的地方

### 1. 统一“路径 / URL / 文件服务”协议

当前这套规则分散在：
- `config.py`
- `utils.py`
- `single_port_server.py`
- `mcp_tool.py`

而且已经出现了三套不同约定：
- `/download/file/...`
- `/static/...`
- `/static/models/...` / `/static/model/...` / `/static/reports/...`

建议提炼成一个单独模块，例如 `storage_paths.py`，统一负责：
- 安全路径生成
- 相对路径与绝对路径转换
- 下载 URL / 静态 URL 生成
- 服务端路由常量

### 2. 统一“数据加载 + 目标列拆分 + 基础校验”逻辑

相似逻辑重复出现在：
- `DataExplorer.load_data()`
- `MaterialsFeatureGenerator.load_data()`
- `auto_feature_engineering_with_openfe()`
- `select_best_features()`

现在每个入口对：
- CSV / Excel 回退逻辑
- URL 处理
- `target_dims` 校验
- 缺失值处理

都写了一遍，导致行为不一致。建议抽成共享 helper，例如：
- `load_tabular_data(data_path)`
- `split_features_and_targets(df, target_dims)`
- `normalize_target_series(y, task_type)`

### 3. 减少“吞异常 + print”的风格，改成结构化错误

当前很多入口直接：
- `except Exception as e`
- 返回一个泛化字典
- 或继续 silently fallback

这会让真正的问题被隐藏，排查成本很高。建议：
- 内部函数抛明确异常类型
- MCP 工具层统一做用户友好的错误包装
- 使用 `logging` 替代大面积 `print`

### 4. 把报告渲染从业务逻辑里拆出来

`feature_generator.py` 和 `feature_selector.py` 里都塞了很长的 HTML 字符串，已经直接引入了可维护性问题，甚至这次还演变成了语法错误。

建议：
- 模板放单独文件，或最少拆成多个小函数
- 业务层只准备数据
- 渲染层只负责模板拼接和转义

## 测试与覆盖率建议

建议尽快补的自动化测试：
- 路径安全测试：`user_id` 含 `../`、绝对路径、空值
- 路由契约测试：`/api/info` 返回的 URL 必须与实际挂载路由一致
- 小样本数据测试：1 行、2 行、全空列、常量列
- 特征生成测试：非法化学式、多个组成列、关闭某类 featurizer
- 报告生成测试：验证 HTML 生成函数至少可 import、可落盘、包含关键字段

## 本次审查结论

当前代码里已经存在 3 个会直接影响可用性的高优先级问题：
- `feature_generator.py` 语法错误，导致相关模块无法导入
- 输出目录存在目录穿越风险
- 静态文件 URL 与实际路由不一致

在修复这些问题前，不建议把当前版本继续作为稳定 MCP 服务对外暴露。
