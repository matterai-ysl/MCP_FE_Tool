# Materials Feature Engineering MCP

面向材料科学场景的特征工程 MCP 工具，当前聚焦两类输入：

1. 表格中的 `composition` / `SMILES` 字符串
2. `CIF zip + metadata table` 结构化输入

它的职责是把这些输入稳定地转换成特征列，并把训练阶段用过的特征工程配置长期保存在本地，后续可以通过 `pipeline_id` 对测试集或预测集复用同一套处理流程。

当前工具明确不负责：

- 归一化、标准化、正则化
- 模型训练与推理
- 自动猜列并自动编排整条 Agent 流程

## 当前对外 MCP 工具

### 表格特征工程

| Tool | 职责 |
| --- | --- |
| `summarize_dataset` | 返回列名、dtype、缺失率、示例值和预览行，供 Agent 判断列语义 |
| `fit_feature_pipeline` | 在训练集上执行 composition / SMILES 特征工程，并持久化本地 pipeline |
| `transform_with_pipeline` | 基于已有 `pipeline_id` 对测试集或预测集执行完全一致的特征工程 |
| `inspect_pipeline` | 查看 pipeline 配置、特征顺序、产物路径和摘要 |
| `list_pipelines` | 列出本地已保存的表格 pipeline |

### CIF 特征工程

| Tool | 职责 |
| --- | --- |
| `summarize_cif_archive` | 读取 `zip + metadata table`，检查 `cif_filename` 对齐情况 |
| `fit_cif_pipeline` | 从 CIF 压缩包和 metadata 训练并保存 CIF pipeline |
| `transform_with_cif_pipeline` | 用已有 `pipeline_id` 对新的 CIF 压缩包复用相同特征工程 |
| `inspect_cif_pipeline` | 查看 CIF pipeline 配置、特征顺序、产物路径和摘要 |
| `list_cif_pipelines` | 列出本地已保存的 CIF pipeline |

## 环境要求

- Python `>= 3.10`
- `uv`

当前仓库文档按下面这套真实环境校验过：

- `uv 0.10.10`
- `Python 3.12.13`
- `.venv` 由 `uv sync` 管理

## 安装

### 1. 创建并同步 uv 环境

```bash
uv sync
```

### 2. 可选：确认当前解释器

```bash
uv --version
./.venv/bin/python --version
```

### 3. 可选：激活虚拟环境

```bash
source .venv/bin/activate
```

## MCP 启动方式

这个仓库当前有两种启动方式：

1. `stdio` 模式，适合桌面 MCP 客户端直接拉起
2. `streamable-http` 模式，适合本地或服务端部署成 HTTP MCP 服务

### 1. `stdio` 启动

仓库根目录执行：

```bash
uv run python __main__.py
```

这条命令会启动 FastMCP 的 `STDIO` transport。当前仓库里已验证该命令可以正常拉起：

- Server name: `Materials Feature Engineering Tool`
- Transport: `STDIO`

如果你要把它接到 MCP 客户端，最常见的配置方式可以写成：

```json
{
  "mcpServers": {
    "materials-feature-engineering": {
      "command": "uv",
      "args": ["run", "python", "__main__.py"],
      "cwd": "/absolute/path/to/MCP_FE_Tool"
    }
  }
}
```

### 2. `streamable-http` 启动

仓库根目录执行：

```bash
uv run python single_port_server.py --host 0.0.0.0 --port 8180
```

如果你只是本机验证，也可以显式绑定回环地址：

```bash
uv run python single_port_server.py --host 127.0.0.1 --port 8101
```

当前仓库里已验证 HTTP 模式可以正常启动，并通过健康检查：

```bash
curl http://127.0.0.1:8101/api/health
```

返回示例：

```json
{
  "status": "healthy",
  "mcp_status": "active",
  "transport": "streamable-http",
  "directories": {
    "trained_models_exists": true,
    "reports_exists": true
  },
  "port": 8101,
  "host": "127.0.0.1"
}
```

HTTP 模式下的主要端点：

- MCP endpoint: `http://host:port/mcp`
- 服务信息: `http://host:port/api/info`
- 健康检查: `http://host:port/api/health`
- 数据目录文件列表: `http://host:port/api/models/list`
- 文件下载: `http://host:port/api/download/file/{path}`
- 静态文件: `http://host:port/static/{path}`

### 3. 关于返回的下载 URL

工具返回的产物 URL 由 [src/materials_feature_engineering_mcp/config.py](/Users/songlin/Desktop/Code/MCP_FE_Tool/src/materials_feature_engineering_mcp/config.py) 中的 `BASE_URL` 拼接。

当前默认值是：

```python
BASE_URL = "https://www.matterai.cn/fe"
```

如果你在本地或自己的服务器上部署 HTTP MCP，请把它改成你的实际服务地址，否则工具返回的下载链接不会指向你的本地服务。

## 典型工作流

### 表格数据：composition / SMILES

1. 先用 `summarize_dataset` 看列信息
2. 用 `fit_feature_pipeline` 在训练集上生成并保存 pipeline
3. 用 `transform_with_pipeline` 对测试集或预测集复用同一个 `pipeline_id`
4. 用 `inspect_pipeline` / `list_pipelines` 回看历史 pipeline

示例：

```python
from src.materials_feature_engineering_mcp.pipeline_runner import (
    summarize_dataset,
    fit_feature_pipeline,
    transform_with_pipeline,
)

summary = summarize_dataset("train.csv")

fit_result = fit_feature_pipeline(
    data_path="train.csv",
    target_column="target",
    task_type="regression",
    composition_columns=["composition"],
    smiles_columns=["smiles"],
    passthrough_columns=["temperature"],
    pipeline_config={
        "featurization": {
            "composition_feature_types": ["element_amount"],
            "smiles_feature_types": ["descriptors", "morgan"],
        },
        "selection": {"method": "rfecv"},
    },
)

pipeline_id = fit_result["pipeline_id"]

predict_result = transform_with_pipeline(
    pipeline_id=pipeline_id,
    data_path="predict.csv",
)
```

### CIF 数据：`zip + metadata table`

1. `zip` 中放一批 `.cif` 文件
2. metadata 表必须包含 `cif_filename` 列，并与压缩包内文件名精确对齐
3. 用 `fit_cif_pipeline` 保存训练时的 CIF pipeline
4. 用 `transform_with_cif_pipeline` 复用同一个 `pipeline_id`

示例：

```python
from src.materials_feature_engineering_mcp.cif_pipeline_runner import (
    summarize_cif_archive,
    fit_cif_pipeline,
    transform_with_cif_pipeline,
)

summary = summarize_cif_archive(
    structure_archive="train_cifs.zip",
    metadata_table="train_labels.csv",
    cif_filename_column="cif_filename",
)

fit_result = fit_cif_pipeline(
    structure_archive="train_cifs.zip",
    metadata_table="train_labels.csv",
    cif_filename_column="cif_filename",
    target_column="band_gap",
    task_type="regression",
    pipeline_config={
        "featurization": {
            "structure_feature_types": ["basic", "symmetry", "density", "complexity"],
            "composition_feature_types": ["element_amount"],
        },
        "selection": {"method": "rfecv"},
    },
)

pipeline_id = fit_result["pipeline_id"]

predict_result = transform_with_cif_pipeline(
    pipeline_id=pipeline_id,
    structure_archive="predict_cifs.zip",
    metadata_table="predict_metadata.csv",
    cif_filename_column="cif_filename",
)
```

## 本地持久化目录

### 表格 pipeline

- `data/pipelines/<pipeline_id>/metadata.json`
- `data/pipelines/<pipeline_id>/training_features.csv`
- `data/pipelines/<pipeline_id>/transforms/*.csv`

### CIF pipeline

- `data/cif_pipelines/<pipeline_id>/metadata.json`
- `data/cif_pipelines/<pipeline_id>/training_features.csv`
- `data/cif_pipelines/<pipeline_id>/transforms/*.csv`

## `requirements.txt` 说明

仓库现在包含一个真实环境快照版的 `requirements.txt`。

它不是手写的“顶层依赖列表”，而是基于当前 uv 管理的 `.venv` 直接冻结出来的版本清单，适合：

- 用 `pip` 复现当前环境
- 做部署排查
- 和 `uv.lock` 做差异核对

生成基准环境：

- `uv 0.10.10`
- `Python 3.12.13`
- 命令来源：`uv pip freeze --python ./.venv/bin/python`

说明：

- 开发时的主依赖来源仍然是 `pyproject.toml + uv.lock`
- `requirements.txt` 更像一个“当前真实环境快照”

如果你想直接用 `pip` 按这个快照安装：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 验证命令

建议在改动后至少跑下面两条：

```bash
./.venv/bin/python -m unittest tests.test_regressions tests.test_cif_pipeline -v
./.venv/bin/python -m compileall src single_port_server.py test_html_report.py tests/test_regressions.py tests/test_cif_pipeline.py
```

## 仓库里和文档相关的关键文件

- [README.md](/Users/songlin/Desktop/Code/MCP_FE_Tool/README.md)
- [requirements.txt](/Users/songlin/Desktop/Code/MCP_FE_Tool/requirements.txt)
- [pyproject.toml](/Users/songlin/Desktop/Code/MCP_FE_Tool/pyproject.toml)
- [uv.lock](/Users/songlin/Desktop/Code/MCP_FE_Tool/uv.lock)
- [__main__.py](/Users/songlin/Desktop/Code/MCP_FE_Tool/__main__.py)
- [single_port_server.py](/Users/songlin/Desktop/Code/MCP_FE_Tool/single_port_server.py)
