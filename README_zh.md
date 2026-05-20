# TopoStateGrid

[English](https://github.com/MingLeiZhou/TopoStateGrid/blob/main/README.md) | [中文](https://github.com/MingLeiZhou/TopoStateGrid/blob/main/README_zh.md) | [Português](https://github.com/MingLeiZhou/TopoStateGrid/blob/main/README_pt.md)

TopoStateGrid 是一种面向电力系统机器学习的、具有物理信息约束的图构建方法。它将电网拓扑、设备属性和运行状态变量转换为可直接用于机器学习的图数据集。

Python 包导入名：

```python
import topostategrid
```

TopoStateGrid 的目标是构建可复用的电力系统图数据流水线。它不包含 GNN 模型、不包含级联故障仿真器、不支持 `.mat` 文件，也不构建异构图。

## 构建的图

每个图样本表示为：

```text
G_t = (V, E, X_t, A_t, y_t)
```

其中：

- `V` 是母线节点。
- `E` 是线路和变压器等物理连接。
- `X_t` 是某个场景或时间戳下的节点特征。
- `A_t` 是某个场景或时间戳下的边特征。
- `y_t` 是可选标签。

输出是同构 bus-branch PyTorch Geometric `Data` 对象，包含：

```text
data.x
data.edge_index
data.edge_attr
data.y
data.network_id
data.sample_id
data.timestamp
data.scenario_id
data.contingency_id
data.node_feature_names
data.edge_feature_names
data.metadata
```

边会被存储为双向边，便于消息传递。不同输入源的元数据会保存为 JSON 字符串，避免 PyG `DataLoader` 在跨来源 batching 时因为 Python 字典结构不一致而失败。

## 安装

从 PyPI 安装：

```bash
python -m pip install TopoStateGrid
```

可选依赖：

```bash
python -m pip install "TopoStateGrid[pandapower]"
python -m pip install "TopoStateGrid[visual]"
python -m pip install "TopoStateGrid[test]"
```

从本地仓库安装：

```bash
git clone https://github.com/MingLeiZhou/TopoStateGrid.git
cd TopoStateGrid
python -m pip install -e ".[test,pandapower,visual]"
```

## 支持的输入

TopoStateGrid v1.1 支持：

| 输入源 | 状态 | 说明 |
| --- | --- | --- |
| OPFData JSON | 支持 | 使用已解压 JSON 样本和可用的求解状态字段。 |
| MATPOWER / PGLib `.m` | 支持 | 解析 `mpc.bus` 和 `mpc.branch`；静态 case 默认没有潮流结果。 |
| pandapower `net` | 支持，可选依赖 | 需要 `TopoStateGrid[pandapower]`；支持母线、线路和变压器。 |
| pandas DataFrame | 支持 | 适合自定义 bus/branch/state 表。 |
| CSV 表格 | 支持 | 基于 DataFrame 接口的便捷封装。 |

MATPOWER 解析器支持逗号或空格分隔的矩阵行、分号、`%` 注释、科学计数法、多行矩阵，以及 `mpc.branch = [ ];` 这类显式空矩阵。

## 特征定义

节点特征顺序：

```text
bus_status, bus_type, pd, qd, vm, va, vmax, vmin, normalized_demand
```

边特征顺序：

```text
component_type, r, x, b_from, b_to, rate_a, pf, qf, pt, qt, loading_ratio, outage_flag
```

缺失的可选数值字段会被安全填充为 0。对于 pandapower 线路，如果没有直接的 MVA 额定值，`rate_a` 会使用 `max_i_ka` 作为近似代理。

## 快速开始

从 pandas 表格构建图：

```python
import pandas as pd
from topostategrid import build_graph_from_tables

bus_df = pd.DataFrame({
    "bus_id": [1, 2, 3],
    "bus_type": [3, 1, 1],
    "pd": [0.0, 1.5, 0.8],
    "qd": [0.0, 0.4, 0.2],
})

branch_df = pd.DataFrame({
    "from_bus": [1, 2],
    "to_bus": [2, 3],
    "r": [0.01, 0.02],
    "x": [0.05, 0.06],
})

data = build_graph_from_tables(
    bus_df,
    branch_df,
    network_id="toy_3bus",
    sample_id="sample_0",
)
```

从 CSV 构建图：

```python
from topostategrid import build_graph_from_csv_tables

data = build_graph_from_csv_tables(
    "bus.csv",
    "branch.csv",
    network_id="toy_3bus",
)
```

从 pandapower 构建图：

```python
import pandapower as pp
from topostategrid import build_graph_from_pandapower

net = pp.create_empty_network()
b1 = pp.create_bus(net, vn_kv=110)
b2 = pp.create_bus(net, vn_kv=110)
b3 = pp.create_bus(net, vn_kv=110)

pp.create_ext_grid(net, b1)
pp.create_load(net, b2, p_mw=10.0, q_mvar=3.0)
pp.create_line_from_parameters(net, b1, b2, 1.0, 0.1, 0.2, 0.0, 0.4)
pp.create_line_from_parameters(net, b2, b3, 1.0, 0.1, 0.2, 0.0, 0.4)
pp.runpp(net)

data = build_graph_from_pandapower(net, network_id="pandapower_3bus")
```

跨来源 batching：

```python
from torch_geometric.loader import DataLoader

loader = DataLoader([graph_a, graph_b, graph_c], batch_size=3)
batch = next(iter(loader))
```

## 拓扑与运行状态

TopoStateGrid 明确区分：

- 静态拓扑：母线、线路、变压器、支路参数。
- 运行状态：负荷、电压、相角、支路潮流、线路负载率。

对于同一个网络，`edge_index` 可以保持不变，而 `data.x` 和 `data.edge_attr` 会随着场景或时间变化。这支持后续监督学习、自监督预训练、时序预测和跨拓扑评估。

## 时序窗口

```python
from topostategrid import make_temporal_windows

windows = make_temporal_windows(
    graphs,
    input_window=6,
    forecast_horizon=1,
    target="y",
)
```

如果所有时间戳都有效且可比较，时序工具会按时间戳排序；否则保留输入顺序。

## 标签

TopoStateGrid 可以附加临时压力代理标签：

```text
risk_score = max_line_loading_ratio
y_cls = 1 if max_line_loading_ratio > 1.0 else 0
y_reg = risk_score
```

该标签只用于图构建实验，不是真实级联故障标签。默认不会覆盖已有标签，除非传入 `overwrite=True`。

## 划分与归一化

支持的数据集划分：

- 随机划分
- 按时间划分
- Leave-One-Network-Out

LONO 适合跨拓扑泛化评估，例如在 `case14`、`case30`、`case57` 上训练，在 `case118` 上测试。

`FeatureNormalizer` 只在训练集上拟合统计量，然后用同一组统计量转换训练、验证和测试图，避免数据泄漏。

## 可视化

可选 GIF/MP4 渲染用于检查图序列：

```python
from topostategrid import render_graph_sequence

render_graph_sequence(
    graphs,
    "outputs/topostategrid_sequence.gif",
    node_value="vm",
    edge_value="loading_ratio",
)
```

该功能只可视化已有图样本，不做电网动态仿真。

大规模 pandapower 示例：

```bash
python examples/08_render_large_pandapower_gif.py
```

该脚本使用 pandapower `case300`，生成 300 节点图序列，并渲染 20 秒 GIF。

## 示例脚本

| 脚本 | 用途 |
| --- | --- |
| `examples/01_build_single_graph.py` | 构建一个本地 OPFData 图。 |
| `examples/02_build_multiple_state_graphs.py` | 构建多个 OPFData 场景图。 |
| `examples/03_create_temporal_windows.py` | 创建时序窗口。 |
| `examples/04_create_splits.py` | 创建随机、时间和 LONO 划分。 |
| `examples/05_build_from_tables.py` | 从 pandas 表格构建图。 |
| `examples/06_build_from_pandapower.py` | 从小型 pandapower 网络构建图。 |
| `examples/07_render_graph_animation.py` | 渲染小型图状态 GIF。 |
| `examples/08_render_large_pandapower_gif.py` | 从 pandapower `case300` 渲染 20 秒 GIF。 |

## 测试

```bash
python -m unittest discover -s tests -q
pytest -q
```

如果 matplotlib 默认缓存目录不可写，可以指定：

```bash
MPLCONFIGDIR=/private/tmp/topostategrid-mpl pytest -q
```

pandapower 可能提示未安装 `numba`，这只影响 pandapower 运行速度。

## 研究定位

TopoStateGrid 不是 PowerGraph 或 pandapower 的简单封装。

PowerGraph 可以作为参考数据集，pandapower 可以作为解析或仿真工具。TopoStateGrid 的核心输出是一个面向电力系统机器学习的、物理相关、状态相关、可选时间索引的图构建流水线。

## 限制

- 只支持同构 bus-branch 图。
- 不包含 GNN 模型。
- 不包含级联故障仿真器。
- 不支持 `.mat`。
- 不支持异构组件图。
- 不提供真实级联故障标签。
- pandapower 线路额定值映射在只有 `max_i_ka` 时是近似的。
- MP4 渲染需要 ffmpeg；GIF 渲染依赖 matplotlib、networkx 和 Pillow。
