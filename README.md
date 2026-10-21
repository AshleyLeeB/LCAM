# LCAM

在 **AMLWorld** 图数据上的少样本反洗钱（AML）节点分类实现：**ProtoMAML + 加权 GraphConv（GCN）**，并通过 **LCAM**（layer-wise 内外梯度门控混合）调节 meta 更新。

本仓库为论文/复现用精简版：单数据集 **AMLWorld**、单方法 **LCAM**、骨干 **GraphConv**（加权 MPNN，`model_types: ['GCN']`）。

## 环境

- Python **3.8**（推荐）
- **CUDA** 可选；CPU 可跑但较慢
- 依赖见 [`requirements.txt`](requirements.txt)

PyTorch 与 DGL 需版本匹配。示例（CUDA 12.4）：

```bash
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124
pip install dgl -f https://data.dgl.ai/wheels/torch-2.4/cu124/repo.html
pip install -r requirements.txt
```

## 数据

数据根目录：`Data/AMLWorld/Data/`（亦可通过 [`paths.py`](paths.py) 中的 `AMLWORLD` 修改）。

训练脚本会读取：

| 文件 | 说明 |
|------|------|
| `features_5.npy` | 节点特征 |
| `new_graph_guiyihua.pkl` | DGL 图（需 `time_id` 与边属性 `amount`） |
| `label.pkl` | 节点标签 |
| `train_1.csv` … `train_6.csv` | 6 个 meta-train 划分 |
| `val.csv` / `test.csv` | 验证 / 测试划分 |

若仓库中仅有 CSV 而无 `.npy`/`.pkl`，需按 AMLWorld 官方流程预处理后再训练。

## 快速开始

```bash
git clone <your-repo-url>
cd LCAM-github

# 只跑一个 seed：start 与 end 设为相同值
python LCAM.py --start_seed 45 --end_seed 45 --no_early_stop

# 多种子（闭区间，含 43、44、45、46）
python LCAM.py --start_seed 43 --end_seed 46 --no_early_stop
```

与主项目 [`LCAM/实验记录.txt`](../LCAM/实验记录.txt) 中 **§2 PUFA/AMLWorld few-shot** 的 **mainpath** 设置一致：

| 项 | 主仓库 `LCAM-1.py --only mainpath` | 本仓库 `LCAM.py` |
|----|-----------------------------------|------------------|
| 方法 | `V3_blend`（LCAM 门控） | `LCAM`（同名逻辑） |
| 门控目标 | `gate_objective=mainpath`，`aux_lambda=0.01` | 同左（见 `config.py`） |
| GNN | 全参数 meta 训练（unfrozen） | `run_lcam` 中 `freeze_gnn=False` |
| 训练 | `num_epochs=10`，`--no_early_stop` 跑满 | 同左 |
| 超参 | `n_way=2`, `k_spt=10`, `k_qry=24`, hops=2, layers=2 等 | 同左（`config.py`） |
| 数据 | `amlworld` / `pufa` | 仅 **AMLWorld**（`Data/AMLWorld/Data/`） |
| 骨干 | `GCN_3`（`model_types: GCN_3`） | **`GraphConv`（`GCN`）** — 精简复现版差异 |

等价命令示例（只跑 AMLWorld、seed 45）：

```bash
# 主仓库
python LCAM-1.py --datasets amlworld --only mainpath --start_seed 45 --end_seed 45 --no_early_stop

# 本仓库
python LCAM.py --start_seed 45 --end_seed 45 --no_early_stop
```

默认写入 `results/`：日志、汇总 CSV，以及每个 seed 的子目录 `stability_amlworld_lcam_seed_<seed>/`。

## 命令行

### `LCAM.py`（推荐入口）

| 参数 | 含义 | 默认 |
|------|------|------|
| `--start_seed` / `--end_seed` | 随机种子区间（闭区间；两者相同则只跑 1 个 seed） | `10` / `18` |
| `--no_early_stop` | 关闭早停，跑满 `num_epochs` | 关闭 |
| `--min_epochs` | 早停前至少训练的 epoch 数 | `0` |
| `--output_root` | 结果根目录 | `results/` |

### `run_lcam.py`（底层多种子脚本）

与 `LCAM.py` 等价逻辑，可直接调用，例如：

```bash
python run_lcam.py --dataset amlworld --only lcam \
  --start_seed 43 --end_seed 46 --no_early_stop --output_root results
```

`python train.py` 会转发到 `run_lcam.main()`。

## 配置

超参与数据规模默认值在 [`config.py`](config.py)，数据集路径与采样上限在 [`dataset_configs.py`](dataset_configs.py)。常用项：

- **模型**：`model_types: ['GCN']`（GNN++ 加权 GraphConv）；`run_lcam` 中 `freeze_gnn=False`（GNN 与分类头一并 meta 训练）
- **LCAM**：`modulation_version: 'LCAM'`，`blend_layers: 'all'`，`modulation_aux_weight: 0.01`
- **子图**：`neighborhood_hops: 2`，`num_gnn_layers: 2`，`max_subgraph_nodes: 100`
- **训练**：`num_epochs: 10`，`patience: 4`（可被 `--no_early_stop` 覆盖）

## 仓库结构

```
LCAM.py              # 训练入口
run_lcam.py          # 多种子实验与 CSV 汇总
train.py             # 单次实验训练循环
meta.py              # ProtoMAML + LCAM 梯度调制
learner.py           # GraphConv + 分类头
gradient_modulation.py
subgraph_*.py        # 子图采样与缓存
config.py / dataset_configs.py / paths.py
Data/AMLWorld/Data/
results/             # 运行输出
```
