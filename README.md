# LCAM
LCAM introduces a layer-wise gated fusion mechanism to adaptively integrate inner-loop and outer-loop gradients, capturing potentially useful gradient conflict while balancing task-specific adaptation and cross-task generalization.

## Environment

- Python **3.8** (recommended)
- **CUDA** optional; CPU works but is slower
- Dependencies: [`requirements.txt`](requirements.txt)

PyTorch and DGL versions must match. Example (CUDA 12.4):

```bash
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124
pip install dgl -f https://data.dgl.ai/wheels/torch-2.4/cu124/repo.html
pip install -r requirements.txt
```

## Data

Data root: `Data/AMLWorld/Data/` (override via `AMLWORLD` in [`paths.py`](paths.py)). The data link: 

Training expects:

| File | Description |
|------|-------------|
| `features_5.npy` | Node features |
| `new_graph_guiyihua.pkl` | DGL graph (`time_id`, edge attr `amount`) |
| `label.pkl` | Node labels |
| `train_1.csv` … `train_6.csv` | Six meta-train splits |
| `val.csv` / `test.csv` | Validation / test splits |



## Quick start

```bash
git clone <your-repo-url>
cd LCAM-github

# Single seed: set start and end to the same value
python LCAM.py --start_seed 45 --end_seed 45 --no_early_stop

# Multiple seeds (inclusive range, e.g. 43–46)
python LCAM.py --start_seed 43 --end_seed 46 --no_early_stop
```



| Item | Main repo `LCAM-1.py --only mainpath` | This repo `LCAM.py` |
|------|----------------------------------------|---------------------|
| Method | `V3_blend` (LCAM gating) | `LCAM` (same logic) |
| Gate objective | `gate_objective=mainpath`, `aux_lambda=0.01` | Same (`config.py`) |
| GNN | Full meta-training (unfrozen) | `freeze_gnn=False` in `run_lcam` |
| Training | `num_epochs=10`, `--no_early_stop` for full run | Same |
| Hyperparams | `n_way=2`, `k_spt=10`, `k_qry=24`, hops=2, layers=2, etc. | Same (`config.py`) |
| Data | `amlworld` / `pufa` | **AMLWorld** only (`Data/AMLWorld/Data/`) |
| Backbone | `GCN_3` (`GraphConv_3`) | **`GraphConv` (`GCN`)** — slim-repo difference |

Outputs go under `results/`: logs, summary CSVs, and per-seed folders `stability_amlworld_lcam_seed_<seed>/`.


## Configuration

Defaults live in [`config.py`](config.py); dataset paths and sampling caps in [`dataset_configs.py`](dataset_configs.py). Common knobs:

- **Model**: `model_types: ['GCN']` (GNN++ weighted GraphConv); `run_lcam` sets `freeze_gnn=False` (GNN + head meta-trained)
- **LCAM**: `modulation_version: 'LCAM'`, `blend_layers: 'all'`, `gate_objective: 'mainpath'`, `aux_lambda: 0.01`
- **Subgraphs**: `neighborhood_hops: 2`, `num_gnn_layers: 2`, `max_subgraph_nodes: 100`
- **Training**: `num_epochs: 10`, `patience: 4` (overridden by `--no_early_stop`)

## Layout

```
LCAM.py              # Training entry
run_lcam.py          # Multi-seed runs + CSV summary
train.py             # Single-run training loop
meta.py              # ProtoMAML + LCAM gradient modulation
learner.py           # GraphConv + classifier head
gradient_modulation.py
mainpath.py          # Mainpath gate loss (Eq. 16)
subgraph_*.py        # Subgraph sampling + cache
config.py / dataset_configs.py / paths.py
Data/AMLWorld/Data/
results/             # Run outputs
```
