# LCAM

Few-shot anti–money laundering (AML) **node classification** on **AMLWorld**: **ProtoMAML + weighted GraphConv (GCN)**, with **LCAM** (layer-wise inner/outer gradient gating) modulating meta-updates.

This repo is a slim reproduction bundle: **AMLWorld** only, **LCAM** only, backbone **GraphConv** (weighted MPNN, `model_types: ['GCN']`).

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

Data root: `Data/AMLWorld/Data/` (override via `AMLWORLD` in [`paths.py`](paths.py)).

Training expects:

| File | Description |
|------|-------------|
| `features_5.npy` | Node features |
| `new_graph_guiyihua.pkl` | DGL graph (`time_id`, edge attr `amount`) |
| `label.pkl` | Node labels |
| `train_1.csv` … `train_6.csv` | Six meta-train splits |
| `val.csv` / `test.csv` | Validation / test splits |

If you only have CSVs and no `.npy`/`.pkl`, run the official AMLWorld preprocessing first.

## Quick start

```bash
git clone <your-repo-url>
cd LCAM-github

# Single seed: set start and end to the same value
python LCAM.py --start_seed 45 --end_seed 45 --no_early_stop

# Multiple seeds (inclusive range, e.g. 43–46)
python LCAM.py --start_seed 43 --end_seed 46 --no_early_stop
```

Aligned with **mainpath** settings in the main project [`LCAM/实验记录.txt`](../LCAM/实验记录.txt) §2 (PUFA/AMLWorld few-shot):

| Item | Main repo `LCAM-1.py --only mainpath` | This repo `LCAM.py` |
|------|----------------------------------------|---------------------|
| Method | `V3_blend` (LCAM gating) | `LCAM` (same logic) |
| Gate objective | `gate_objective=mainpath`, `aux_lambda=0.01` | Same (`config.py`) |
| GNN | Full meta-training (unfrozen) | `freeze_gnn=False` in `run_lcam` |
| Training | `num_epochs=10`, `--no_early_stop` for full run | Same |
| Hyperparams | `n_way=2`, `k_spt=10`, `k_qry=24`, hops=2, layers=2, etc. | Same (`config.py`) |
| Data | `amlworld` / `pufa` | **AMLWorld** only (`Data/AMLWorld/Data/`) |
| Backbone | `GCN_3` (`GraphConv_3`) | **`GraphConv` (`GCN`)** — slim-repo difference |

Equivalent one-seed AMLWorld run:

```bash
# Main repo
python LCAM-1.py --datasets amlworld --only mainpath --start_seed 45 --end_seed 45 --no_early_stop

# This repo
python LCAM.py --start_seed 45 --end_seed 45 --no_early_stop
```

Outputs go under `results/`: logs, summary CSVs, and per-seed folders `stability_amlworld_lcam_seed_<seed>/`.

## CLI

### `LCAM.py` (recommended entry)

| Flag | Meaning | Default |
|------|---------|---------|
| `--start_seed` / `--end_seed` | Inclusive seed range (equal values → one seed) | `10` / `18` |
| `--no_early_stop` | Disable early stopping; train all `num_epochs` | off |
| `--min_epochs` | Minimum epochs before early stop can trigger | `0` |
| `--output_root` | Results root | `results/` |

### `run_lcam.py` (multi-seed driver)

Same behavior as `LCAM.py`, e.g.:

```bash
python run_lcam.py --dataset amlworld --only lcam \
  --start_seed 43 --end_seed 46 --no_early_stop --output_root results
```

`python train.py` forwards to `run_lcam.main()`.

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
