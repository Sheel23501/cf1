# CF1 Recommender Systems Project

Single source of documentation for this repository.

This project implements LightGCN for implicit-feedback recommendation and compares it with classic collaborative filtering baselines.

## Features

- LightGCN model for bipartite user-item graph recommendation
- Baselines: MostPop, ItemKNN, and BPR-MF
- Metrics: HR@K and NDCG@K
- Experiment scripts for:
	- single-model training
	- baseline comparison
	- ablation studies

## Repository Structure

```text
cf1/
├── README.md
├── requirements.txt
└── lightgcn-project/
		├── requirements.txt
		├── pyproject.toml
		├── scripts/
		│   ├── train_lightgcn.py
		│   ├── run_all_baselines.py
		│   └── ablation_study.py
		├── src/lightgcn_project/
		│   ├── data/
		│   ├── evaluation/
		│   └── models/
		├── data/
		│   ├── raw/
		│   └── processed/
		├── outputs/tables/
		├── docs/
		└── notebooks/
```

## Data Path

Scripts resolve dataset path in this order:

1. `LIGHTGCN_DATA_PATH` environment variable
2. `lightgcn-project/data/raw/u.data`

Recommended default location:

- [lightgcn-project/data/raw/u.data](lightgcn-project/data/raw/u.data)

Optional override:

```bash
export LIGHTGCN_DATA_PATH=/absolute/path/to/u.data
```

## Setup

```bash
cd /home/vikas/Documents/cf1/lightgcn-project
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run Commands

Run all baselines and generate comparison table:

```bash
python scripts/run_all_baselines.py
```

Train LightGCN only:

```bash
python scripts/train_lightgcn.py
```

Run ablation study:

```bash
python scripts/ablation_study.py
```

## Outputs

Generated tables are saved to:

- [lightgcn-project/outputs/tables/comparison_ml100k.csv](lightgcn-project/outputs/tables/comparison_ml100k.csv)
- [lightgcn-project/outputs/tables/ablation_ml100k.csv](lightgcn-project/outputs/tables/ablation_ml100k.csv)

## Notes

- Device fallback order: CUDA -> MPS -> CPU
- Objective: BPR loss with L2 regularization
- Root [requirements.txt](requirements.txt) is separate from [lightgcn-project/requirements.txt](lightgcn-project/requirements.txt)
