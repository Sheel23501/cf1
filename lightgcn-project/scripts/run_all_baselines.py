"""
run_all_baselines.py — Trains and evaluates ALL models on ML-100K.

Outputs a comparison table with HR@5, HR@10, HR@20, NDCG@5, NDCG@10, NDCG@20
for: MostPop, ItemKNN, BPR-MF, and LightGCN.
"""

import sys
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    # Make package imports work when script is run directly.
    sys.path.append(str(SRC_ROOT))

from lightgcn_project.data.data_loader import BipartiteGraphLoader, BPRDataset
from lightgcn_project.evaluation.metrics import hit_rate_at_k, ndcg_at_k
from lightgcn_project.models.mostpop import MostPopular
from lightgcn_project.models.itemknn import ItemKNN
from lightgcn_project.models.bprmf import BPRMF
from lightgcn_project.models.lightgcn import LightGCN


def resolve_data_path():
    """Resolve dataset path from env var or standard project locations."""
    env_path = os.getenv("LIGHTGCN_DATA_PATH")
    candidates = [
        Path(env_path) if env_path else None,
        PROJECT_ROOT / "data" / "raw" / "u.data",
    ]

    for candidate in candidates:
        if candidate and candidate.exists():
            # Prefer env override first, then project default path.
            return candidate

    raise FileNotFoundError(
        "Dataset file not found. Set LIGHTGCN_DATA_PATH or place u.data in data/raw/u.data"
    )


# ============================================================
# Shared evaluation function
# ============================================================

def evaluate_scores(scores_matrix, test_dict, train_user_item_net, k_values=[5, 10, 20]):
    """
    Given a (n_users, n_items) score matrix, evaluate HR@K and NDCG@K.
    Masks out training items before ranking.
    """
    results = {f"HR@{k}": [] for k in k_values}
    results.update({f"NDCG@{k}": [] for k in k_values})

    for user, test_items in test_dict.items():
        if len(test_items) == 0:
            continue

        scores = scores_matrix[user].copy()

        # Prevent recommending already-seen train items during evaluation.
        train_items = np.where(train_user_item_net[user].toarray()[0] == 1)[0]
        scores[train_items] = -np.inf

        # Rank items by descending score.
        ranked = np.argsort(-scores)

        for k in k_values:
            results[f"HR@{k}"].append(hit_rate_at_k(ranked, test_items, k))
            results[f"NDCG@{k}"].append(ndcg_at_k(ranked, test_items, k))

    return {name: float(np.mean(vals)) for name, vals in results.items()}


def build_test_dict(test_data):
    """Convert test array to {user: set(items)} dict."""
    test_dict = defaultdict(set)
    for u, i in test_data:
        test_dict[int(u)].add(int(i))
    return dict(test_dict)


# ============================================================
# BPR Loss (shared by BPR-MF and LightGCN)
# ============================================================

class BPRLoss(nn.Module):
    def __init__(self, decay=1e-4):
        super().__init__()
        self.decay = decay

    def forward(self, pos_scores, neg_scores, u_emb, pos_emb, neg_emb):
        loss = -torch.mean(nn.functional.logsigmoid(pos_scores - neg_scores))
        reg = (1/2) * (u_emb.norm(2).pow(2) +
                       pos_emb.norm(2).pow(2) +
                       neg_emb.norm(2).pow(2)) / float(len(pos_scores))
        return loss + self.decay * reg


# ============================================================
# Train a BPR-based model (works for both BPR-MF and LightGCN)
# ============================================================

def train_bpr_model(model, bpr_loader, device, epochs=100, lr=0.001, decay=1e-4):
    """Train a model using BPR loss."""
    loss_fn = BPRLoss(decay=decay)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for batch_u, batch_pos, batch_neg in bpr_loader:
            batch_u = batch_u.long().to(device)
            batch_pos = batch_pos.long().to(device)
            batch_neg = batch_neg.long().to(device)

            pos_scores, neg_scores, u_emb, pos_emb, neg_emb = model(batch_u, batch_pos, batch_neg)
            loss = loss_fn(pos_scores, neg_scores, u_emb, pos_emb, neg_emb)

            # Standard optimization step.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if epoch % 20 == 0:
            print(f"    Epoch {epoch:03d} | Loss: {total_loss / len(bpr_loader):.4f}")


def get_neural_model_scores(model, n_users, device, batch_size=1024):
    """Get full score matrix from a PyTorch model."""
    model.eval()
    all_scores = []

    # Disable gradients for faster and memory-safe full-matrix scoring.
    with torch.no_grad():
        for i in range(0, n_users, batch_size):
            end = min(i + batch_size, n_users)
            users_t = torch.arange(i, end).long().to(device)

            if hasattr(model, 'get_users_rating'):
                scores = model.get_users_rating(users_t)
            else:
                scores = model.get_all_ratings(users_t)

            all_scores.append(scores.cpu().numpy())

    return np.vstack(all_scores)


# ============================================================
# Main comparison
# ============================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available()
                          else "cpu")
    print(f"Device: {device}\n")

    # ---- Load Data ----
    # Build graph structures once and reuse across all models.
    filepath = resolve_data_path()
    loader = BipartiteGraphLoader(str(filepath), threshold=1.0)
    loader.load_raw_csv(str(filepath))

    test_dict = build_test_dict(loader.test_data)
    bpr_dataset = BPRDataset(loader, sampling_strategy="mixed", popular_prob=0.7)
    bpr_loader = DataLoader(bpr_dataset, batch_size=2048, shuffle=True, drop_last=True)

    all_results = {}
    k_values = [5, 10, 20]

    # =========================================
    # 1. MostPop
    # =========================================
    print("=" * 50)
    print("Training: MostPop")
    print("=" * 50)
    t0 = time.time()

    pop = MostPopular()
    pop.fit(loader.train_data, loader.n_items)
    pop_scores = pop.predict(list(range(loader.n_users)), loader.n_items)
    all_results["MostPop"] = evaluate_scores(pop_scores, test_dict, loader.user_item_net, k_values)

    print(f"  Done in {time.time()-t0:.1f}s")
    for k, v in sorted(all_results["MostPop"].items()):
        print(f"    {k}: {v:.4f}")

    # =========================================
    # 2. ItemKNN
    # =========================================
    print("\n" + "=" * 50)
    print("Training: ItemKNN (k=50)")
    print("=" * 50)
    t0 = time.time()

    knn = ItemKNN(k=50)
    knn.fit(loader.train_data, loader.n_users, loader.n_items)
    knn_scores = knn.predict(list(range(loader.n_users)), loader.n_items)
    all_results["ItemKNN"] = evaluate_scores(knn_scores, test_dict, loader.user_item_net, k_values)

    print(f"  Done in {time.time()-t0:.1f}s")
    for k, v in sorted(all_results["ItemKNN"].items()):
        print(f"    {k}: {v:.4f}")

    # =========================================
    # 3. BPR-MF
    # =========================================
    print("\n" + "=" * 50)
    print("Training: BPR-MF (100 epochs)")
    print("=" * 50)
    t0 = time.time()

    bprmf = BPRMF(loader.n_users, loader.n_items, latent_dim=64).to(device)
    train_bpr_model(bprmf, bpr_loader, device, epochs=100, lr=0.001, decay=1e-4)
    bprmf_scores = get_neural_model_scores(bprmf, loader.n_users, device)
    all_results["BPR-MF"] = evaluate_scores(bprmf_scores, test_dict, loader.user_item_net, k_values)

    print(f"  Done in {time.time()-t0:.1f}s")
    for k, v in sorted(all_results["BPR-MF"].items()):
        print(f"    {k}: {v:.4f}")

    # =========================================
    # 4. LightGCN
    # =========================================
    print("\n" + "=" * 50)
    print("Training: LightGCN (K=1 layer, dim=128, 8 epochs)")
    print("=" * 50)
    t0 = time.time()

    config = {'latent_dim': 128, 'n_layers': 1}
    lgcn = LightGCN(loader.n_users, loader.n_items, loader.norm_adj.to(device), config).to(device)
    train_bpr_model(lgcn, bpr_loader, device, epochs=8, lr=0.005, decay=1e-4)
    lgcn_scores = get_neural_model_scores(lgcn, loader.n_users, device)
    all_results["LightGCN"] = evaluate_scores(lgcn_scores, test_dict, loader.user_item_net, k_values)

    print(f"  Done in {time.time()-t0:.1f}s")
    for k, v in sorted(all_results["LightGCN"].items()):
        print(f"    {k}: {v:.4f}")

    # =========================================
    # Print Final Comparison Table
    # =========================================
    print("\n\n")
    print("=" * 80)
    print("  FINAL RESULTS TABLE — MovieLens 100K (Binarized Implicit Feedback)")
    print("=" * 80)

    header = f"{'Model':<12}"
    for k in k_values:
        header += f"  {'HR@'+str(k):>8}  {'NDCG@'+str(k):>8}"
    print(header)
    print("-" * 80)

    model_order = ["MostPop", "ItemKNN", "BPR-MF", "LightGCN"]

    for model_name in model_order:
        r = all_results[model_name]
        row = f"{model_name:<12}"
        for k in k_values:
            row += f"  {r[f'HR@{k}']:>8.4f}  {r[f'NDCG@{k}']:>8.4f}"
        print(row)

    print("=" * 80)

    # ---- Best accuracy summary ----
    print("\nBest Accuracy Summary")
    print("-" * 80)
    best_hr20_model = max(model_order, key=lambda m: all_results[m]["HR@20"])
    best_ndcg20_model = max(model_order, key=lambda m: all_results[m]["NDCG@20"])
    print(f"Best HR@20: {best_hr20_model} ({all_results[best_hr20_model]['HR@20']:.4f})")
    print(f"Best NDCG@20: {best_ndcg20_model} ({all_results[best_ndcg20_model]['NDCG@20']:.4f})")

    # ---- Save to CSV ----
    # Persist table for reporting and reproducibility.
    output_dir = PROJECT_ROOT / "outputs" / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / "comparison_ml100k.csv"

    with open(output_csv, "w") as f:
        f.write("Model," + ",".join(f"HR@{k},NDCG@{k}" for k in k_values) + "\n")
        for model_name in model_order:
            r = all_results[model_name]
            vals = ",".join(f"{r[f'HR@{k}']:.4f},{r[f'NDCG@{k}']:.4f}" for k in k_values)
            f.write(f"{model_name},{vals}\n")

    print(f"\nResults saved to {output_csv}")


if __name__ == "__main__":
    main()
