"""
Evaluation Metrics for Implicit Feedback Recommendation.

Implements Hit Rate (HR@K) and NDCG@K with binary relevance,
which are the two required metrics for the CF course project.
"""

from __future__ import annotations

import numpy as np
from typing import List, Set


def hit_rate_at_k(ranked_items: np.ndarray, test_items: Set[int], k: int) -> float:
    """
    Hit Rate @ K (HR@K)
    
    Returns 1.0 if ANY test item appears in the top-K ranked items, else 0.0.
    
    Args:
        ranked_items: Array of item IDs sorted by predicted score (descending)
        test_items: Set of ground-truth item IDs the user interacted with in test
        k: Number of top items to consider
    
    Returns:
        1.0 if hit, 0.0 if miss
    """
    top_k = set(ranked_items[:k].tolist())
    return 1.0 if len(top_k & test_items) > 0 else 0.0


def ndcg_at_k(ranked_items: np.ndarray, test_items: Set[int], k: int) -> float:
    """
    Normalized Discounted Cumulative Gain @ K (NDCG@K)
    
    Uses binary relevance: relevant = 1 if item is in test_items, else 0.
    
    Args:
        ranked_items: Array of item IDs sorted by predicted score (descending)
        test_items: Set of ground-truth item IDs
        k: Number of top items to consider
    
    Returns:
        NDCG score between 0.0 and 1.0
    """
    top_k = ranked_items[:k]
    
    # DCG: sum of 1/log2(rank+1) for relevant items
    dcg = 0.0
    for i, item in enumerate(top_k):
        if int(item) in test_items:
            dcg += 1.0 / np.log2(i + 2)  # +2 because rank is 1-indexed
    
    # IDCG: best possible DCG if all test items were ranked at the top
    n_relevant = min(len(test_items), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(n_relevant))
    
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_model(
    model_scores: np.ndarray,
    test_interactions: dict[int, Set[int]],
    train_interactions: dict[int, Set[int]],
    k_values: List[int] = [5, 10, 20],
) -> dict[str, float]:
    """
    Full evaluation pipeline for a recommendation model.
    
    For each user:
    1. Get predicted scores for all items
    2. Mask out items the user already interacted with in training
    3. Rank remaining items by score
    4. Compute HR@K and NDCG@K against test items
    
    Args:
        model_scores: (n_users, n_items) matrix of predicted scores
        test_interactions: {user_id: set of test item IDs}
        train_interactions: {user_id: set of train item IDs}
        k_values: List of K values to evaluate at
    
    Returns:
        Dictionary with metric names as keys and averaged scores as values
        e.g., {"HR@5": 0.312, "HR@10": 0.445, "NDCG@5": 0.201, ...}
    """
    metrics = {f"HR@{k}": [] for k in k_values}
    metrics.update({f"NDCG@{k}": [] for k in k_values})
    
    for user_id, test_items in test_interactions.items():
        if len(test_items) == 0:
            continue
        
        scores = model_scores[user_id].copy()
        
        # Mask training items (set to -inf so they won't be recommended)
        if user_id in train_interactions:
            for item_id in train_interactions[user_id]:
                scores[item_id] = -np.inf
        
        # Rank items by predicted score (descending)
        ranked_items = np.argsort(-scores)
        
        # Compute metrics at each K
        for k in k_values:
            metrics[f"HR@{k}"].append(hit_rate_at_k(ranked_items, test_items, k))
            metrics[f"NDCG@{k}"].append(ndcg_at_k(ranked_items, test_items, k))
    
    # Average across all users
    return {name: float(np.mean(values)) for name, values in metrics.items()}


def print_results(results: dict[str, float], model_name: str = "Model") -> None:
    """Pretty-print evaluation results."""
    print(f"\n{'='*50}")
    print(f"  Results for: {model_name}")
    print(f"{'='*50}")
    print(f"  {'Metric':<12} {'Value':>10}")
    print(f"  {'-'*22}")
    for metric, value in sorted(results.items()):
        print(f"  {metric:<12} {value:>10.4f}")
    print(f"{'='*50}\n")
