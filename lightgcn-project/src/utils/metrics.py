import numpy as np

def hit_rate_at_k(ranked_items, test_items, k):
    """
    Hit Rate @ K
    Returns 1.0 if any test item is in the top-k ranked items, else 0.0.
    """
    top_k = set(ranked_items[:k].tolist())
    return 1.0 if len(top_k & test_items) > 0 else 0.0

def ndcg_at_k(ranked_items, test_items, k):
    """
    NDCG @ K with binary relevance.
    """
    top_k = ranked_items[:k]
    dcg = sum(1.0 / np.log2(i + 2) for i, item in enumerate(top_k) if item in test_items)
    n_relevant = min(len(test_items), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(n_relevant))
    return dcg / idcg if idcg > 0 else 0.0
