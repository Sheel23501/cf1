"""
llm_baseline.py — Zero-shot LLM Ranking Baseline using Gemini API.

Strategy:
  For each sampled user:
    1. Build their watch history (movie titles from training data)
    2. Sample 20 candidate items: 1 true test item + 19 random negatives
    3. Ask Gemini to rank all 20 by predicted user preference
    4. Evaluate HR@K and NDCG@K on the ranking

Output:
    Prints HR@5, HR@10, HR@20 and NDCG@5, NDCG@10, NDCG@20
    Appends a row to results/tables/comparison_ml100k.csv
"""

import os
import sys
import time
import json
import random
import numpy as np
from collections import defaultdict
from google import genai

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.data_loader import BipartiteGraphLoader
from src.utils.metrics import hit_rate_at_k, ndcg_at_k

# Using the provided Gemini API key
GENAI_API_KEY = "AIzaSyD5XjJE0Q5OhGr5UI6pAK8QRg8zceKPHa8"
client = genai.Client(api_key=GENAI_API_KEY)


# ============================================================
# Config
# ============================================================

FILEPATH      = "data/movielens-1m/ml-100k 4/u.data"
ITEM_NAMES    = "data/movielens-1m/ml-100k 4/u.item"   # movie titles file
N_USERS       = 10       # users to sample (reduced to avoid API quota limits)
N_CANDIDATES  = 20       # candidates per user (1 positive + 19 negatives)
K_VALUES      = [5, 10, 20]
SLEEP_BETWEEN = 5.0      # Increased buffer to avoid rate limits
SEED          = 42


# ============================================================
# Load movie titles (ML-100K u.item format)
# ============================================================

def load_movie_titles(path):
    """
    Returns dict: original_item_id -> movie title string.
    ML-100K u.item is pipe-separated: id|title|...
    Falls back to "Movie {id}" if file not found.
    """
    titles = {}
    if not os.path.exists(path):
        print(f"  Warning: {path} not found — using generic movie names.")
        return titles
    with open(path, encoding='latin-1') as f:
        for line in f:
            parts = line.strip().split('|')
            if len(parts) >= 2:
                titles[int(parts[0])] = parts[1]
    return titles


# ============================================================
# Build prompt
# ============================================================

def build_prompt(watch_history_titles, candidate_titles):
    """
    Builds a zero-shot ranking prompt for Gemini.
    Returns the prompt string.
    """
    history_str = "\n".join(f"  - {t}" for t in watch_history_titles)
    candidates_str = "\n".join(f"  {i+1}. {t}" for i, t in enumerate(candidate_titles))

    prompt = f"""You are a movie recommendation system.

A user has watched and enjoyed the following movies:
{history_str}

Based ONLY on this watch history, rank the following {len(candidate_titles)} candidate movies from most to least likely to be enjoyed by this user.

Candidate movies:
{candidates_str}

Reply with ONLY a JSON array of the candidate numbers in ranked order (most likely first).
Example format: [3, 1, 7, 2, ...]

Your ranking:"""
    return prompt


# ============================================================
# Parse LLM response
# ============================================================

def parse_ranking(response_text, n_candidates):
    """
    Extracts the ranked list of 1-based indices from Gemini's response.
    Falls back to random order if parsing fails.
    """
    text = response_text.strip()
    # Find JSON array in response
    start = text.find('[')
    end = text.rfind(']')
    if start != -1 and end != -1:
        try:
            ranked = json.loads(text[start:end+1])
            # Convert to 0-based indices, validate range
            ranked_0 = [int(r) - 1 for r in ranked if 1 <= int(r) <= n_candidates]
            # Fill in any missing indices
            seen = set(ranked_0)
            missing = [i for i in range(n_candidates) if i not in seen]
            return ranked_0 + missing
        except (json.JSONDecodeError, ValueError):
            pass

    # Fallback: try to extract numbers from text
    import re
    nums = re.findall(r'\b(\d+)\b', text)
    ranked = []
    seen = set()
    for n in nums:
        idx = int(n) - 1
        if 0 <= idx < n_candidates and idx not in seen:
            ranked.append(idx)
            seen.add(idx)
    missing = [i for i in range(n_candidates) if i not in seen]
    return ranked + missing


# ============================================================
# Main evaluation
# ============================================================

def main():
    random.seed(SEED)
    np.random.seed(SEED)

    # ---- Load data ----
    print(f"Loading data from {FILEPATH}...")
    loader = BipartiteGraphLoader(FILEPATH, threshold=1.0)
    loader.load_raw_csv(FILEPATH)

    # Build test dict: user -> set of test item ids (remapped)
    test_dict = defaultdict(set)
    for u, i in loader.test_data:
        test_dict[int(u)].add(int(i))

    # Build train dict: user -> set of train item ids
    train_dict = defaultdict(set)
    for u, i in loader.train_data:
        train_dict[int(u)].add(int(i))

    # ---- Load movie titles ----
    raw_titles = load_movie_titles(ITEM_NAMES)

    def get_title(remapped_id):
        if raw_titles:
            original_id = remapped_id + 1  # approximate — good enough for prompting
            return raw_titles.get(original_id, f"Movie {remapped_id}")
        return f"Movie {remapped_id}"

    # ---- Sample users ----
    eligible = [u for u, items in test_dict.items() if len(items) >= 1 and len(train_dict[u]) >= 3]
    if len(eligible) < N_USERS:
        sampled_users = eligible
    else:
        sampled_users = random.sample(eligible, N_USERS)

    print(f"Evaluating {len(sampled_users)} users with {N_CANDIDATES} candidates each...")
    print(f"Using Gemini-2.5-flash via google-genai SDK...\n")

    all_items = list(range(loader.n_items))
    results = {f"HR@{k}": [] for k in K_VALUES}
    results.update({f"NDCG@{k}": [] for k in K_VALUES})

    failed = 0

    for idx, user in enumerate(sampled_users):
        test_items = test_dict[user]
        train_items = train_dict[user]

        pos_item = random.choice(list(test_items))

        all_neg = [i for i in all_items if i not in train_items and i not in test_items]
        neg_items = random.sample(all_neg, min(N_CANDIDATES - 1, len(all_neg)))

        candidates = [pos_item] + neg_items
        random.shuffle(candidates)

        history_titles = [get_title(i) for i in list(train_items)[:15]]  
        candidate_titles = [get_title(i) for i in candidates]
        prompt = build_prompt(history_titles, candidate_titles)

        try:
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt,
            )
            raw_text = response.text
            ranked_candidate_indices = parse_ranking(raw_text, len(candidates))

            ranked_items = np.array([candidates[i] for i in ranked_candidate_indices])

            eval_test_set = {pos_item}
            for k in K_VALUES:
                results[f"HR@{k}"].append(hit_rate_at_k(ranked_items, eval_test_set, k))
                results[f"NDCG@{k}"].append(ndcg_at_k(ranked_items, eval_test_set, k))

            print(f"  [{idx+1}/{len(sampled_users)}] User {user} processed. HR@10: {np.mean(results['HR@10']):.4f}")

        except Exception as e:
            print(f"  User {user} failed: {e}")
            failed += 1
            for k in K_VALUES:
                results[f"HR@{k}"].append(0.0)
                results[f"NDCG@{k}"].append(0.0)

        time.sleep(SLEEP_BETWEEN)

    final = {name: float(np.mean(vals)) for name, vals in results.items()}

    print(f"\n{'='*60}")
    print(f"  LLM Baseline Results (n={len(sampled_users)}, {N_CANDIDATES} candidates)")
    print(f"  Failed calls: {failed}")
    print(f"{'='*60}")
    for k in K_VALUES:
        print(f"  HR@{k}: {final[f'HR@{k}']:.4f}   NDCG@{k}: {final[f'NDCG@{k}']:.4f}")
    print(f"{'='*60}\n")

    # ---- Append to comparison CSV ----
    csv_path = "results/tables/comparison_ml100k.csv"
    if os.path.exists(csv_path):
        with open(csv_path, 'a') as f:
            vals = ",".join(
                f"{final[f'HR@{k}']:.4f},{final[f'NDCG@{k}']:.4f}" for k in K_VALUES
            )
            f.write(f"LLM-ZeroShot,{vals}\n")
        print(f"Row appended to {csv_path}")
    else:
        print(f"Note: {csv_path} not found — skipping CSV append.")

    print("\nDone!")


if __name__ == "__main__":
    main()
