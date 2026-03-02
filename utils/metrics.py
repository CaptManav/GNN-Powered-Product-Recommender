"""
Recommendation Evaluation Metrics

Standard metrics used in academic papers AND industry:
- Recall@K: fraction of relevant items retrieved in top K
- NDCG@K: normalized discounted cumulative gain (position-aware)
- Precision@K: fraction of top-K that are relevant
- Hit Rate@K: did any relevant item appear in top K?

K=10 and K=20 are standard reporting values.
"""

import numpy as np
import torch


def recall_at_k(recommended, relevant, k):
    """Recall@K = |recommended[:k] & relevant| / |relevant|"""
    if not relevant:
        return 0.0
    hits = len(set(recommended[:k]) & set(relevant))
    return hits / len(relevant)


def ndcg_at_k(recommended, relevant, k):
    """
    NDCG@K = DCG@K / IDCG@K
    
    DCG gives logarithmically discounted credit for relevant items.
    Position 1 gets full credit, position 2 gets credit/log2(2), etc.
    This penalizes putting relevant items far down the list.
    """
    relevant_set = set(relevant)
    dcg = 0.0
    for i, item in enumerate(recommended[:k]):
        if item in relevant_set:
            dcg += 1.0 / np.log2(i + 2)  # i+2 because log2(1)=0

    # Ideal DCG: all relevant items at the top
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / idcg if idcg > 0 else 0.0


def precision_at_k(recommended, relevant, k):
    """Precision@K = |recommended[:k] & relevant| / k"""
    hits = len(set(recommended[:k]) & set(relevant))
    return hits / k


def hit_rate_at_k(recommended, relevant, k):
    """Hit Rate@K = 1 if any relevant item in top-K else 0"""
    return float(bool(set(recommended[:k]) & set(relevant)))


def evaluate_model(model, edge_index, test_df, train_user_items,
                   num_users, num_items, k_list=[10, 20], batch_size=256):
    """
    Full evaluation loop over all test users.
    
    For each user:
    1. Get model's top-K recommendations (excluding training items)
    2. Compare against held-out test item
    3. Aggregate metrics
    
    Returns dict of metric_name -> value
    """
    model.eval()
    device = next(model.parameters()).device

    all_metrics = {f'Recall@{k}': [] for k in k_list}
    all_metrics.update({f'NDCG@{k}': [] for k in k_list})
    all_metrics.update({f'Hit@{k}': [] for k in k_list})

    test_users = test_df['user_idx'].unique()

    with torch.no_grad():
        # Compute all embeddings once
        user_emb, item_emb = model.forward(edge_index)
        all_scores = user_emb @ item_emb.T  # [num_users, num_items]

    for user_id in test_users:
        user_test = test_df[test_df['user_idx'] == user_id]['item_idx'].tolist()
        if not user_test:
            continue

        scores = all_scores[user_id].cpu().numpy()

        # Mask out training items
        seen_items = train_user_items.get(user_id, set())
        scores[list(seen_items)] = -np.inf

        # Get top-max(k_list) recommendations
        max_k = max(k_list)
        top_items = np.argpartition(scores, -max_k)[-max_k:]
        top_items = top_items[np.argsort(scores[top_items])[::-1]]

        for k in k_list:
            all_metrics[f'Recall@{k}'].append(recall_at_k(top_items, user_test, k))
            all_metrics[f'NDCG@{k}'].append(ndcg_at_k(top_items, user_test, k))
            all_metrics[f'Hit@{k}'].append(hit_rate_at_k(top_items, user_test, k))

    return {key: np.mean(vals) for key, vals in all_metrics.items()}
