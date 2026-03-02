"""
Amazon Product Reviews Dataset Loader

Graph Construction:
- Nodes: Users + Items (bipartite graph)
- Edges: user->item if user rated item >= 4 stars (positive implicit feedback)
- Bidirectional edges for LightGCN message passing
"""

import numpy as np
import pandas as pd
import torch
from collections import defaultdict


def generate_synthetic_amazon_data(num_users=2000, num_items=5000,
                                    num_interactions=50000, seed=42):
    """
    Generate synthetic Amazon Electronics-style interaction data.
    Power-law distributions mimic real Amazon data.
    """
    rng = np.random.default_rng(seed)

    item_popularity = rng.zipf(1.5, num_items)
    item_probs = item_popularity / item_popularity.sum()

    interactions = []
    for user_id in range(num_users):
        n = int(rng.zipf(2.0)) 
        n = max(5, min(n, 80))
        items = rng.choice(num_items, size=n, replace=False, p=item_probs)
        for item_id in items:
            rating = rng.choice([3, 4, 4, 5, 5, 5])
            interactions.append({'user_id': user_id, 'item_id': int(item_id), 'rating': rating,
                                   'category': rng.choice(['Electronics','Computers','Cameras','Phones','Audio'])})

    df = pd.DataFrame(interactions).drop_duplicates(['user_id', 'item_id'])
    df = df[df['rating'] >= 4].reset_index(drop=True)

    user_counts = df['user_id'].value_counts()
    active_users = user_counts[user_counts >= 5].index
    df = df[df['user_id'].isin(active_users)].reset_index(drop=True)

    user_map = {u: i for i, u in enumerate(df['user_id'].unique())}
    item_map = {it: i for i, it in enumerate(df['item_id'].unique())}
    df['user_idx'] = df['user_id'].map(user_map)
    df['item_idx'] = df['item_id'].map(item_map)

    print(f"Users: {df['user_idx'].nunique():,}  Items: {df['item_idx'].nunique():,}  Interactions: {len(df):,}")
    sparsity = 1 - len(df) / (df['user_idx'].nunique() * df['item_idx'].nunique())
    print(f"Sparsity: {sparsity:.4f}")

    return df, user_map, item_map


def train_val_test_split(df):
    """Leave-last-out split per user (standard recommendation protocol)."""
    train_data, val_data, test_data = [], [], []

    for user_id, group in df.groupby('user_idx'):
        items = group['item_idx'].tolist()
        if len(items) < 3:
            train_data.extend([(user_id, i) for i in items])
            continue
        test_data.append((user_id, items[-1]))
        val_data.append((user_id, items[-2]))
        train_data.extend([(user_id, i) for i in items[:-2]])

    return (pd.DataFrame(train_data, columns=['user_idx', 'item_idx']),
            pd.DataFrame(val_data, columns=['user_idx', 'item_idx']),
            pd.DataFrame(test_data, columns=['user_idx', 'item_idx']))


def build_graph(train_df, num_users, num_items):
    """Build bidirectional bipartite user-item graph."""
    users = torch.tensor(train_df['user_idx'].values, dtype=torch.long)
    items = torch.tensor(train_df['item_idx'].values + num_users, dtype=torch.long)

    edge_index = torch.stack([
        torch.cat([users, items]),
        torch.cat([items, users])
    ], dim=0)
    return edge_index


def get_user_positive_items(train_df):
    """Returns dict: user_id -> set of item_ids they interacted with."""
    user_pos = defaultdict(set)
    for _, row in train_df.iterrows():
        user_pos[int(row['user_idx'])].add(int(row['item_idx']))
    return user_pos
