"""
Training Script for LightGCN Amazon Product Recommender

Usage:
    python train.py
    python train.py --epochs 50 --dim 128 --layers 4 --lr 0.001
    python train.py --device cuda  # for GPU training
"""

import argparse
import torch
import numpy as np
import json
import os
from torch.utils.data import DataLoader, TensorDataset

from models.lightgcn import LightGCN
from models.loss import bpr_loss, sample_negative_items
from utils.data_loader import (generate_synthetic_amazon_data,
                                train_val_test_split, build_graph,
                                get_user_positive_items)
from utils.metrics import evaluate_model


def parse_args():
    parser = argparse.ArgumentParser(description='Train LightGCN Recommender')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--dim', type=int, default=64, help='Embedding dimension')
    parser.add_argument('--layers', type=int, default=3, help='Number of GCN layers')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--reg', type=float, default=1e-4, help='L2 regularization')
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--save_path', type=str, default='checkpoints/best_model.pt')
    return parser.parse_args()


def train_epoch(model, edge_index, train_df, user_pos_items,
                optimizer, args, num_items):
    model.train()
    total_loss, total_bpr, total_reg = 0, 0, 0
    n_batches = 0

    users = torch.tensor(train_df['user_idx'].values, dtype=torch.long)
    pos_items = torch.tensor(train_df['item_idx'].values, dtype=torch.long)

    # Shuffle
    perm = torch.randperm(len(users))
    users, pos_items = users[perm], pos_items[perm]

    for start in range(0, len(users), args.batch_size):
        end = min(start + args.batch_size, len(users))
        batch_users = users[start:end].to(args.device)
        batch_pos = pos_items[start:end].to(args.device)
        batch_neg = sample_negative_items(batch_pos, num_items).to(args.device)

        # Forward pass
        user_emb, item_emb = model(edge_index)

        # Get final embeddings for this batch
        u_emb = user_emb[batch_users]
        pi_emb = item_emb[batch_pos]
        ni_emb = item_emb[batch_neg]

        # Get initial (layer-0) embeddings for regularization
        u_emb_0 = model.user_embedding(batch_users)
        pi_emb_0 = model.item_embedding(batch_pos)
        ni_emb_0 = model.item_embedding(batch_neg)

        loss, bpr, reg = bpr_loss(u_emb, pi_emb, ni_emb,
                                   u_emb_0, pi_emb_0, ni_emb_0,
                                   args.reg)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_bpr += bpr
        total_reg += reg
        n_batches += 1

    return total_loss / n_batches, total_bpr / n_batches, total_reg / n_batches


def main():
    args = parse_args()
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    print("=" * 60)
    print("LightGCN - Amazon Product Recommendation")
    print("=" * 60)
    print(f"Config: dim={args.dim}, layers={args.layers}, lr={args.lr}")

    # Data
    print("\nLoading data...")
    df, user_map, item_map = generate_synthetic_amazon_data()
    train_df, val_df, test_df = train_val_test_split(df)

    num_users = df['user_idx'].nunique()
    num_items = df['item_idx'].nunique()
    print(f"Train: {len(train_df):,}  Val: {len(val_df):,}  Test: {len(test_df):,}")

    # Build graph
    edge_index = build_graph(train_df, num_users, num_items).to(args.device)
    user_pos_items = get_user_positive_items(train_df)

    # Model
    model = LightGCN(
        num_users=num_users,
        num_items=num_items,
        embedding_dim=args.dim,
        num_layers=args.layers,
        dropout=args.dropout
    ).to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    print(f"\nModel params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"\n{'Epoch':>6} {'Loss':>10} {'BPR':>10} {'Reg':>10} {'R@10':>10} {'NDCG@10':>10}")
    print("-" * 60)

    best_ndcg = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        loss, bpr, reg = train_epoch(model, edge_index, train_df,
                                      user_pos_items, optimizer, args, num_items)

        # Evaluate every 5 epochs
        if epoch % 5 == 0 or epoch == 1:
            metrics = evaluate_model(model, edge_index, val_df,
                                      user_pos_items, num_users, num_items)
            r10 = metrics['Recall@10']
            ndcg10 = metrics['NDCG@10']
            print(f"{epoch:>6} {loss:>10.4f} {bpr:>10.4f} {reg:>10.4f} {r10:>10.4f} {ndcg10:>10.4f}")

            history.append({'epoch': epoch, 'loss': loss, **metrics})

            # Save best model
            if ndcg10 > best_ndcg:
                best_ndcg = ndcg10
                torch.save({
                    'epoch': epoch,
                    'model_state': model.state_dict(),
                    'num_users': num_users,
                    'num_items': num_items,
                    'config': vars(args),
                    'metrics': metrics
                }, args.save_path)
        else:
            print(f"{epoch:>6} {loss:>10.4f} {bpr:>10.4f} {reg:>10.4f}")

        scheduler.step()

    # Final test evaluation
    print("\n" + "=" * 60)
    print("FINAL TEST EVALUATION")
    checkpoint = torch.load(args.save_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state'])
    test_metrics = evaluate_model(model, edge_index, test_df,
                                   user_pos_items, num_users, num_items)
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")

    with open('results/test_metrics.json', 'w') as f:
        json.dump(test_metrics, f, indent=2)
    with open('results/training_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\nBest model saved to: {args.save_path}")
    print("Results saved to: results/")


if __name__ == '__main__':
    main()
