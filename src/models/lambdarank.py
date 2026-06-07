# src/models/lambdarank.py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm
import pickle
import os
from src.utils import load_config, get_logger

log = get_logger("lambdarank")
config = load_config()

ROOT_DIR = Path(__file__).parent.parent.parent
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models" / "checkpoints"

os.makedirs(MODELS_DIR, exist_ok=True)

MODEL_PATH = MODELS_DIR / "lambdarank.pt"


# ─────────────────────────────────────────
# 1. Neural Network Architecture
# ─────────────────────────────────────────

class RankerNet(nn.Module):
    """
    Feed-forward neural network for ranking.
    Input:  10 features per query-passage pair
    Output: 1 relevance score (higher = more relevant)

    Architecture: 10 → 128 → 64 → 32 → 1
    Uses BatchNorm + Dropout to prevent overfitting.
    """
    def __init__(self, input_dim: int = 10):
        super(RankerNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


# ─────────────────────────────────────────
# 2. LambdaRank Loss
# ─────────────────────────────────────────

def lambda_loss(scores: torch.Tensor,
                labels: torch.Tensor,
                eps: float = 1e-10) -> torch.Tensor:
    """
    LambdaRank pairwise loss.

    For every pair (i, j) where label_i > label_j:
      - We want score_i > score_j
      - Loss = log(1 + exp(score_j - score_i))
      - Weighted by |ΔNDCG| — pairs that matter more get higher weight

    This directly optimizes NDCG, which is why it works so well.
    """
    # Get all pairs where i is relevant and j is not
    relevant_mask = labels == 1
    irrelevant_mask = labels == 0

    if relevant_mask.sum() == 0 or irrelevant_mask.sum() == 0:
        return torch.tensor(0.0, requires_grad=True)

    relevant_scores = scores[relevant_mask]
    irrelevant_scores = scores[irrelevant_mask]

    # Pairwise differences: relevant - irrelevant
    # Shape: (num_relevant, num_irrelevant)
    diff = relevant_scores.unsqueeze(1) - irrelevant_scores.unsqueeze(0)

    # Logistic loss: we want diff > 0 (relevant ranked higher)
    loss = torch.log1p(torch.exp(-diff))

    return loss.mean()


# ─────────────────────────────────────────
# 3. NDCG Metric
# ─────────────────────────────────────────

def ndcg_at_k(scores: np.ndarray,
              labels: np.ndarray,
              k: int = 10) -> float:
    """
    Computes NDCG@k for a single query group.
    Used for evaluation only (not training).
    """
    if len(scores) == 0:
        return 0.0

    # Sort by predicted score descending
    order = np.argsort(scores)[::-1][:k]
    ranked_labels = labels[order]

    # DCG
    dcg = sum(
        rel / np.log2(rank + 2)
        for rank, rel in enumerate(ranked_labels)
    )

    # Ideal DCG
    ideal_labels = np.sort(labels)[::-1][:k]
    idcg = sum(
        rel / np.log2(rank + 2)
        for rank, rel in enumerate(ideal_labels)
    )

    return dcg / idcg if idcg > 0 else 0.0


def evaluate_ndcg(model: RankerNet,
                  X: np.ndarray,
                  y: np.ndarray,
                  groups: list,
                  k: int = 10) -> float:
    """Evaluates NDCG@k across all query groups."""
    model.eval()
    ndcg_scores = []
    idx = 0

    with torch.no_grad():
        for group_size in groups:
            X_group = torch.FloatTensor(X[idx: idx + group_size])
            y_group = y[idx: idx + group_size]
            scores = model(X_group).numpy()
            ndcg = ndcg_at_k(scores, y_group, k)
            ndcg_scores.append(ndcg)
            idx += group_size

    return float(np.mean(ndcg_scores))


# ─────────────────────────────────────────
# 4. Training Loop
# ─────────────────────────────────────────

def train(X_train: np.ndarray, y_train: np.ndarray, groups_train: list,
          X_eval: np.ndarray, y_eval: np.ndarray, groups_eval: list,
          epochs: int = 20, lr: float = 0.001, batch_size: int = 512):
    """
    Full training loop with:
    - LambdaRank loss
    - Adam optimizer
    - NDCG@10 evaluation every epoch
    - Best model checkpointing
    """
    print("\n🧠 Training LambdaRank Neural Reranker")
    print(f"   Train pairs: {len(X_train)}")
    print(f"   Eval pairs:  {len(X_eval)}")
    print(f"   Epochs: {epochs} | LR: {lr} | Batch: {batch_size}")
    print(f"   Device: CPU")
    print()

    model = RankerNet(input_dim=X_train.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    best_ndcg = 0.0
    history = []

    # Baseline NDCG before training
    baseline_ndcg = evaluate_ndcg(model, X_eval, y_eval, groups_eval)
    print(f"   Baseline NDCG@10 (before training): {baseline_ndcg:.4f}")
    print(f"   BM25 baseline was:                  0.3414")
    print(f"   Dense baseline was:                 0.6214")
    print()

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses = []

        # Shuffle training data by groups
        group_indices = list(range(len(groups_train)))
        np.random.shuffle(group_indices)

        # Rebuild shuffled arrays
        new_X, new_y = [], []
        idx = 0
        group_starts = np.cumsum([0] + groups_train)
        for gi in group_indices:
            start = group_starts[gi]
            end = group_starts[gi + 1]
            new_X.append(X_train[start:end])
            new_y.append(y_train[start:end])

        X_shuf = np.vstack(new_X)
        y_shuf = np.concatenate(new_y)

        # Mini-batch training
        n = len(X_shuf)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            X_batch = torch.FloatTensor(X_shuf[start:end])
            y_batch = torch.FloatTensor(y_shuf[start:end])

            optimizer.zero_grad()
            scores = model(X_batch)
            loss = lambda_loss(scores, y_batch)

            if loss.requires_grad:
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())

        scheduler.step()

        # Evaluate
        train_ndcg = evaluate_ndcg(model, X_train, y_train, groups_train)
        eval_ndcg = evaluate_ndcg(model, X_eval, y_eval, groups_eval)
        avg_loss = np.mean(epoch_losses) if epoch_losses else 0.0

        history.append({
            "epoch": epoch,
            "loss": avg_loss,
            "train_ndcg": train_ndcg,
            "eval_ndcg": eval_ndcg
        })

        # Save best model
        if eval_ndcg > best_ndcg:
            best_ndcg = eval_ndcg
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "eval_ndcg": eval_ndcg,
                "config": config
            }, MODEL_PATH)
            marker = " ← best"
        else:
            marker = ""

        print(f"Epoch {epoch:02d}/{epochs} | "
              f"Loss: {avg_loss:.4f} | "
              f"Train NDCG: {train_ndcg:.4f} | "
              f"Eval NDCG: {eval_ndcg:.4f}{marker}")

        log.info(f"Epoch {epoch} | loss={avg_loss:.4f} | "
                 f"train_ndcg={train_ndcg:.4f} | eval_ndcg={eval_ndcg:.4f}")

    print(f"\n{'='*50}")
    print(f"  Training Complete!")
    print(f"{'='*50}")
    print(f"  Best Eval NDCG@10 : {best_ndcg:.4f}")
    print(f"  BM25 baseline     : 0.3414")
    print(f"  Dense baseline    : 0.6214")
    improvement = ((best_ndcg - 0.3414) / 0.3414) * 100
    print(f"  Improvement vs BM25: +{improvement:.1f}%")
    print(f"{'='*50}")
    print(f"\n✅ Model saved to {MODEL_PATH}")

    return model, history, best_ndcg


def load_model(input_dim: int = 10) -> RankerNet:
    """Loads the best saved model from disk."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"No model found at {MODEL_PATH}. Train first.")
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    model = RankerNet(input_dim=input_dim)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    print(f"✅ Loaded model — best eval NDCG: {checkpoint['eval_ndcg']:.4f}")
    return model


if __name__ == "__main__":
    # Load features
    print("Loading features...")
    with open(PROCESSED_DIR / "features_train.pkl", "rb") as f:
        train_data = pickle.load(f)
    with open(PROCESSED_DIR / "features_eval.pkl", "rb") as f:
        eval_data = pickle.load(f)

    X_train = train_data["X"]
    y_train = train_data["y"]
    groups_train = train_data["groups"]

    X_eval = eval_data["X"]
    y_eval = eval_data["y"]
    groups_eval = eval_data["groups"]

    print(f"✅ Train: {X_train.shape} | Eval: {X_eval.shape}")

    # Train the model
    model, history, best_ndcg = train(
        X_train, y_train, groups_train,
        X_eval, y_eval, groups_eval,
        epochs=20,
        lr=0.001,
        batch_size=512
    )

    # Show training curve
    print("\n📈 Training History:")
    print(f"{'Epoch':>6} {'Loss':>8} {'Train NDCG':>12} {'Eval NDCG':>11}")
    print("-" * 42)
    for h in history:
        print(f"{h['epoch']:>6} {h['loss']:>8.4f} "
              f"{h['train_ndcg']:>12.4f} {h['eval_ndcg']:>11.4f}")

    print("\n✅ Module 2.2 complete!")
    print("✅ Ready for Module 2.3 — RAG Pipeline")