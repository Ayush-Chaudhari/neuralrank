# src/mlops/tracking.py
"""
MLflow Experiment Tracking

Tracks every training run with:
  - Parameters (learning rate, epochs, batch size)
  - Metrics (NDCG@10, Recall@10, loss per epoch)
  - Artifacts (model file, feature importance plot)
  - Tags (dataset, model type, developer name)

This is how real ML teams track experiments.
"""
import os
import pickle
import numpy as np
import pandas as pd
import torch
import mlflow
import mlflow.pytorch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from src.utils import load_config, get_logger
from src.models.lambdarank import (
    RankerNet, train, evaluate_ndcg, ndcg_at_k
)

log = get_logger("mlflow_tracking")
config = load_config()

ROOT_DIR = Path(__file__).parent.parent.parent
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models" / "checkpoints"
MLFLOW_DIR = ROOT_DIR / "mlruns"

os.makedirs(MLFLOW_DIR, exist_ok=True)


def plot_training_curve(history: list, run_id: str) -> str:
    """Plots and saves training curve as artifact."""
    epochs = [h["epoch"] for h in history]
    train_ndcg = [h["train_ndcg"] for h in history]
    eval_ndcg = [h["eval_ndcg"] for h in history]
    losses = [h["loss"] for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # NDCG curve
    ax1.plot(epochs, train_ndcg, label="Train NDCG@10", color="#4C72B0")
    ax1.plot(epochs, eval_ndcg, label="Eval NDCG@10",
             color="#DD8452", linewidth=2)
    ax1.axhline(y=0.3414, color="gray", linestyle="--",
                alpha=0.7, label="BM25 baseline")
    ax1.axhline(y=0.6214, color="green", linestyle="--",
                alpha=0.7, label="Dense baseline")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("NDCG@10")
    ax1.set_title("Training Progress — NDCG@10")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Loss curve
    ax2.plot(epochs, losses, color="#C44E52", linewidth=2)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("LambdaRank Loss")
    ax2.set_title("Training Loss")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    plot_path = MODELS_DIR / f"training_curve_{run_id[:8]}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    return str(plot_path)


def plot_feature_importance(model: RankerNet) -> str:
    """Plots feature importance from first layer weights."""
    feature_names = [
        "BM25", "Dense", "PassLen", "QueryLen",
        "Overlap", "OverlapRatio", "LogPassLen",
        "LogBM25", "DenseSquared", "Combined"
    ]

    # Use absolute weight sum from first layer as importance
    first_layer = model.network[0]
    importance = first_layer.weight.data.abs().mean(dim=0).numpy()
    importance = importance / importance.sum()

    # Sort
    order = np.argsort(importance)[::-1]
    sorted_names = [feature_names[i] for i in order]
    sorted_imp = importance[order]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(sorted_names, sorted_imp, color="#4C72B0", alpha=0.8)
    ax.set_xlabel("Feature")
    ax.set_ylabel("Relative Importance")
    ax.set_title("Feature Importance — LambdaRank Model")
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels
    for bar, val in zip(bars, sorted_imp):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.001,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()

    plot_path = MODELS_DIR / "feature_importance.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    return str(plot_path)


def run_tracked_experiment(
        experiment_name: str = "neuralrank-ltr",
        run_name: str = "lambdarank-v1",
        epochs: int = 15,
        lr: float = 0.001,
        batch_size: int = 512,
        bm25_weight: float = 0.2,
        dense_weight: float = 0.5,
        neural_weight: float = 0.3):
    """
    Runs a full training experiment with MLflow tracking.
    Every parameter, metric, and artifact is logged.
    """

    # Set MLflow tracking URI
    mlflow.set_tracking_uri(f"sqlite:///{ROOT_DIR}/mlflow.db")
    mlflow.set_experiment(experiment_name)

    print(f"\n{'='*55}")
    print(f"  MLflow Experiment: {experiment_name}")
    print(f"  Run: {run_name}")
    print(f"{'='*55}\n")

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

    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        print(f"MLflow Run ID: {run_id}\n")

        # ── Log parameters ──
        mlflow.log_params({
            "epochs": epochs,
            "learning_rate": lr,
            "batch_size": batch_size,
            "bm25_weight": bm25_weight,
            "dense_weight": dense_weight,
            "neural_weight": neural_weight,
            "embedding_model": config["model"]["embedding_model"],
            "train_pairs": len(X_train),
            "eval_pairs": len(X_eval),
            "num_features": X_train.shape[1],
            "dataset": "msmarco-v2.1",
        })

        # ── Log tags ──
        mlflow.set_tags({
            "model_type": "LambdaRank",
            "retrieval": "BM25+FAISS",
            "developer": "neuralrank",
            "framework": "pytorch",
        })

        # ── Log baselines ──
        mlflow.log_metrics({
            "baseline_bm25_ndcg": 0.3414,
            "baseline_dense_ndcg": 0.6214,
            "baseline_hybrid_ndcg": 0.6048,
        })

        # ── Train model ──
        print("Training model...")
        model, history, best_ndcg = train(
            X_train, y_train, groups_train,
            X_eval, y_eval, groups_eval,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size
        )

        # ── Log metrics per epoch ──
        print("\nLogging epoch metrics to MLflow...")
        for h in history:
            mlflow.log_metrics({
                "train_ndcg": h["train_ndcg"],
                "eval_ndcg": h["eval_ndcg"],
                "loss": h["loss"],
            }, step=h["epoch"])

        # ── Log final metrics ──
        mlflow.log_metrics({
            "best_eval_ndcg": best_ndcg,
            "final_train_ndcg": history[-1]["train_ndcg"],
            "improvement_vs_bm25": (best_ndcg - 0.3414) / 0.3414,
        })

        # ── Save plots as artifacts ──
        print("Saving artifacts...")
        curve_path = plot_training_curve(history, run_id)
        mlflow.log_artifact(curve_path, "plots")
        print(f"✅ Training curve saved")

        importance_path = plot_feature_importance(model)
        mlflow.log_artifact(importance_path, "plots")
        print(f"✅ Feature importance saved")

        # ── Save model ──
        mlflow.pytorch.log_model(model, "model")
        print(f"✅ Model logged to MLflow")

        print(f"\n{'='*55}")
        print(f"  Experiment Complete!")
        print(f"{'='*55}")
        print(f"  Run ID      : {run_id[:16]}...")
        print(f"  Best NDCG   : {best_ndcg:.4f}")
        print(f"  vs BM25     : +{((best_ndcg-0.3414)/0.3414)*100:.1f}%")
        print(f"{'='*55}")

        return run_id, best_ndcg


if __name__ == "__main__":
    # Run experiment
    run_id, best_ndcg = run_tracked_experiment(
        experiment_name="neuralrank-ltr",
        run_name="lambdarank-v1",
        epochs=15,
        lr=0.001,
        batch_size=512,
    )

    print("\n📊 Starting MLflow UI...")
    print("   Run this command in a NEW terminal:")
    print(f"   mlflow ui --backend-store-uri file:///{MLFLOW_DIR}")
    print("   Then open: http://localhost:5000")
    print("\n✅ Module 4.1 complete!")
    print("✅ Ready for Module 4.2 — Evidently Monitoring")