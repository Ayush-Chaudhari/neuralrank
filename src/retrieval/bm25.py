# src/retrieval/bm25.py
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from rank_bm25 import BM25Okapi
from tqdm import tqdm
from src.utils import load_config, get_logger

log = get_logger("bm25")
config = load_config()

ROOT_DIR = Path(__file__).parent.parent.parent
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
INDEX_DIR = ROOT_DIR / "data" / "indexes"

import os
os.makedirs(INDEX_DIR, exist_ok=True)


def tokenize(text: str) -> list:
    """Simple whitespace + lowercase tokenizer."""
    return text.lower().split()


def build_bm25_index(train_df: pd.DataFrame):
    """
    Builds a BM25 index from all passages in the training set.
    Saves it to disk so we don't rebuild every time.
    """
    index_path = INDEX_DIR / "bm25_index.pkl"

    if index_path.exists():
        print("⚡ BM25 index already exists — loading from disk...")
        with open(index_path, "rb") as f:
            data = pickle.load(f)
        print(f"✅ Loaded BM25 index with {len(data['passages'])} passages")
        return data["index"], data["passages"], data["passage_ids"]

    print("\n🔨 Building BM25 index...")
    log.info("Building BM25 index")

    # Get unique passages
    passages_df = train_df[["passage_id", "passage"]].drop_duplicates("passage_id")
    passages = passages_df["passage"].tolist()
    passage_ids = passages_df["passage_id"].tolist()

    print(f"   Indexing {len(passages)} unique passages...")

    # Tokenize all passages
    tokenized = [tokenize(p) for p in tqdm(passages, desc="Tokenizing")]

    # Build BM25 index
    bm25 = BM25Okapi(tokenized)

    # Save to disk
    with open(index_path, "wb") as f:
        pickle.dump({
            "index": bm25,
            "passages": passages,
            "passage_ids": passage_ids
        }, f)

    print(f"✅ BM25 index built and saved — {len(passages)} passages")
    log.info(f"BM25 index built with {len(passages)} passages")

    return bm25, passages, passage_ids


def bm25_retrieve(query: str, bm25, passages: list,
                  passage_ids: list, top_k: int = 100) -> list:
    """
    Retrieves top-k passages for a query using BM25.
    Returns list of (passage_id, passage, score) sorted by score descending.
    """
    tokenized_query = tokenize(query)
    scores = bm25.get_scores(tokenized_query)

    # Get top-k indices
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        results.append({
            "passage_id": passage_ids[idx],
            "passage": passages[idx],
            "bm25_score": float(scores[idx]),
            "rank": len(results) + 1
        })

    return results


def evaluate_bm25(eval_df: pd.DataFrame, bm25,
                  passages: list, passage_ids: list,
                  top_k: int = 10) -> dict:
    """
    Evaluates BM25 using NDCG@10 and Recall@10.
    Fixed version — matches by passage text instead of ID.
    """
    print(f"\n📊 Evaluating BM25 @ top-{top_k}...")
    log.info(f"Evaluating BM25 @ top-{top_k}")

    # Build passage text -> index mapping for fast lookup
    passage_text_to_idx = {p: i for i, p in enumerate(passages)}

    ndcg_scores = []
    recall_scores = []
    unique_queries = eval_df["query_id"].unique()[:200]

    for qid in tqdm(unique_queries, desc="Evaluating queries"):
        query_rows = eval_df[eval_df["query_id"] == qid]
        query_text = query_rows["query"].iloc[0]

        # Get relevant passage texts directly
        relevant_texts = set(
            query_rows[query_rows["label"] == 1]["passage"].tolist()
        )

        if not relevant_texts:
            continue

        # Retrieve top-k by BM25
        tokenized_query = tokenize(query_text)
        scores = bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]
        retrieved_texts = [passages[i] for i in top_indices]

        # NDCG@k — compare by passage text
        dcg = 0.0
        for rank, text in enumerate(retrieved_texts, 1):
            if text in relevant_texts:
                dcg += 1.0 / np.log2(rank + 1)

        ideal_dcg = sum(
            1.0 / np.log2(i + 2)
            for i in range(min(len(relevant_texts), top_k))
        )

        ndcg = dcg / ideal_dcg if ideal_dcg > 0 else 0.0
        ndcg_scores.append(ndcg)

        # Recall@k
        retrieved_set = set(retrieved_texts)
        recall = len(relevant_texts & retrieved_set) / len(relevant_texts)
        recall_scores.append(recall)

    metrics = {
        "ndcg@10": float(np.mean(ndcg_scores)),
        "recall@10": float(np.mean(recall_scores)),
        "num_queries_evaluated": len(ndcg_scores)
    }

    print(f"\n{'='*40}")
    print(f"  BM25 Baseline Results")
    print(f"{'='*40}")
    print(f"  NDCG@10  : {metrics['ndcg@10']:.4f}")
    print(f"  Recall@10: {metrics['recall@10']:.4f}")
    print(f"  Queries  : {metrics['num_queries_evaluated']}")
    print(f"{'='*40}")
    print(f"\n💡 This is our baseline to beat with neural ranking!")

    log.info(f"BM25 NDCG@10={metrics['ndcg@10']:.4f} Recall@10={metrics['recall@10']:.4f}")

    return metrics


if __name__ == "__main__":
    # Load processed data
    print("Loading processed data...")
    train_df = pd.read_csv(PROCESSED_DIR / "train.csv")
    eval_df = pd.read_csv(PROCESSED_DIR / "eval.csv")
    print(f"✅ Train: {len(train_df)} pairs | Eval: {len(eval_df)} pairs")

    # Combine train + eval passages into one index
    # This is correct — in real search, the index has ALL documents
    all_df = pd.concat([train_df, eval_df], ignore_index=True)

    # Delete old index so it rebuilds with all passages
    import os
    old_index = INDEX_DIR / "bm25_index.pkl"
    if old_index.exists():
        os.remove(old_index)
        print("🗑️  Removed old index — rebuilding with all passages...")

    # Build index on all passages
    bm25, passages, passage_ids = build_bm25_index(all_df)

    # Quick search demo
    print("\n🔍 Demo search:")
    query = "what was the impact of the manhattan project"
    results = bm25_retrieve(query, bm25, passages, passage_ids, top_k=5)
    print(f"Query: '{query}'")
    for r in results:
        print(f"  Rank {r['rank']} (score={r['bm25_score']:.3f}): {r['passage'][:100]}...")

    # Evaluate on eval set
    metrics = evaluate_bm25(eval_df, bm25, passages, passage_ids, top_k=10)

    print("\n✅ Module 1.3 complete!")
    print("✅ Ready for Module 1.4 — FAISS Dense Retrieval")