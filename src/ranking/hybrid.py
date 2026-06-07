# src/ranking/hybrid.py
"""
Hybrid Reranker — combines BM25 + Dense + LambdaRank

This is the final ranking layer that combines all signals:
  1. BM25 score        (keyword match)
  2. Dense score       (semantic similarity)
  3. LambdaRank score  (learned neural ranking)

Final score = weighted combination of all three.
This is exactly how production search engines work.
"""
import numpy as np
import pandas as pd
import torch
import pickle
import faiss
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from src.utils import load_config, get_logger
from src.models.lambdarank import RankerNet, load_model, ndcg_at_k
from src.features.extractor import compute_features, tokenize

log = get_logger("hybrid_reranker")
config = load_config()

ROOT_DIR = Path(__file__).parent.parent.parent
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
INDEX_DIR = ROOT_DIR / "data" / "indexes"


def load_all_components():
    """Loads BM25, FAISS, embedding model, and LambdaRank model."""
    print("🔧 Loading all components...")

    # BM25
    with open(INDEX_DIR / "bm25_index.pkl", "rb") as f:
        bm25_data = pickle.load(f)
    bm25 = bm25_data["index"]
    passages = bm25_data["passages"]
    passage_ids = bm25_data["passage_ids"]
    print(f"✅ BM25: {len(passages)} passages")

    # FAISS
    faiss_index = faiss.read_index(str(INDEX_DIR / "faiss_index.bin"))
    with open(INDEX_DIR / "passage_map.pkl", "rb") as f:
        passage_map = pickle.load(f)
    print(f"✅ FAISS: {faiss_index.ntotal} vectors")

    # Embedding model
    embed_model = SentenceTransformer(config["model"]["embedding_model"])
    print(f"✅ Embedding model loaded")

    # LambdaRank model
    ranker = load_model(input_dim=10)
    print(f"✅ LambdaRank model loaded")

    return bm25, passages, passage_ids, faiss_index, passage_map, embed_model, ranker


def hybrid_rerank(query: str,
                  bm25, passages: list, passage_ids: list,
                  faiss_index, passage_map: dict,
                  embed_model: SentenceTransformer,
                  ranker: RankerNet,
                  top_k: int = 10,
                  bm25_weight: float = 0.2,
                  dense_weight: float = 0.5,
                  neural_weight: float = 0.3) -> list:
    """
    Full hybrid reranking pipeline for one query.

    Steps:
      1. BM25 retrieves top-100 candidates
      2. Dense retrieves top-100 candidates
      3. Merge into candidate pool
      4. Extract 10 features per candidate
      5. LambdaRank scores each candidate
      6. Final score = weighted(BM25, dense, neural)
      7. Return top-k
    """
    # ── Step 1: BM25 retrieval ──
    tokens = tokenize(query)
    bm25_scores_arr = bm25.get_scores(tokens)
    top_bm25_idx = np.argsort(bm25_scores_arr)[::-1][:100]

    bm25_candidates = {}
    for idx in top_bm25_idx:
        bm25_candidates[passages[idx]] = float(bm25_scores_arr[idx])

    # ── Step 2: Dense retrieval ──
    query_vec = embed_model.encode(
        [query], convert_to_numpy=True
    ).astype(np.float32)
    faiss.normalize_L2(query_vec)
    dense_scores_raw, dense_idx = faiss_index.search(query_vec, 100)

    dense_candidates = {}
    for score, idx in zip(dense_scores_raw[0], dense_idx[0]):
        if idx >= 0:
            text = passage_map["passages"][idx]
            dense_candidates[text] = float(score)

    # ── Step 3: Merge candidates ──
    all_texts = set(bm25_candidates.keys()) | set(dense_candidates.keys())

    # ── Step 4 & 5: Feature extraction + LambdaRank scoring ──
    candidate_list = []
    feature_matrix = []

    for text in all_texts:
        b_score = bm25_candidates.get(text, 0.0)
        d_score = dense_candidates.get(text, 0.0)
        features = compute_features(query, text, b_score, d_score)
        feature_matrix.append(features)
        candidate_list.append({
            "passage": text,
            "bm25_score": b_score,
            "dense_score": d_score,
        })

    # Neural scoring
    ranker.eval()
    with torch.no_grad():
        X = torch.FloatTensor(np.vstack(feature_matrix))
        neural_scores = ranker(X).numpy()

    # ── Step 6: Normalize and combine scores ──
    def normalize(arr):
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-8:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)

    bm25_arr = normalize(np.array([c["bm25_score"] for c in candidate_list]))
    dense_arr = normalize(np.array([c["dense_score"] for c in candidate_list]))
    neural_arr = normalize(neural_scores)

    final_scores = (
        bm25_weight * bm25_arr +
        dense_weight * dense_arr +
        neural_weight * neural_arr
    )

    # ── Step 7: Sort and return top-k ──
    for i, cand in enumerate(candidate_list):
        cand["neural_score"] = float(neural_scores[i])
        cand["final_score"] = float(final_scores[i])

    candidate_list.sort(key=lambda x: x["final_score"], reverse=True)
    return candidate_list[:top_k]


def evaluate_hybrid(eval_df: pd.DataFrame,
                    bm25, passages, passage_ids,
                    faiss_index, passage_map,
                    embed_model, ranker,
                    top_k: int = 10,
                    max_queries: int = 200) -> dict:
    """
    Evaluates the full hybrid reranker on eval set.
    Compares against all previous baselines.
    """
    print(f"\n📊 Evaluating Hybrid Reranker @ top-{top_k}...")
    log.info("Evaluating hybrid reranker")

    ndcg_scores = []
    recall_scores = []
    unique_queries = eval_df["query_id"].unique()[:max_queries]

    for qid in tqdm(unique_queries, desc="Evaluating"):
        query_rows = eval_df[eval_df["query_id"] == qid]
        query_text = query_rows["query"].iloc[0]

        relevant_texts = set(
            query_rows[query_rows["label"] == 1]["passage"].tolist()
        )

        if not relevant_texts:
            continue

        # Hybrid rerank
        results = hybrid_rerank(
            query_text,
            bm25, passages, passage_ids,
            faiss_index, passage_map,
            embed_model, ranker,
            top_k=top_k
        )

        retrieved_texts = [r["passage"] for r in results]

        # NDCG@k
        labels = np.array([
            1.0 if t in relevant_texts else 0.0
            for t in retrieved_texts
        ])
        scores = np.array([r["final_score"] for r in results])
        ndcg = ndcg_at_k(scores, labels, k=top_k)
        ndcg_scores.append(ndcg)

        # Recall@k
        recall = len(relevant_texts & set(retrieved_texts)) / len(relevant_texts)
        recall_scores.append(recall)

    metrics = {
        "ndcg@10": float(np.mean(ndcg_scores)),
        "recall@10": float(np.mean(recall_scores)),
        "num_queries": len(ndcg_scores)
    }

    print(f"\n{'='*50}")
    print(f"  Final Results — All Models")
    print(f"{'='*50}")
    print(f"  BM25 baseline     NDCG@10 = 0.3414")
    print(f"  LambdaRank alone  NDCG@10 = 0.4126")
    print(f"  Dense retrieval   NDCG@10 = 0.6214")
    print(f"  Hybrid reranker   NDCG@10 = {metrics['ndcg@10']:.4f}  ← YOU ARE HERE")
    print(f"{'='*50}")
    print(f"  Recall@10  = {metrics['recall@10']:.4f}")
    print(f"  Queries    = {metrics['num_queries']}")

    improvement = ((metrics['ndcg@10'] - 0.3414) / 0.3414) * 100
    print(f"  Improvement vs BM25: +{improvement:.1f}%")
    print(f"{'='*50}")

    log.info(f"Hybrid NDCG@10={metrics['ndcg@10']:.4f} "
             f"Recall@10={metrics['recall@10']:.4f}")

    return metrics


if __name__ == "__main__":
    # Load data
    eval_df = pd.read_csv(PROCESSED_DIR / "eval.csv")

    # Load all components
    (bm25, passages, passage_ids,
     faiss_index, passage_map,
     embed_model, ranker) = load_all_components()

    # Demo search
    print("\n🔍 Demo hybrid search:")
    query = "what was the impact of the manhattan project"
    results = hybrid_rerank(
        query,
        bm25, passages, passage_ids,
        faiss_index, passage_map,
        embed_model, ranker,
        top_k=5
    )
    print(f"\nQuery: '{query}'")
    print(f"{'Rank':<5} {'Final':>7} {'BM25':>7} "
          f"{'Dense':>7} {'Neural':>8}  Passage")
    print("-" * 80)
    for i, r in enumerate(results, 1):
        print(f"{i:<5} {r['final_score']:>7.4f} "
              f"{r['bm25_score']:>7.3f} "
              f"{r['dense_score']:>7.4f} "
              f"{r['neural_score']:>8.4f}  "
              f"{r['passage'][:50]}...")

    # Full evaluation
    metrics = evaluate_hybrid(
        eval_df,
        bm25, passages, passage_ids,
        faiss_index, passage_map,
        embed_model, ranker,
        top_k=10,
        max_queries=200
    )

    print("\n✅ Module 2.4 complete!")
    print("✅ Phase 2 DONE — Ready for Phase 3: Real-time Pipeline")