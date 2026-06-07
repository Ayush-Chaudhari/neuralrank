# src/retrieval/dense.py
import os
import pickle
import numpy as np
import pandas as pd
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from src.utils import load_config, get_logger

log = get_logger("dense_retrieval")
config = load_config()

ROOT_DIR = Path(__file__).parent.parent.parent
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
INDEX_DIR = ROOT_DIR / "data" / "indexes"

os.makedirs(INDEX_DIR, exist_ok=True)

# Paths
FAISS_INDEX_PATH = INDEX_DIR / "faiss_index.bin"
PASSAGE_MAP_PATH = INDEX_DIR / "passage_map.pkl"

# Embedding model — small but powerful, runs on CPU fine
MODEL_NAME = config["model"]["embedding_model"]


def get_model():
    """Loads the sentence transformer model."""
    print(f"   Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    print(f"   Embedding dimension: {model.get_sentence_embedding_dimension()}")
    return model


def build_faiss_index(all_df: pd.DataFrame, model: SentenceTransformer,
                      batch_size: int = 256):
    """
    Builds a FAISS index from all passages.

    FAISS = Facebook AI Similarity Search.
    Stores all passage vectors and finds nearest neighbors
    in milliseconds even with millions of vectors.
    """
    if FAISS_INDEX_PATH.exists() and PASSAGE_MAP_PATH.exists():
        print("⚡ FAISS index already exists — loading from disk...")
        index = faiss.read_index(str(FAISS_INDEX_PATH))
        with open(PASSAGE_MAP_PATH, "rb") as f:
            passage_map = pickle.load(f)
        print(f"✅ Loaded FAISS index — {index.ntotal} vectors")
        return index, passage_map

    print("\n🔨 Building FAISS index...")
    log.info("Building FAISS index")

    # Get unique passages
    passages_df = all_df[["passage_id", "passage"]].drop_duplicates("passage_id")
    passages = passages_df["passage"].tolist()
    passage_ids = passages_df["passage_id"].tolist()

    print(f"   Encoding {len(passages)} passages in batches of {batch_size}...")
    print(f"   This takes ~5-10 minutes on CPU. Please wait...")

    # Encode in batches with progress bar
    all_embeddings = []
    for i in tqdm(range(0, len(passages), batch_size), desc="Encoding passages"):
        batch = passages[i: i + batch_size]
        embeddings = model.encode(
            batch,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        all_embeddings.append(embeddings)

    all_embeddings = np.vstack(all_embeddings).astype(np.float32)
    print(f"✅ Encoded {len(passages)} passages — shape: {all_embeddings.shape}")

    # Normalize for cosine similarity
    faiss.normalize_L2(all_embeddings)

    # Build FAISS index — IndexFlatIP = exact inner product search
    dim = all_embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(all_embeddings)

    print(f"✅ FAISS index built — {index.ntotal} vectors, dim={dim}")

    # Save index and passage map
    faiss.write_index(index, str(FAISS_INDEX_PATH))

    passage_map = {
        "passages": passages,
        "passage_ids": passage_ids
    }
    with open(PASSAGE_MAP_PATH, "wb") as f:
        pickle.dump(passage_map, f)

    print(f"✅ Saved FAISS index to {FAISS_INDEX_PATH}")
    log.info(f"FAISS index built — {index.ntotal} vectors, dim={dim}")

    return index, passage_map


def dense_retrieve(query: str, model: SentenceTransformer,
                   index, passage_map: dict, top_k: int = 100) -> list:
    """
    Retrieves top-k passages using dense vector similarity.
    Much better than BM25 for semantic/conceptual queries.
    """
    # Encode query
    query_vec = model.encode([query], convert_to_numpy=True).astype(np.float32)
    faiss.normalize_L2(query_vec)

    # Search FAISS index
    scores, indices = index.search(query_vec, top_k)

    results = []
    for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), 1):
        if idx == -1:
            continue
        results.append({
            "passage_id": passage_map["passage_ids"][idx],
            "passage": passage_map["passages"][idx],
            "dense_score": float(score),
            "rank": rank
        })

    return results


def evaluate_dense(eval_df: pd.DataFrame, model: SentenceTransformer,
                   index, passage_map: dict, top_k: int = 10) -> dict:
    """
    Evaluates dense retrieval using NDCG@10 and Recall@10.
    Compare these numbers against BM25 to see improvement.
    """
    print(f"\n📊 Evaluating Dense Retrieval @ top-{top_k}...")
    log.info(f"Evaluating Dense Retrieval @ top-{top_k}")

    ndcg_scores = []
    recall_scores = []
    unique_queries = eval_df["query_id"].unique()[:200]

    for qid in tqdm(unique_queries, desc="Evaluating queries"):
        query_rows = eval_df[eval_df["query_id"] == qid]
        query_text = query_rows["query"].iloc[0]

        relevant_texts = set(
            query_rows[query_rows["label"] == 1]["passage"].tolist()
        )

        if not relevant_texts:
            continue

        # Dense retrieve
        results = dense_retrieve(query_text, model, index, passage_map, top_k)
        retrieved_texts = [r["passage"] for r in results]

        # NDCG@k
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
        recall = len(relevant_texts & set(retrieved_texts)) / len(relevant_texts)
        recall_scores.append(recall)

    metrics = {
        "ndcg@10": float(np.mean(ndcg_scores)),
        "recall@10": float(np.mean(recall_scores)),
        "num_queries_evaluated": len(ndcg_scores)
    }

    print(f"\n{'='*40}")
    print(f"  Dense Retrieval Results")
    print(f"{'='*40}")
    print(f"  NDCG@10  : {metrics['ndcg@10']:.4f}  (BM25 was 0.3414)")
    print(f"  Recall@10: {metrics['recall@10']:.4f}  (BM25 was 0.5807)")
    print(f"  Queries  : {metrics['num_queries_evaluated']}")
    print(f"{'='*40}")

    if metrics["ndcg@10"] > 0.3414:
        print("🚀 Dense retrieval beats BM25!")
    else:
        print("💡 BM25 still ahead — hybrid will beat both!")

    log.info(f"Dense NDCG@10={metrics['ndcg@10']:.4f} Recall@10={metrics['recall@10']:.4f}")

    return metrics


if __name__ == "__main__":
    # Load data
    print("Loading processed data...")
    train_df = pd.read_csv(PROCESSED_DIR / "train.csv")
    eval_df = pd.read_csv(PROCESSED_DIR / "eval.csv")
    all_df = pd.concat([train_df, eval_df], ignore_index=True)
    print(f"✅ Total passages to index: {all_df['passage_id'].nunique()}")

    # Load model
    model = get_model()

    # Build FAISS index
    index, passage_map = build_faiss_index(all_df, model, batch_size=256)

    # Demo search
    print("\n🔍 Demo search:")
    query = "what was the impact of the manhattan project"
    results = dense_retrieve(query, model, index, passage_map, top_k=5)
    print(f"Query: '{query}'")
    for r in results:
        print(f"  Rank {r['rank']} (score={r['dense_score']:.3f}): {r['passage'][:100]}...")

    # Evaluate
    metrics = evaluate_dense(eval_df, model, index, passage_map, top_k=10)

    print("\n✅ Module 1.4 complete!")
    print("✅ Phase 1 DONE — Ready for Phase 2: Neural Reranker")