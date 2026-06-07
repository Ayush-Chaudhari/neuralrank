# src/features/extractor.py
import numpy as np
import pandas as pd
from pathlib import Path
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import pickle
import os
from src.utils import load_config, get_logger

log = get_logger("feature_extractor")
config = load_config()

ROOT_DIR = Path(__file__).parent.parent.parent
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
INDEX_DIR = ROOT_DIR / "data" / "indexes"
FEATURES_DIR = ROOT_DIR / "data" / "processed"

os.makedirs(FEATURES_DIR, exist_ok=True)


def tokenize(text: str) -> list:
    return text.lower().split()


def compute_features(query: str, passage: str,
                     bm25_score: float, dense_score: float) -> np.ndarray:
    """
    Extracts 10 features for a query-passage pair.
    These are the inputs to our LambdaRank model.

    Features:
        0  - BM25 score (keyword match strength)
        1  - Dense cosine similarity (semantic match)
        2  - Passage length (number of words)
        3  - Query length (number of words)
        4  - Exact query term overlap (how many query words in passage)
        5  - Term overlap ratio (overlap / query length)
        6  - Passage length normalized (log scale)
        7  - BM25 score normalized (log scale)
        8  - Dense score squared (emphasize high similarity)
        9  - Combined score (BM25 + dense weighted sum)
    """
    query_tokens = tokenize(query)
    passage_tokens = tokenize(passage)
    passage_token_set = set(passage_tokens)

    # Feature 0: BM25 score
    f0 = bm25_score

    # Feature 1: Dense cosine similarity
    f1 = dense_score

    # Feature 2: Passage length
    f2 = len(passage_tokens)

    # Feature 3: Query length
    f3 = len(query_tokens)

    # Feature 4: Exact term overlap count
    overlap = sum(1 for t in query_tokens if t in passage_token_set)
    f4 = float(overlap)

    # Feature 5: Term overlap ratio
    f5 = overlap / len(query_tokens) if query_tokens else 0.0

    # Feature 6: Log passage length
    f6 = np.log1p(f2)

    # Feature 7: Log BM25 score
    f7 = np.log1p(max(f0, 0))

    # Feature 8: Dense score squared
    f8 = f1 ** 2

    # Feature 9: Weighted combination
    f9 = 0.4 * f0 + 0.6 * f1

    return np.array([f0, f1, f2, f3, f4, f5, f6, f7, f8, f9],
                    dtype=np.float32)


def build_feature_dataset(df: pd.DataFrame,
                          bm25, passages: list,
                          passage_ids: list,
                          model: SentenceTransformer,
                          split: str = "train",
                          max_queries: int = 2000) -> tuple:
    """
    Builds feature matrix X and label vector y for LTR training.

    For each query:
      - Retrieve top-100 passages with BM25
      - Retrieve top-100 passages with dense
      - Merge candidates
      - Compute 10 features per pair
      - Label = 1 if relevant, 0 if not

    Returns:
      X: (N, 10) feature matrix
      y: (N,) labels
      groups: list of group sizes (for LTR — how many passages per query)
    """
    save_path = FEATURES_DIR / f"features_{split}.pkl"

    if save_path.exists():
        print(f"⚡ Features already exist for {split} — loading from disk...")
        with open(save_path, "rb") as f:
            data = pickle.load(f)
        print(f"✅ Loaded {len(data['X'])} feature vectors")
        return data["X"], data["y"], data["groups"]

    print(f"\n🔨 Building feature dataset for {split}...")
    log.info(f"Building features for {split}")

    # Build passage text -> BM25 score lookup
    passage_text_to_idx = {p: i for i, p in enumerate(passages)}

    unique_queries = df["query_id"].unique()[:max_queries]
    print(f"   Processing {len(unique_queries)} queries...")

    all_X = []
    all_y = []
    groups = []

    for qid in tqdm(unique_queries, desc=f"Building {split} features"):
        query_rows = df[df["query_id"] == qid]
        query_text = query_rows["query"].iloc[0]

        # Ground truth relevant passages
        relevant_texts = set(
            query_rows[query_rows["label"] == 1]["passage"].tolist()
        )

        # BM25 candidates
        query_tokens = tokenize(query_text)
        bm25_scores = bm25.get_scores(query_tokens)
        top_bm25_idx = np.argsort(bm25_scores)[::-1][:50]
        bm25_candidates = {
            passages[i]: float(bm25_scores[i])
            for i in top_bm25_idx
        }

        # Dense candidates
        query_vec = model.encode([query_text], convert_to_numpy=True).astype(np.float32)
        import faiss as faiss_module
        faiss_module.normalize_L2(query_vec)

        # Load FAISS index
        faiss_index_path = INDEX_DIR / "faiss_index.bin"
        faiss_map_path = INDEX_DIR / "passage_map.pkl"
        faiss_index = faiss_module.read_index(str(faiss_index_path))
        with open(faiss_map_path, "rb") as f:
            passage_map = pickle.load(f)

        dense_scores_raw, dense_indices = faiss_index.search(query_vec, 50)
        dense_candidates = {}
        for score, idx in zip(dense_scores_raw[0], dense_indices[0]):
            if idx >= 0:
                dense_candidates[passage_map["passages"][idx]] = float(score)

        # Merge all candidates
        all_candidates = set(bm25_candidates.keys()) | set(dense_candidates.keys())

        if not all_candidates:
            continue

        # Build features for each candidate
        group_size = 0
        for passage_text in all_candidates:
            bm25_score = bm25_candidates.get(passage_text, 0.0)
            dense_score = dense_candidates.get(passage_text, 0.0)
            features = compute_features(
                query_text, passage_text, bm25_score, dense_score
            )
            label = 1.0 if passage_text in relevant_texts else 0.0
            all_X.append(features)
            all_y.append(label)
            group_size += 1

        groups.append(group_size)

    X = np.vstack(all_X).astype(np.float32)
    y = np.array(all_y, dtype=np.float32)

    print(f"✅ Feature matrix: {X.shape}")
    print(f"   Positive labels: {int(y.sum())} / {len(y)}")
    print(f"   Groups (queries): {len(groups)}")

    # Save
    with open(save_path, "wb") as f:
        pickle.dump({"X": X, "y": y, "groups": groups}, f)
    print(f"✅ Saved features to {save_path}")
    log.info(f"Features saved: X={X.shape}, positives={int(y.sum())}")

    return X, y, groups


if __name__ == "__main__":
    import faiss

    print("Loading data...")
    train_df = pd.read_csv(PROCESSED_DIR / "train.csv")
    eval_df = pd.read_csv(PROCESSED_DIR / "eval.csv")
    all_df = pd.concat([train_df, eval_df], ignore_index=True)

    # Load BM25 index
    print("Loading BM25 index...")
    with open(INDEX_DIR / "bm25_index.pkl", "rb") as f:
        bm25_data = pickle.load(f)
    bm25 = bm25_data["index"]
    passages = bm25_data["passages"]
    passage_ids = bm25_data["passage_ids"]

    # Load embedding model
    print("Loading embedding model...")
    model = SentenceTransformer(config["model"]["embedding_model"])

    # Build train features
    X_train, y_train, groups_train = build_feature_dataset(
        train_df, bm25, passages, passage_ids, model,
        split="train", max_queries=2000
    )

    # Build eval features
    X_eval, y_eval, groups_eval = build_feature_dataset(
        eval_df, bm25, passages, passage_ids, model,
        split="eval", max_queries=500
    )

    print("\n--- Feature Stats ---")
    print(f"Train: X={X_train.shape}, positives={int(y_train.sum())}")
    print(f"Eval:  X={X_eval.shape},  positives={int(y_eval.sum())}")
    print(f"\nFeature names:")
    names = ["BM25", "Dense", "PassLen", "QueryLen",
             "Overlap", "OverlapRatio", "LogPassLen",
             "LogBM25", "DenseSquared", "Combined"]
    for i, name in enumerate(names):
        print(f"  [{i}] {name:15} mean={X_train[:,i].mean():.4f}")

    print("\n✅ Module 2.1 complete!")
    print("✅ Ready for Module 2.2 — LambdaRank Neural Reranker")