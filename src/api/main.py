# src/api/main.py
"""
NeuralRank FastAPI Server

Endpoints:
  GET  /                    - health check
  POST /search              - main search endpoint
  POST /feedback            - record user click feedback
  GET  /stats               - system stats
  GET  /user/{user_id}      - user features
"""
import time
import pickle
import faiss
import numpy as np
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from src.utils import load_config, get_logger
from src.models.lambdarank import load_model
from src.features.extractor import compute_features, tokenize
from src.features.online_store import RedisFeatureStore
from src.rag.pipeline import expand_query_with_llm

log = get_logger("api")
config = load_config()

ROOT_DIR = Path(__file__).parent.parent.parent
INDEX_DIR = ROOT_DIR / "data" / "indexes"

# ─────────────────────────────────────────
# Global components (loaded once at startup)
# ─────────────────────────────────────────
components = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Loads all ML components when server starts."""
    print("🚀 Loading NeuralRank components...")

    # BM25
    with open(INDEX_DIR / "bm25_index.pkl", "rb") as f:
        bm25_data = pickle.load(f)
    components["bm25"] = bm25_data["index"]
    components["passages"] = bm25_data["passages"]
    components["passage_ids"] = bm25_data["passage_ids"]
    print(f"✅ BM25 loaded — {len(components['passages'])} passages")

    # FAISS
    components["faiss_index"] = faiss.read_index(
        str(INDEX_DIR / "faiss_index.bin")
    )
    with open(INDEX_DIR / "passage_map.pkl", "rb") as f:
        components["passage_map"] = pickle.load(f)
    print(f"✅ FAISS loaded")

    # Embedding model
    components["embed_model"] = SentenceTransformer(
        config["model"]["embedding_model"]
    )
    print(f"✅ Embedding model loaded")

    # LambdaRank
    components["ranker"] = load_model(input_dim=10)
    print(f"✅ LambdaRank loaded")

    # Redis feature store
    try:
        components["store"] = RedisFeatureStore()
        print(f"✅ Redis connected")
    except Exception:
        components["store"] = None
        print(f"⚠️  Redis unavailable — running without personalization")

    print("✅ All components loaded — server ready!\n")
    yield
    print("Shutting down...")


# ─────────────────────────────────────────
# FastAPI App
# ─────────────────────────────────────────

app = FastAPI(
    title="NeuralRank API",
    description="AI-powered search ranking engine",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────
# Request / Response Models
# ─────────────────────────────────────────

class SearchRequest(BaseModel):
    query: str
    user_id: str = "anonymous"
    top_k: int = 10
    expand_query: bool = True


class SearchResult(BaseModel):
    rank: int
    passage: str
    score: float
    bm25_score: float
    dense_score: float
    neural_score: float


class SearchResponse(BaseModel):
    query: str
    expanded_query: str
    results: list[SearchResult]
    latency_ms: float
    num_candidates: int


class FeedbackRequest(BaseModel):
    user_id: str
    query: str
    passage_id: str
    rank: int
    dwell_time: float


# ─────────────────────────────────────────
# Core ranking function
# ─────────────────────────────────────────

def rank_results(query: str, user_id: str, top_k: int) -> tuple:
    """Full hybrid ranking pipeline."""
    import torch
    from src.models.lambdarank import RankerNet

    bm25 = components["bm25"]
    passages = components["passages"]
    faiss_index = components["faiss_index"]
    passage_map = components["passage_map"]
    embed_model = components["embed_model"]
    ranker = components["ranker"]
    store = components["store"]

    # BM25 candidates
    tokens = tokenize(query)
    bm25_scores_arr = bm25.get_scores(tokens)
    top_bm25_idx = np.argsort(bm25_scores_arr)[::-1][:100]
    bm25_candidates = {
        passages[i]: float(bm25_scores_arr[i])
        for i in top_bm25_idx
    }

    # Dense candidates
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

    # Merge
    all_texts = set(bm25_candidates.keys()) | set(dense_candidates.keys())

    # Get online features from Redis
    online_feats = {}
    if store:
        try:
            online_feats = store.get_ranking_features(user_id, query)
        except Exception:
            pass

    # Build feature matrix
    candidate_list = []
    feature_matrix = []

    for text in all_texts:
        b = bm25_candidates.get(text, 0.0)
        d = dense_candidates.get(text, 0.0)
        features = compute_features(query, text, b, d)
        feature_matrix.append(features)
        candidate_list.append({
            "passage": text,
            "bm25_score": b,
            "dense_score": d,
        })

    # Neural scoring
    ranker.eval()
    with torch.no_grad():
        X = torch.FloatTensor(np.vstack(feature_matrix))
        neural_scores = ranker(X).numpy()

    # Normalize + combine
    def norm(arr):
        mn, mx = arr.min(), arr.max()
        return (arr - mn) / (mx - mn + 1e-8)

    bm25_arr = norm(np.array([c["bm25_score"] for c in candidate_list]))
    dense_arr = norm(np.array([c["dense_score"] for c in candidate_list]))
    neural_arr = norm(neural_scores)

    # Personalization boost
    user_boost = 0.0
    if online_feats.get("user_is_active"):
        user_boost = 0.02

    final_scores = (
        0.2 * bm25_arr +
        0.5 * dense_arr +
        0.3 * neural_arr +
        user_boost
    )

    for i, cand in enumerate(candidate_list):
        cand["neural_score"] = float(neural_scores[i])
        cand["final_score"] = float(final_scores[i])

    candidate_list.sort(key=lambda x: x["final_score"], reverse=True)
    return candidate_list[:top_k], len(candidate_list)


# ─────────────────────────────────────────
# API Endpoints
# ─────────────────────────────────────────

@app.get("/")
async def health_check():
    return {
        "status": "healthy",
        "service": "NeuralRank API",
        "version": "1.0.0",
        "components": {
            "bm25": "bm25" in components,
            "faiss": "faiss_index" in components,
            "ranker": "ranker" in components,
            "redis": components.get("store") is not None,
        }
    }


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """
    Main search endpoint.
    Takes a query, returns ranked results.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    start = time.time()

    # Query expansion
    expanded = request.query
    if request.expand_query:
        expanded = expand_query_with_llm(request.query)

    # Rank
    try:
        results, num_candidates = rank_results(
            expanded, request.user_id, request.top_k
        )
    except Exception as e:
        log.error(f"Ranking error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    latency = (time.time() - start) * 1000

    # Record query in Redis
    store = components.get("store")
    if store:
        try:
            store.record_query(request.user_id, request.query)
        except Exception:
            pass

    log.info(f"Search: '{request.query}' | "
             f"user={request.user_id} | "
             f"latency={latency:.1f}ms")

    return SearchResponse(
        query=request.query,
        expanded_query=expanded,
        results=[
            SearchResult(
                rank=i + 1,
                passage=r["passage"],
                score=round(r["final_score"], 4),
                bm25_score=round(r["bm25_score"], 4),
                dense_score=round(r["dense_score"], 4),
                neural_score=round(r["neural_score"], 4),
            )
            for i, r in enumerate(results)
        ],
        latency_ms=round(latency, 2),
        num_candidates=num_candidates
    )


@app.post("/feedback")
async def feedback(request: FeedbackRequest):
    """Records user click feedback into Redis feature store."""
    store = components.get("store")
    if store:
        try:
            store.record_click(
                user_id=request.user_id,
                query=request.query,
                rank=request.rank,
                dwell_time=request.dwell_time
            )
            return {"status": "recorded"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    return {"status": "redis_unavailable"}


@app.get("/stats")
async def get_stats():
    """Returns system statistics."""
    store = components.get("store")
    store_stats = {}
    if store:
        try:
            store_stats = store.get_store_stats()
        except Exception:
            pass

    return {
        "passages_indexed": len(components.get("passages", [])),
        "faiss_vectors": components["faiss_index"].ntotal
        if "faiss_index" in components else 0,
        "feature_store": store_stats,
    }


@app.get("/user/{user_id}")
async def get_user(user_id: str):
    """Returns real-time features for a user."""
    store = components.get("store")
    if not store:
        raise HTTPException(status_code=503, detail="Redis unavailable")
    try:
        return store.get_user_features(user_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )