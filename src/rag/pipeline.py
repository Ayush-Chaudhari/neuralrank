# src/rag/pipeline.py
"""
RAG Pipeline — Query Expansion + Context Generation

What this does:
  1. Takes a raw user query
  2. Uses an LLM to expand it with related terms
  3. Retrieves passages using the expanded query
  4. Uses LLM to generate a final answer grounded in retrieved passages

We use HuggingFace's free inference API so no GPU needed.
Falls back to rule-based expansion if API is unavailable.
"""
import os
import re
import time
import requests
import numpy as np
import pickle
import faiss
from pathlib import Path
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from src.utils import load_config, get_logger

log = get_logger("rag_pipeline")
config = load_config()

ROOT_DIR = Path(__file__).parent.parent.parent
INDEX_DIR = ROOT_DIR / "data" / "indexes"

# ─────────────────────────────────────────
# 1. Query Expansion (Rule-based fallback)
# ─────────────────────────────────────────

EXPANSION_RULES = {
    "what": ["definition", "explanation", "meaning"],
    "how": ["method", "process", "steps", "procedure"],
    "why": ["reason", "cause", "explanation"],
    "when": ["time", "date", "period", "history"],
    "where": ["location", "place", "region"],
    "who": ["person", "people", "individual"],
    "best": ["top", "recommended", "optimal"],
    "impact": ["effect", "consequence", "result", "influence"],
    "symptoms": ["signs", "indicators", "manifestations"],
    "treatment": ["therapy", "cure", "remedy", "medication"],
    "cause": ["reason", "factor", "source", "origin"],
}


def rule_based_expansion(query: str) -> str:
    """
    Simple rule-based query expansion.
    Used as fallback when LLM API is unavailable.
    """
    tokens = query.lower().split()
    expansions = []
    for token in tokens:
        if token in EXPANSION_RULES:
            expansions.extend(EXPANSION_RULES[token][:2])

    if expansions:
        expanded = query + " " + " ".join(expansions[:4])
    else:
        expanded = query

    return expanded


def expand_query_with_llm(query: str) -> str:
    """
    Expands query using HuggingFace free inference API.
    Falls back to rule-based if API fails.

    Query expansion improves retrieval by adding:
    - Synonyms
    - Related terms
    - Domain-specific vocabulary
    """
    try:
        # Use HuggingFace free inference API
        API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-base"
        HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN", "")

        headers = {}
        if HF_TOKEN:
            headers["Authorization"] = f"Bearer {HF_TOKEN}"

        prompt = (
            f"Expand this search query with 3-5 related terms for better retrieval. "
            f"Query: '{query}'. "
            f"Output only the expanded query, nothing else."
        )

        response = requests.post(
            API_URL,
            headers=headers,
            json={"inputs": prompt, "parameters": {"max_new_tokens": 50}},
            timeout=10
        )

        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and result:
                expanded = result[0].get("generated_text", query)
                # Clean up
                expanded = expanded.replace(prompt, "").strip()
                if len(expanded) > 10:
                    log.info(f"LLM expanded: '{query}' → '{expanded[:80]}'")
                    return expanded

        # Fallback
        return rule_based_expansion(query)

    except Exception as e:
        log.warning(f"LLM expansion failed: {e} — using rule-based")
        return rule_based_expansion(query)


# ─────────────────────────────────────────
# 2. Context-Aware Answer Generation
# ─────────────────────────────────────────

def generate_answer(query: str, passages: list) -> str:
    """
    Generates a grounded answer from retrieved passages.
    Uses HuggingFace API with fallback to extractive answer.
    """
    # Build context from top passages
    context = "\n\n".join([
        f"Passage {i+1}: {p[:300]}"
        for i, p in enumerate(passages[:3])
    ])

    try:
        API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-base"
        HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN", "")

        headers = {}
        if HF_TOKEN:
            headers["Authorization"] = f"Bearer {HF_TOKEN}"

        prompt = (
            f"Answer the question based on the passages below.\n\n"
            f"Question: {query}\n\n"
            f"{context}\n\n"
            f"Answer:"
        )

        response = requests.post(
            API_URL,
            headers=headers,
            json={"inputs": prompt[:1000],
                  "parameters": {"max_new_tokens": 100}},
            timeout=15
        )

        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and result:
                answer = result[0].get("generated_text", "")
                if answer and len(answer) > 10:
                    return answer.strip()

    except Exception as e:
        log.warning(f"Answer generation failed: {e}")

    # Extractive fallback — return most relevant sentence
    if passages:
        first_passage = passages[0]
        sentences = first_passage.split(". ")
        query_words = set(query.lower().split())
        best_sentence = max(
            sentences,
            key=lambda s: len(set(s.lower().split()) & query_words)
        )
        return best_sentence.strip()

    return "No answer found."


# ─────────────────────────────────────────
# 3. Full RAG Pipeline
# ─────────────────────────────────────────

class RAGPipeline:
    """
    Full Retrieval-Augmented Generation pipeline.

    Flow:
      query → expand → retrieve (BM25 + dense) → rerank → generate answer
    """

    def __init__(self):
        print("🔧 Initializing RAG Pipeline...")

        # Load BM25
        bm25_path = INDEX_DIR / "bm25_index.pkl"
        with open(bm25_path, "rb") as f:
            bm25_data = pickle.load(f)
        self.bm25 = bm25_data["index"]
        self.passages = bm25_data["passages"]
        self.passage_ids = bm25_data["passage_ids"]
        print(f"✅ BM25 loaded — {len(self.passages)} passages")

        # Load FAISS
        faiss_path = INDEX_DIR / "faiss_index.bin"
        self.faiss_index = faiss.read_index(str(faiss_path))
        faiss_map_path = INDEX_DIR / "passage_map.pkl"
        with open(faiss_map_path, "rb") as f:
            self.passage_map = pickle.load(f)
        print(f"✅ FAISS loaded — {self.faiss_index.ntotal} vectors")

        # Load embedding model
        self.model = SentenceTransformer(config["model"]["embedding_model"])
        print(f"✅ Embedding model loaded")

        print("✅ RAG Pipeline ready!\n")

    def retrieve(self, query: str, top_k: int = 10) -> list:
        """Hybrid retrieval: BM25 + dense, merged by score."""

        # BM25 retrieval
        tokens = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokens)
        top_bm25 = np.argsort(bm25_scores)[::-1][:top_k * 2]

        # Dense retrieval
        query_vec = self.model.encode(
            [query], convert_to_numpy=True
        ).astype(np.float32)
        faiss.normalize_L2(query_vec)
        dense_scores, dense_idx = self.faiss_index.search(query_vec, top_k * 2)

        # Merge candidates
        candidates = {}

        for idx in top_bm25:
            text = self.passages[idx]
            candidates[text] = {
                "passage": text,
                "bm25_score": float(bm25_scores[idx]),
                "dense_score": 0.0
            }

        for score, idx in zip(dense_scores[0], dense_idx[0]):
            if idx >= 0:
                text = self.passage_map["passages"][idx]
                if text in candidates:
                    candidates[text]["dense_score"] = float(score)
                else:
                    candidates[text] = {
                        "passage": text,
                        "bm25_score": 0.0,
                        "dense_score": float(score)
                    }

        # Hybrid score: weighted combination
        results = []
        for text, scores in candidates.items():
            hybrid = (
                0.3 * scores["bm25_score"] / 20.0 +  # normalize BM25
                0.7 * scores["dense_score"]
            )
            results.append({
                "passage": text,
                "bm25_score": scores["bm25_score"],
                "dense_score": scores["dense_score"],
                "hybrid_score": hybrid
            })

        # Sort by hybrid score
        results.sort(key=lambda x: x["hybrid_score"], reverse=True)
        return results[:top_k]

    def run(self, query: str, verbose: bool = True) -> dict:
        """
        Full RAG pipeline:
        1. Expand query
        2. Retrieve passages
        3. Generate answer
        """
        if verbose:
            print(f"\n{'='*55}")
            print(f"  Query: {query}")
            print(f"{'='*55}")

        # Step 1: Query expansion
        expanded = expand_query_with_llm(query)
        if verbose:
            print(f"\n📝 Expanded query:")
            print(f"   {expanded}")

        # Step 2: Retrieve
        results = self.retrieve(expanded, top_k=5)
        if verbose:
            print(f"\n📚 Top retrieved passages:")
            for i, r in enumerate(results[:3], 1):
                print(f"\n  [{i}] Score: {r['hybrid_score']:.4f}")
                print(f"       {r['passage'][:150]}...")

        # Step 3: Generate answer
        passage_texts = [r["passage"] for r in results]
        answer = generate_answer(query, passage_texts)
        if verbose:
            print(f"\n💡 Generated Answer:")
            print(f"   {answer}")
            print(f"\n{'='*55}")

        return {
            "query": query,
            "expanded_query": expanded,
            "retrieved_passages": results,
            "answer": answer
        }


if __name__ == "__main__":
    # Initialize pipeline
    rag = RAGPipeline()

    # Test with several queries
    test_queries = [
        "what was the impact of the manhattan project?",
        "how does the immune system fight viruses?",
        "what causes inflation in an economy?",
    ]

    for query in test_queries:
        result = rag.run(query, verbose=True)
        time.sleep(1)  # be nice to API

    print("\n✅ Module 2.3 complete!")
    print("✅ Ready for Module 2.4 — Hybrid Reranker")