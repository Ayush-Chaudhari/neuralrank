import sys
print(f"Python version: {sys.version}")

print("\n--- Testing imports ---")

import torch
print(f"✅ PyTorch: {torch.__version__}")

import faiss
print(f"✅ FAISS: {faiss.__version__}")

import numpy as np
print(f"✅ NumPy: {np.__version__}")

import pandas as pd
print(f"✅ Pandas: {pd.__version__}")

from sentence_transformers import SentenceTransformer
print(f"✅ SentenceTransformers: imported")

from rank_bm25 import BM25Okapi
print(f"✅ BM25: imported")

import fastapi
print(f"✅ FastAPI: {fastapi.__version__}")

import mlflow
print(f"✅ MLflow: {mlflow.__version__}")

from src.utils import load_config, get_logger
config = load_config()
print(f"✅ Config loaded: {config['project']['name']} v{config['project']['version']}")

log = get_logger("setup_test")

print("\n--- Running smoke tests ---")

# BM25 test
corpus = [["hello", "world"], ["neural", "ranking", "search"], ["machine", "learning"]]
bm25 = BM25Okapi(corpus)
scores = bm25.get_scores(["neural", "search"])
print(f"✅ BM25 smoke test: top score = {max(scores):.4f}")

# Embedding + FAISS test
print("   Loading embedding model (first time downloads ~90MB, please wait...)")
model = SentenceTransformer("all-MiniLM-L6-v2")
sentences = [
    "neural search ranking system",
    "machine learning recommendation",
    "deep learning for information retrieval"
]
embeddings = model.encode(sentences)
print(f"✅ Embeddings shape: {embeddings.shape}")

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings.astype(np.float32))
query_vec = model.encode(["search ranking"])
D, I = index.search(query_vec.astype(np.float32), 3)
print(f"✅ FAISS search: top result index={I[0][0]}, distance={D[0][0]:.4f}")

# PyTorch test
x = torch.randn(32, 256)
linear = torch.nn.Linear(256, 1)
out = linear(x)
print(f"✅ PyTorch forward pass: input={x.shape}, output={out.shape}")

print("\n=============================")
print("✅ ALL SYSTEMS GO!")
print("✅ Ready for Module 1.2 — Data Pipeline")
print("=============================")