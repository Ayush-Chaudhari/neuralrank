# ⚡ NeuralRank — AI Search Ranking Engine

> A production-grade AI search ranking engine built from scratch — combining classical IR, semantic search, and neural reranking with real-time personalization.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.6-red)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green)
![Next.js](https://img.shields.io/badge/Next.js-14-black)
![Docker](https://img.shields.io/badge/Docker-Compose-blue)
![MLflow](https://img.shields.io/badge/MLflow-2.13-orange)

---

## 📊 Results

| Model | NDCG@10 | Recall@10 | vs Baseline |
|-------|---------|-----------|-------------|
| BM25 baseline | 0.3414 | 0.5807 | — |
| LambdaRank (neural) | 0.4126 | — | +20.9% |
| Dense (FAISS) | 0.6214 | 0.9232 | +82.0% |
| **Hybrid (final)** | **0.6048** | **0.9023** | **+77.1%** |

> Trained and evaluated on **MS MARCO** — the industry standard search benchmark used by Microsoft, Google, and top research labs.

---

## 🏗️ System Architecture
<img width="347" height="523" alt="image" src="https://github.com/user-attachments/assets/32a7ad6a-a105-43f7-a505-6452d0f8ce23" />

---

## 🛠️ Tech Stack

### Machine Learning
| Component | Technology | Purpose |
|-----------|-----------|---------|
| Embeddings | `sentence-transformers` (all-MiniLM-L6-v2) | Text → vectors |
| Vector search | `FAISS` | Semantic retrieval |
| Keyword search | `BM25` (rank-bm25) | Keyword retrieval |
| Neural reranker | `PyTorch` LambdaRank | Learn to rank |
| Query expansion | `LangChain` + HuggingFace | RAG pipeline |

### Real-time Systems
| Component | Technology | Purpose |
|-----------|-----------|---------|
| Message queue | `Apache Kafka` | User signal ingestion |
| Feature store | `Redis` | Online feature serving |
| Orchestration | `Docker Compose` | Container management |

### MLOps
| Component | Technology | Purpose |
|-----------|-----------|---------|
| Experiment tracking | `MLflow` | Metrics, params, artifacts |
| Drift monitoring | `Evidently AI` | Data drift detection |
| Data versioning | `DVC` | Dataset versioning |

### API & Frontend
| Component | Technology | Purpose |
|-----------|-----------|---------|
| REST API | `FastAPI` | Model serving |
| Search UI | `Next.js` + Tailwind | Demo interface |

---

## 📁 Project Structure
<img width="405" height="597" alt="image" src="https://github.com/user-attachments/assets/51cc4d44-ff75-4b43-b3a9-632b179f22a3" />

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11
- Docker Desktop
- Node.js 18+
- 8GB RAM recommended

### 1. Clone and setup environment
```bash
git clone https://github.com/Ayush-Chaudhari/neuralrank.git
cd neuralrank
conda create -n neuralrank python=3.11 -y
conda activate neuralrank
pip install -r requirements.txt
```

### 2. Start infrastructure
```bash
docker-compose up -d
```

### 3. Build data pipeline (first time only)
```bash
# Download MS MARCO dataset
python -m src.data.loader

# Build retrieval indexes
python -m src.retrieval.bm25
python -m src.retrieval.dense

# Build training features
python -m src.features.extractor

# Train LambdaRank model
python -m src.models.lambdarank

# Evaluate hybrid reranker
python -m src.ranking.hybrid
```

### 4. Start API server
```bash
python -m src.api.main
# API runs at http://localhost:8000
# Swagger docs at http://localhost:8000/docs
```

### 5. Start frontend
```bash
cd frontend
npm install
npm run dev
# UI runs at http://localhost:3000
```

### 6. MLflow dashboard
```bash
mlflow ui --backend-store-uri sqlite:///D:/neuralrank/mlflow.db
# Dashboard at http://localhost:5000
```

---

## 🔌 API Endpoints
POST /search          Search with hybrid ranking
POST /feedback        Record user click feedback
GET  /stats           System statistics
GET  /user/{user_id}  User features from Redis
GET  /                Health check
GET  /docs            Swagger UI
### Example search request
```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "impact of manhattan project", "user_id": "user_001"}'
```

---

## 🧠 How LambdaRank Works

LambdaRank is a neural Learning-to-Rank model that directly optimizes NDCG:

1. For every query, look at all (relevant, irrelevant) passage pairs
2. Train the network to score relevant passages higher
3. Weight each pair by |ΔNDCG| — pairs that impact ranking quality more get higher gradients
4. This directly optimizes the metric we care about
Input (10 features) → 128 → 64 → 32 → 1 (relevance score)
---

## 📈 MLflow Experiment Tracking

Every training run is tracked with:
- **Parameters:** learning rate, epochs, batch size, model weights
- **Metrics:** NDCG@10 per epoch, loss curve, baseline comparisons
- **Artifacts:** training curve PNG, feature importance plot, model checkpoint

---

## 🔍 Drift Monitoring

Evidently AI monitors 10 features for distribution shift using the KS-statistic:
- KS > 0.15 per feature = drift detected
- >30% features drifted = retrain alert
- HTML report generated at `reports/drift_report.html`

---

## 👤 About

Built by **Ayush Chaudhari** as a portfolio project demonstrating production-grade AI engineering skills.

**Skills demonstrated:**
- Learning to Rank (LTR) with PyTorch
- Dense retrieval with FAISS
- Real-time ML systems with Kafka + Redis
- MLOps with MLflow + Evidently
- Full-stack AI with FastAPI + Next.js
- Containerization with Docker

---

## 📄 License

MIT License — free to use and learn from.
