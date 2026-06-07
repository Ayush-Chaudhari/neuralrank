# NeuralRank — Control Panel
# Run: make <command>

.PHONY: up down restart api mlflow logs status

# Start all Docker services (Kafka, Redis, MLflow)
up:
	docker-compose up -d
	@echo "✅ All services started"
	@echo "   Kafka  : localhost:9092"
	@echo "   Redis  : localhost:6379"
	@echo "   MLflow : http://localhost:5000"

# Stop all services
down:
	docker-compose down
	@echo "✅ All services stopped"

# Restart services
restart:
	docker-compose down && docker-compose up -d

# Start FastAPI server
api:
	python -m src.api.main

# Start MLflow UI
mlflow:
	mlflow ui --backend-store-uri sqlite:///D:/neuralrank/mlflow.db

# Show running containers
status:
	docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

# Show logs
logs:
	docker-compose logs --tail=50

# Run full pipeline from scratch
pipeline:
	python -m src.data.loader
	python -m src.retrieval.bm25
	python -m src.retrieval.dense
	python -m src.features.extractor
	python -m src.models.lambdarank
	python -m src.ranking.hybrid

# Run tests
test:
	pytest tests/ -v