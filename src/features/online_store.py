# src/features/online_store.py
"""
Redis Feature Store

Stores real-time user signals as online features.
These features are served back to the ranking model
to personalize results every 30 seconds.

Keys stored in Redis:
  user:{user_id}:click_count     - total clicks
  user:{user_id}:avg_dwell       - average dwell time
  user:{user_id}:queries         - recent queries list
  query:{query}:click_through    - CTR for this query
  query:{query}:avg_rank_clicked - avg rank users click
"""
import json
import time
import redis
import random
from datetime import datetime
from src.utils import load_config, get_logger

log = get_logger("online_store")
config = load_config()

REDIS_HOST = config["redis"]["host"]
REDIS_PORT = config["redis"]["port"]
TTL = config["redis"]["ttl_seconds"]  # 1 hour expiry


class RedisFeatureStore:
    """
    Online feature store backed by Redis.
    Provides real-time features for the ranking model.
    """

    def __init__(self):
        self.client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            decode_responses=True
        )
        # Test connection
        self.client.ping()
        print("✅ Redis feature store connected")

    # ─────────────────────────────────────────
    # Write methods — called by Kafka consumer
    # ─────────────────────────────────────────

    def record_click(self, user_id: str, query: str,
                     rank: int, dwell_time: float):
        """Records a click event into Redis."""
        pipe = self.client.pipeline()

        # User features
        user_key = f"user:{user_id}"
        pipe.hincrby(user_key, "click_count", 1)
        pipe.hincrbyfloat(user_key, "total_dwell", dwell_time)
        pipe.expire(user_key, TTL)

        # Query features
        query_key = f"query:{query[:50]}"
        pipe.hincrby(query_key, "click_count", 1)
        pipe.hincrbyfloat(query_key, "total_rank", rank)
        pipe.expire(query_key, TTL)

        # Recent clicks list (keep last 20)
        clicks_key = f"user:{user_id}:recent_clicks"
        click_data = json.dumps({
            "query": query[:50],
            "rank": rank,
            "dwell": round(dwell_time, 2),
            "ts": datetime.utcnow().isoformat()
        })
        pipe.lpush(clicks_key, click_data)
        pipe.ltrim(clicks_key, 0, 19)
        pipe.expire(clicks_key, TTL)

        pipe.execute()
        log.info(f"Recorded click: user={user_id} rank={rank}")

    def record_query(self, user_id: str, query: str):
        """Records a query event into Redis."""
        pipe = self.client.pipeline()

        # Query history
        queries_key = f"user:{user_id}:queries"
        pipe.lpush(queries_key, query[:50])
        pipe.ltrim(queries_key, 0, 49)
        pipe.expire(queries_key, TTL)

        # Query frequency
        freq_key = f"query_freq:{query[:50]}"
        pipe.incr(freq_key)
        pipe.expire(freq_key, TTL)

        pipe.execute()

    # ─────────────────────────────────────────
    # Read methods — called by ranking model
    # ─────────────────────────────────────────

    def get_user_features(self, user_id: str) -> dict:
        """
        Returns real-time user features for ranking.
        These personalize results for each user.
        """
        user_key = f"user:{user_id}"
        data = self.client.hgetall(user_key)

        click_count = int(data.get("click_count", 0))
        total_dwell = float(data.get("total_dwell", 0.0))
        avg_dwell = total_dwell / click_count if click_count > 0 else 0.0

        # Recent queries
        queries_key = f"user:{user_id}:queries"
        recent_queries = self.client.lrange(queries_key, 0, 9)

        return {
            "user_id": user_id,
            "click_count": click_count,
            "avg_dwell_time": round(avg_dwell, 2),
            "total_dwell_time": round(total_dwell, 2),
            "recent_queries": recent_queries,
            "is_active_user": click_count > 5
        }

    def get_query_features(self, query: str) -> dict:
        """
        Returns real-time query-level features.
        Shows how popular/clickable this query is.
        """
        query_key = f"query:{query[:50]}"
        data = self.client.hgetall(query_key)

        click_count = int(data.get("click_count", 0))
        total_rank = float(data.get("total_rank", 0.0))
        avg_rank = total_rank / click_count if click_count > 0 else 5.0

        freq_key = f"query_freq:{query[:50]}"
        frequency = int(self.client.get(freq_key) or 0)

        return {
            "query": query[:50],
            "click_count": click_count,
            "avg_rank_clicked": round(avg_rank, 2),
            "query_frequency": frequency,
            "is_popular_query": frequency > 3
        }

    def get_ranking_features(self, user_id: str, query: str) -> dict:
        """
        Returns combined user + query features for ranking.
        This is what gets fed to the reranker at serving time.
        """
        user_feats = self.get_user_features(user_id)
        query_feats = self.get_query_features(query)

        return {
            "user_click_count": user_feats["click_count"],
            "user_avg_dwell": user_feats["avg_dwell_time"],
            "user_is_active": int(user_feats["is_active_user"]),
            "query_ctr": query_feats["click_count"],
            "query_avg_rank": query_feats["avg_rank_clicked"],
            "query_is_popular": int(query_feats["is_popular_query"]),
        }

    def get_store_stats(self) -> dict:
        """Returns overall stats about the feature store."""
        total_keys = self.client.dbsize()
        user_keys = len(self.client.keys("user:*"))
        query_keys = len(self.client.keys("query:*"))

        return {
            "total_keys": total_keys,
            "user_keys": user_keys,
            "query_keys": query_keys,
        }


def simulate_and_store(store: RedisFeatureStore,
                       num_events: int = 100):
    """
    Simulates user events and stores them in Redis.
    Shows how the feature store fills up with real data.
    """
    QUERIES = [
        "what was the impact of the manhattan project",
        "how does the immune system work",
        "causes of inflation in economy",
        "machine learning algorithms explained",
        "climate change effects on ocean",
    ]
    USERS = [f"user_{i:03d}" for i in range(1, 11)]

    print(f"\n🎭 Simulating {num_events} events into Redis...")

    for i in range(num_events):
        user_id = random.choice(USERS)
        query = random.choice(QUERIES)

        # Record query
        store.record_query(user_id, query)

        # Record click with realistic dwell time
        rank = random.randint(1, 10)
        dwell = random.uniform(5, 120) if rank <= 3 else random.uniform(1, 20)
        store.record_click(user_id, query, rank, dwell)

        if (i + 1) % 20 == 0:
            print(f"   Stored {i + 1}/{num_events} events...")

    print(f"✅ Stored {num_events} events in Redis")


if __name__ == "__main__":
    print("=" * 50)
    print("  NeuralRank — Redis Feature Store")
    print("=" * 50)

    # Connect
    store = RedisFeatureStore()

    # Simulate events
    simulate_and_store(store, num_events=100)

    # Show stats
    stats = store.get_store_stats()
    print(f"\n📊 Feature Store Stats:")
    print(f"   Total keys : {stats['total_keys']}")
    print(f"   User keys  : {stats['user_keys']}")
    print(f"   Query keys : {stats['query_keys']}")

    # Show sample user features
    print(f"\n👤 Sample User Features:")
    for user_id in ["user_001", "user_005", "user_010"]:
        feats = store.get_user_features(user_id)
        print(f"\n  {user_id}:")
        print(f"    Clicks     : {feats['click_count']}")
        print(f"    Avg dwell  : {feats['avg_dwell_time']}s")
        print(f"    Active user: {feats['is_active_user']}")
        print(f"    Recent queries: {feats['recent_queries'][:2]}")

    # Show sample query features
    print(f"\n🔍 Sample Query Features:")
    for query in ["how does the immune system work",
                  "causes of inflation in economy"]:
        feats = store.get_query_features(query)
        print(f"\n  '{query[:40]}':")
        print(f"    Clicks    : {feats['click_count']}")
        print(f"    Avg rank  : {feats['avg_rank_clicked']}")
        print(f"    Popular   : {feats['is_popular_query']}")

    # Show ranking features for a user+query pair
    print(f"\n⚡ Live Ranking Features (user_001 + immune system query):")
    ranking_feats = store.get_ranking_features(
        "user_001",
        "how does the immune system work"
    )
    for k, v in ranking_feats.items():
        print(f"   {k:25} = {v}")

    print("\n" + "=" * 50)
    print("✅ Module 3.2 complete!")
    print("✅ Ready for Module 3.3 — FastAPI Server")
    print("=" * 50)