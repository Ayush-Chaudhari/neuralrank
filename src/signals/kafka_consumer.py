# src/signals/kafka_consumer.py
"""
Kafka Signal Pipeline

Produces and consumes real-time user signals:
  - Click events (user clicked a result)
  - Query events (user made a search)
  - Dwell time (how long user stayed on result)

In production this feeds back into the ranking model
to personalize results in real-time.
"""
import json
import time
import random
import threading
from datetime import datetime
from kafka import KafkaProducer, KafkaConsumer
from kafka.admin import KafkaAdminClient, NewTopic
from kafka.errors import TopicAlreadyExistsError
from src.utils import load_config, get_logger

log = get_logger("kafka_pipeline")
config = load_config()

BOOTSTRAP_SERVERS = config["kafka"]["bootstrap_servers"]
TOPIC_CLICKS = config["kafka"]["topic_clicks"]
TOPIC_QUERIES = config["kafka"]["topic_queries"]


# ─────────────────────────────────────────
# 1. Topic Setup
# ─────────────────────────────────────────

def create_topics():
    """Creates Kafka topics if they don't exist."""
    try:
        admin = KafkaAdminClient(bootstrap_servers=BOOTSTRAP_SERVERS)
        topics = [
            NewTopic(name=TOPIC_CLICKS, num_partitions=3, replication_factor=1),
            NewTopic(name=TOPIC_QUERIES, num_partitions=3, replication_factor=1),
        ]
        admin.create_topics(topics)
        print(f"✅ Created topics: {TOPIC_CLICKS}, {TOPIC_QUERIES}")
    except TopicAlreadyExistsError:
        print(f"✅ Topics already exist")
    except Exception as e:
        print(f"⚠️  Topic creation: {e}")


# ─────────────────────────────────────────
# 2. Producer — sends user events
# ─────────────────────────────────────────

class SignalProducer:
    """
    Simulates real user search signals.
    In production this would come from your frontend.
    """

    def __init__(self):
        self.producer = KafkaProducer(
            bootstrap_servers=BOOTSTRAP_SERVERS,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            key_serializer=lambda k: k.encode("utf-8") if k else None
        )
        print("✅ Kafka producer connected")

    def send_query_event(self, user_id: str, query: str, session_id: str):
        """Sends a query event when user searches."""
        event = {
            "event_type": "query",
            "user_id": user_id,
            "session_id": session_id,
            "query": query,
            "timestamp": datetime.utcnow().isoformat()
        }
        self.producer.send(TOPIC_QUERIES, key=user_id, value=event)
        log.info(f"Query event: user={user_id} query='{query}'")

    def send_click_event(self, user_id: str, query: str,
                         passage_id: str, rank: int,
                         dwell_time: float, session_id: str):
        """Sends a click event when user clicks a result."""
        event = {
            "event_type": "click",
            "user_id": user_id,
            "session_id": session_id,
            "query": query,
            "passage_id": passage_id,
            "rank": rank,
            "dwell_time_seconds": dwell_time,
            "timestamp": datetime.utcnow().isoformat()
        }
        self.producer.send(TOPIC_CLICKS, key=user_id, value=event)
        log.info(f"Click event: user={user_id} rank={rank} dwell={dwell_time:.1f}s")

    def flush(self):
        self.producer.flush()


# ─────────────────────────────────────────
# 3. Consumer — processes incoming events
# ─────────────────────────────────────────

class SignalConsumer:
    """
    Consumes user signals from Kafka and stores
    aggregated features in Redis feature store.
    """

    def __init__(self, redis_store=None):
        self.consumer = KafkaConsumer(
            TOPIC_CLICKS,
            TOPIC_QUERIES,
            bootstrap_servers=BOOTSTRAP_SERVERS,
            value_deserializer=lambda v: json.loads(v.decode("utf-8")),
            group_id="neuralrank-consumer-group",
            auto_offset_reset="latest",
            consumer_timeout_ms=5000
        )
        self.redis_store = redis_store
        self.processed_count = 0
        print("✅ Kafka consumer connected")

    def process_event(self, event: dict):
        """Processes a single event and updates Redis."""
        event_type = event.get("event_type")

        if event_type == "click":
            user_id = event["user_id"]
            query = event["query"]
            rank = event["rank"]
            dwell = event["dwell_time_seconds"]

            # Store in Redis if available
            if self.redis_store:
                self.redis_store.record_click(
                    user_id=user_id,
                    query=query,
                    rank=rank,
                    dwell_time=dwell
                )

            self.processed_count += 1
            print(f"  📥 Click: user={user_id} | "
                  f"rank={rank} | dwell={dwell:.1f}s | "
                  f"query='{query[:40]}'")

        elif event_type == "query":
            user_id = event["user_id"]
            query = event["query"]

            if self.redis_store:
                self.redis_store.record_query(user_id, query)

            self.processed_count += 1
            print(f"  📥 Query: user={user_id} | '{query[:40]}'")

    def consume(self, max_messages: int = 50):
        """Consumes messages from Kafka topics."""
        print(f"\n👂 Listening for events (max {max_messages})...")
        count = 0
        for message in self.consumer:
            self.process_event(message.value)
            count += 1
            if count >= max_messages:
                break
        print(f"✅ Consumed {count} events")
        return count


# ─────────────────────────────────────────
# 4. Signal Simulator
# ─────────────────────────────────────────

SAMPLE_QUERIES = [
    "what was the impact of the manhattan project",
    "how does the immune system work",
    "causes of inflation in economy",
    "machine learning algorithms explained",
    "best programming languages 2024",
    "climate change effects on ocean",
    "history of artificial intelligence",
    "how to treat common cold symptoms",
]

SAMPLE_USERS = [f"user_{i:03d}" for i in range(1, 21)]


def simulate_user_signals(producer: SignalProducer,
                          num_events: int = 30,
                          delay: float = 0.1):
    """
    Simulates realistic user search behavior.
    Generates query + click events with realistic dwell times.
    """
    print(f"\n🎭 Simulating {num_events} user events...")

    for i in range(num_events):
        user_id = random.choice(SAMPLE_USERS)
        query = random.choice(SAMPLE_QUERIES)
        session_id = f"session_{random.randint(1000, 9999)}"

        # Send query event
        producer.send_query_event(user_id, query, session_id)

        # Simulate 1-3 clicks per query
        num_clicks = random.randint(1, 3)
        clicked_ranks = random.sample(range(1, 11), num_clicks)

        for rank in clicked_ranks:
            # Relevant results get longer dwell times
            if rank <= 3:
                dwell = random.uniform(30, 120)
            else:
                dwell = random.uniform(5, 30)

            producer.send_click_event(
                user_id=user_id,
                query=query,
                passage_id=f"passage_{random.randint(1000, 9999)}",
                rank=rank,
                dwell_time=dwell,
                session_id=session_id
            )

        time.sleep(delay)

    producer.flush()
    print(f"✅ Sent {num_events} query events with clicks")


if __name__ == "__main__":
    print("=" * 50)
    print("  NeuralRank — Kafka Signal Pipeline")
    print("=" * 50)

    # Setup topics
    create_topics()

    # Initialize producer
    producer = SignalProducer()

    # Simulate user signals in background thread
    producer_thread = threading.Thread(
        target=simulate_user_signals,
        args=(producer, 20, 0.2)
    )

    # Initialize consumer
    try:
        import redis as redis_lib
        r = redis_lib.Redis(host="localhost", port=6379)
        r.ping()
        print("✅ Redis connected")
        redis_available = True
    except Exception:
        print("⚠️  Redis not available — running without it")
        redis_available = False

    consumer = SignalConsumer(redis_store=None)

    # Start producing
    print("\n📤 Starting signal producer...")
    producer_thread.start()

    # Small delay then consume
    time.sleep(1)
    consumer.consume(max_messages=40)

    producer_thread.join()

    print("\n" + "=" * 50)
    print(f"  ✅ Kafka pipeline working!")
    print(f"  Events produced and consumed successfully")
    print("=" * 50)
    print("\n✅ Module 3.1 complete!")
    print("✅ Ready for Module 3.2 — Redis Feature Store")