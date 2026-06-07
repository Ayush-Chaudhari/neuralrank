# src/data/loader.py
import os
import json
import pandas as pd
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
from src.utils import load_config, get_logger

log = get_logger("data_loader")
config = load_config()

ROOT_DIR = Path(__file__).parent.parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)


def download_msmarco(num_train=10000, num_eval=1000):
    """
    Downloads MS MARCO passage ranking dataset.
    Only downloads a small subset so it's fast and free.
    Full dataset is 8.8M passages — we use 100K for the project.
    """
    log.info("Downloading MS MARCO dataset...")
    print("\n📥 Downloading MS MARCO passages (this may take 2-3 minutes)...")

    # Load the passage ranking dataset
    dataset = load_dataset(
        "ms_marco",
        "v2.1",
        split="train",
        trust_remote_code=True
    )

    print(f"✅ Dataset loaded — total records: {len(dataset)}")
    log.info(f"Total records in dataset: {len(dataset)}")

    return dataset


def process_dataset(dataset, num_train=10000, num_eval=1000):
    """
    Processes raw MS MARCO into clean query-passage pairs.
    Each record has:
        - query_id: unique query identifier
        - query: the search query text
        - passage_id: unique passage identifier
        - passage: the document/passage text
        - label: 1 if relevant, 0 if not relevant
    """
    print(f"\n⚙️  Processing {num_train + num_eval} records...")
    log.info(f"Processing {num_train + num_eval} records")

    records = []
    total = min(num_train + num_eval, len(dataset))

    for i in tqdm(range(total), desc="Processing records"):
        item = dataset[i]

        query = item["query"]
        query_id = item["query_id"]
        passages = item["passages"]

        # Each query has multiple passages with relevance labels
        for j, (passage_text, is_selected) in enumerate(
            zip(passages["passage_text"], passages["is_selected"])
        ):
            records.append({
                "query_id": query_id,
                "query": query,
                "passage_id": f"{query_id}_{j}",
                "passage": passage_text,
                "label": int(is_selected)  # 1=relevant, 0=not relevant
            })

    df = pd.DataFrame(records)
    log.info(f"Total query-passage pairs created: {len(df)}")
    print(f"✅ Created {len(df)} query-passage pairs")
    print(f"   Relevant pairs: {df['label'].sum()}")
    print(f"   Non-relevant pairs: {len(df) - df['label'].sum()}")

    return df


def split_and_save(df, num_train=10000, num_eval=1000):
    """
    Splits into train/eval sets and saves as CSV files.
    """
    # Get unique query IDs
    unique_queries = df["query_id"].unique()
    print(f"\n📊 Total unique queries: {len(unique_queries)}")

    # Split queries into train and eval
    train_queries = unique_queries[:num_train]
    eval_queries = unique_queries[num_train:num_train + num_eval]

    train_df = df[df["query_id"].isin(train_queries)].reset_index(drop=True)
    eval_df = df[df["query_id"].isin(eval_queries)].reset_index(drop=True)

    # Save to disk
    train_path = PROCESSED_DIR / "train.csv"
    eval_path = PROCESSED_DIR / "eval.csv"

    train_df.to_csv(train_path, index=False)
    eval_df.to_csv(eval_path, index=False)

    print(f"\n💾 Saved files:") 
    print(f"   Train: {train_path} ({len(train_df)} pairs, {len(train_queries)} queries)")
    print(f"   Eval:  {eval_path} ({len(eval_df)} pairs, {len(eval_queries)} queries)")
    log.info(f"Train: {len(train_df)} pairs | Eval: {len(eval_df)} pairs")

    return train_df, eval_df


def load_processed_data():
    """
    Loads already processed data from disk.
    Use this after first download to avoid re-downloading.
    """
    train_path = PROCESSED_DIR / "train.csv"
    eval_path = PROCESSED_DIR / "eval.csv"

    if not train_path.exists() or not eval_path.exists():
        raise FileNotFoundError(
            "Processed data not found. Run download_and_prepare() first."
        )

    train_df = pd.read_csv(train_path)
    eval_df = pd.read_csv(eval_path)

    print(f"✅ Loaded train: {len(train_df)} pairs")
    print(f"✅ Loaded eval:  {len(eval_df)} pairs")

    return train_df, eval_df


def download_and_prepare():
    """
    Main function — downloads and prepares everything in one shot.
    Run this once to set up all data.
    """
    print("=" * 50)
    print("  NeuralRank — Data Pipeline")
    print("=" * 50)

    # Check if already processed
    if (PROCESSED_DIR / "train.csv").exists():
        print("\n⚡ Processed data already exists! Loading from disk...")
        return load_processed_data()

    # Download
    dataset = download_msmarco(
        num_train=config["data"]["num_train_queries"],
        num_eval=config["data"]["num_eval_queries"]
    )

    # Process
    df = process_dataset(
        dataset,
        num_train=config["data"]["num_train_queries"],
        num_eval=config["data"]["num_eval_queries"]
    )

    # Split and save
    train_df, eval_df = split_and_save(
        df,
        num_train=config["data"]["num_train_queries"],
        num_eval=config["data"]["num_eval_queries"]
    )

    print("\n✅ Data pipeline complete!")
    print("✅ Ready for Module 1.3 — BM25 Retrieval")

    return train_df, eval_df


if __name__ == "__main__":
    train_df, eval_df = download_and_prepare()

    # Show sample
    print("\n--- Sample record ---")
    sample = train_df.iloc[0]
    print(f"Query:   {sample['query']}")
    print(f"Passage: {sample['passage'][:150]}...")
    print(f"Label:   {sample['label']}")