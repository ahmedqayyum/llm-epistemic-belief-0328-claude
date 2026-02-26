"""
Data preparation script for the epistemic belief differentiation study.

Loads the agentlans/fact-or-opinion dataset, filters to English,
and creates a balanced evaluation set for experiments.
"""

import json
import random
import os
import sys

import numpy as np
from datasets import load_from_disk

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

BASE_DIR = "/workspaces/llm-epistemic-belief-0328-claude"
DATASET_DIR = os.path.join(BASE_DIR, "datasets", "fact_or_opinion")
OUTPUT_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_fact_opinion_data():
    """Load the fact-or-opinion dataset from local arrow files."""
    # Try loading with HuggingFace datasets from disk
    try:
        ds = load_from_disk(DATASET_DIR)
        print(f"Loaded dataset with splits: {list(ds.keys())}")
        return ds
    except Exception as e:
        print(f"Could not load with load_from_disk: {e}")

    # Fallback: load arrow files directly
    import pyarrow as pa
    import pyarrow.ipc as ipc
    import pandas as pd

    data = {}
    for split in ["train", "test"]:
        arrow_path = os.path.join(DATASET_DIR, split, "data-00000-of-00001.arrow")
        if os.path.exists(arrow_path):
            reader = ipc.open_file(arrow_path)
            table = reader.read_all()
            df = table.to_pandas()
            data[split] = df
            print(f"Loaded {split}: {len(df)} rows")
        else:
            print(f"Arrow file not found: {arrow_path}")
    return data


def filter_english(data):
    """Filter dataset to English-only entries."""
    filtered = {}
    for split, df in data.items():
        if not isinstance(df, pd.DataFrame):
            df = df.to_pandas() if hasattr(df, 'to_pandas') else pd.DataFrame(df)

        en_df = df[df['language'] == 'en'].copy()
        filtered[split] = en_df
        print(f"{split} (English only): {len(en_df)} rows")
    return filtered


def create_evaluation_set(data, n_per_class=100):
    """
    Create a balanced evaluation set from the test split.
    Returns n_per_class Fact + n_per_class Opinion statements.
    Also samples some Both/Neither for exploratory analysis.
    """
    test_df = data.get('test', data.get('train'))
    if hasattr(test_df, 'to_pandas'):
        test_df = test_df.to_pandas()

    print(f"\nLabel distribution in test set:")
    print(test_df['label'].value_counts())

    # Sample balanced Fact and Opinion sets
    facts = test_df[test_df['label'] == 'Fact']
    opinions = test_df[test_df['label'] == 'Opinion']

    n_fact = min(n_per_class, len(facts))
    n_opinion = min(n_per_class, len(opinions))

    fact_sample = facts.sample(n=n_fact, random_state=SEED)
    opinion_sample = opinions.sample(n=n_opinion, random_state=SEED)

    # Primary evaluation set (binary: Fact vs Opinion)
    eval_set = pd.concat([fact_sample, opinion_sample]).sample(frac=1, random_state=SEED).reset_index(drop=True)
    print(f"\nPrimary evaluation set: {len(eval_set)} statements")
    print(f"  Facts: {len(fact_sample)}, Opinions: {len(opinion_sample)}")

    # Exploratory set (Both and Neither)
    both = test_df[test_df['label'] == 'Both']
    neither = test_df[test_df['label'] == 'Neither']
    n_exploratory = min(25, len(both), len(neither))

    exploratory_set = None
    if n_exploratory > 0:
        both_sample = both.sample(n=n_exploratory, random_state=SEED)
        neither_sample = neither.sample(n=n_exploratory, random_state=SEED)
        exploratory_set = pd.concat([both_sample, neither_sample]).sample(frac=1, random_state=SEED).reset_index(drop=True)
        print(f"Exploratory set: {len(exploratory_set)} statements (Both: {n_exploratory}, Neither: {n_exploratory})")

    return eval_set, exploratory_set


def create_training_set(data):
    """Create training set for TF-IDF baseline (English, Fact/Opinion only)."""
    train_df = data.get('train')
    if hasattr(train_df, 'to_pandas'):
        train_df = train_df.to_pandas()

    # Filter to Fact and Opinion only for binary classification
    binary_train = train_df[train_df['label'].isin(['Fact', 'Opinion'])].copy()
    print(f"\nTraining set (Fact/Opinion only): {len(binary_train)} statements")
    print(binary_train['label'].value_counts())

    return binary_train


def save_data(eval_set, exploratory_set, train_set):
    """Save prepared datasets as JSON."""
    # Save evaluation set
    eval_records = eval_set[['text', 'label']].to_dict('records')
    eval_path = os.path.join(OUTPUT_DIR, "eval_set.json")
    with open(eval_path, 'w') as f:
        json.dump(eval_records, f, indent=2)
    print(f"\nSaved evaluation set to {eval_path}")

    # Save exploratory set
    if exploratory_set is not None:
        expl_records = exploratory_set[['text', 'label']].to_dict('records')
        expl_path = os.path.join(OUTPUT_DIR, "exploratory_set.json")
        with open(expl_path, 'w') as f:
            json.dump(expl_records, f, indent=2)
        print(f"Saved exploratory set to {expl_path}")

    # Save training set
    train_records = train_set[['text', 'label']].to_dict('records')
    train_path = os.path.join(OUTPUT_DIR, "train_set.json")
    with open(train_path, 'w') as f:
        json.dump(train_records, f, indent=2)
    print(f"Saved training set to {train_path}")

    # Print sample statements
    print("\n--- Sample Evaluation Statements ---")
    for i, row in eval_set.head(5).iterrows():
        print(f"  [{row['label']:>7s}] {row['text'][:80]}")


import pandas as pd

if __name__ == "__main__":

    print("=" * 60)
    print("Data Preparation for Epistemic Belief Study")
    print("=" * 60)

    # Load data
    print("\n1. Loading dataset...")
    raw_data = load_fact_opinion_data()

    # Handle different return types
    if isinstance(raw_data, dict) and all(isinstance(v, pd.DataFrame) for v in raw_data.values()):
        data = raw_data
    else:
        # HuggingFace DatasetDict
        data = {}
        for split in raw_data:
            df = raw_data[split].to_pandas()
            data[split] = df

    # Filter to English
    print("\n2. Filtering to English...")
    en_data = filter_english(data)

    # Create training set
    print("\n3. Creating training set...")
    train_set = create_training_set(en_data)

    # Create evaluation set
    print("\n4. Creating evaluation set...")
    eval_set, exploratory_set = create_evaluation_set(en_data, n_per_class=100)

    # Save
    print("\n5. Saving data...")
    save_data(eval_set, exploratory_set, train_set)

    print("\n✓ Data preparation complete!")
