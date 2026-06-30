"""
data_loader.py — Dataset loading utilities.
Loads raw datasets from data/raw/ and processed JSONL files from data/processed/.
"""

import json
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict, Any
from loguru import logger

# ── Base paths ────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parents[2]
RAW_DIR  = BASE_DIR / "data" / "raw"
PROC_DIR = BASE_DIR / "data" / "processed"
EVAL_DIR = BASE_DIR / "data" / "evaluation"


# ── JSONL Loader ──────────────────────────────────────────────────────────────
def load_jsonl(filepath: str | Path) -> List[Dict[str, Any]]:
    """Load a JSONL file and return a list of dicts."""
    filepath = Path(filepath)
    if not filepath.exists():
        logger.warning(f"JSONL file not found: {filepath}")
        return []

    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    logger.info(f"Loaded {len(records)} records from {filepath.name}")
    return records


def save_jsonl(records: List[Dict], filepath: str | Path) -> None:
    """Save a list of dicts as a JSONL file."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info(f"Saved {len(records)} records to {filepath.name}")


# ── CSV Loader ────────────────────────────────────────────────────────────────
def load_csv(filepath: str | Path, **kwargs) -> pd.DataFrame:
    """Load a CSV file into a pandas DataFrame."""
    filepath = Path(filepath)
    if not filepath.exists():
        logger.error(f"CSV not found: {filepath}")
        return pd.DataFrame()

    df = pd.read_csv(filepath, **kwargs)
    logger.info(f"Loaded CSV: {filepath.name} | shape={df.shape}")
    return df


# ── HuggingFace Dataset Loader ────────────────────────────────────────────────
def load_hf_benchmark(dataset_name: str, config: Optional[str] = None, split: str = "validation") -> List[Dict]:
    """
    Load a HuggingFace benchmark dataset.

    Common benchmarks:
    - "truthful_qa", config="generation"
    - "coqa"
    - "halueval" (custom)

    Args:
        dataset_name: HuggingFace dataset identifier.
        config: Dataset configuration name (optional).
        split: Which split to load ("train", "validation", "test").

    Returns:
        List of dicts, one per sample.
    """
    try:
        from datasets import load_dataset
        logger.info(f"Loading HF dataset: {dataset_name} | config={config} | split={split}")
        ds = load_dataset(dataset_name, config, split=split)
        records = [dict(row) for row in ds]
        logger.success(f"Loaded {len(records)} samples from {dataset_name}")
        return records
    except Exception as e:
        logger.error(f"Failed to load HF dataset '{dataset_name}': {e}")
        return []


# ── Evaluation Output Loader ──────────────────────────────────────────────────
def load_evaluation_outputs(filename: str = "model_outputs.jsonl") -> List[Dict]:
    """Load previously saved model outputs from data/evaluation/."""
    return load_jsonl(EVAL_DIR / filename)


def save_evaluation_outputs(records: List[Dict], filename: str = "model_outputs.jsonl") -> None:
    """Save model outputs to data/evaluation/."""
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    save_jsonl(records, EVAL_DIR / filename)
