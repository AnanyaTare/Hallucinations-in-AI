# Data Directory

This folder is dedicated to storing datasets, prompts, generated outputs, and evaluation samples used for hallucination detection.

## 📂  Folder Structure
```text
data/
├── raw/           # Original datasets (e.g., TruthfulQA, CoQA, or custom prompt datasets)
├── processed/     # Cleaned data, tokenized samples, or structured inputs
└── evaluation/    # Model outputs, annotated generations, and ground-truth comparison sets
```

## 📊 Expected Data Formats
- **JSON/JSONL:** For prompt-response pairs, model outputs, and evaluation scores.
- **CSV:** For metadata, statistics, and tabular benchmark results.

> [!NOTE]
> Ensure all large datasets are ignored via `.gitignore` if they exceed Git's file size limit, and keep only representative samples in the repository.
