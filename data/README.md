# Data Directory

This folder is dedicated to storing datasets, prompts, generated outputs, and evaluation samples used for hallucination detection.

## 📂  Folder Structure
```text
data/
├── raw/
│   ├── llm_hallucination_benchmark/  
 Hallucination Benchmark Dataset (CSV/JSON)
│   ├── rag_hallucination_reduction/  # RAG-Based Hallucination Reduction Dataset (CSV/JSON/Text)
│   └── other_benchmarks/             # TruthfulQA, CoQA, or other raw downloads
├── processed/                        # Cleaned data, standardized prompt-response pairs, or structured inputs
└── evaluation/                       # Model outputs, annotated generations, and ground-truth comparison sets
```

## 📊 Expected Data Formats
- **JSON/JSONL:** For prompt-response pairs, model outputs, and evaluation scores.
- **CSV:** For metadata, statistics, and tabular benchmark results.

> [!NOTE]
> Ensure all large datasets are ignored via `.gitignore` if they exceed Git's file size limit, and keep only representative samples in the repository.
