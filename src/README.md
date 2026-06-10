# Source Code Directory

This directory contains the core implementation of the evaluation frameworks, analysis scripts, and mitigation techniques.

## 📂  Folder Structure
```text
src/
├── analysis/       # Scripts to inspect and profile LLM output hallucinations
├── evaluation/     # Metrics implementation (e.g., semantic similarity, NLI-based consistency, GPT-assisted evaluation)
├── mitigation/     # Implementations of mitigation techniques (RAG, self-correction, decoding-level adjustments)
└── utils/          # Helper modules for API connections, text preprocessing, and logging
```

## 🛠️ Key Components
1. **Output Analyzer:** Code to run queries against models and classify outputs.
2. **Evaluation Pipeline:** Scoring scripts to benchmark hallucination rates.
3. **Mitigation Sandbox:** Scripts comparing performance before and after applying mitigation techniques.
