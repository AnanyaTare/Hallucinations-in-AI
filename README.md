# U2UInnovate-Project 22: Hallucination Detection in Large Language Models

This project focuses on the detection, evaluation, and mitigation of hallucinations in Large Language Models (LLMs) over a 6-week duration.

---

## 📌 Project Overview
- **Description:** Detect and reduce AI hallucinations to ensure safer, more factual, and reliable LLM outputs.
- **Technology Stack:** Large Language Models (LLMs), Natural Language Processing (NLP), Evaluation Metrics
- **Mentor:** Dr. Ritu Jain
- **Duration:** 6 Weeks

---

## 🎯 Objectives
- **Analyze LLM Outputs:** Identify patterns, types, and frequencies of hallucinations across different model generations.
- **Develop Evaluation Framework:** Build automated and LLM-assisted evaluation pipelines and metrics to quantify hallucination rates.
- **Compare Mitigation Techniques:** Implement and benchmark various techniques (e.g., prompt engineering, Retrieval-Augmented Generation (RAG), self-consistency checks, decoding interventions) to minimize hallucination.

---

## 📁 Repository Structure
```text
Hallucinations-in-AI/
├── .env.example                        # Environment variable template (copy to .env)
├── .gitignore
├── requirements.txt                    # Python dependencies
├── setup.py
├── README.md
│
├── data/                               # Datasets and benchmark files
│   ├── README.md
│   └── raw/
│       ├── llm_hallucination_benchmark/
│       ├── rag_hallucination_reduction/
│       └── other_benchmarks/
│
├── src/                                # Core source code
│   ├── analysis/
│   │   ├── hallucination_classifier.py
│   │   └── output_profiler.py
│   ├── evaluation/
│   │   ├── llm_judge.py
│   │   ├── metrics.py
│   │   └── pipeline.py
│   ├── mitigation/
│   │   ├── prompt_engineering.py
│   │   ├── rag_pipeline.py
│   │   └── self_consistency.py
│   └── utils/
│       ├── api_client.py
│       ├── data_loader.py
│       └── logger.py
│
├── deployment/                         # API backend and frontend app
│   ├── api/
│   │   ├── main.py                     # FastAPI app entry point
│   │   ├── config.py                   # Pydantic settings (reads .env)
│   │   ├── database.py                 # SQLAlchemy models & DB setup
│   │   ├── models/
│   │   │   └── schemas.py              # Pydantic request/response schemas
│   │   ├── routers/
│   │   │   ├── chat.py                 # POST /api/chat
│   │   │   ├── evaluate.py             # POST /api/evaluate
│   │   │   ├── analyze.py              # POST /api/analyze
│   │   │   ├── mitigate.py             # POST /api/mitigate
│   │   │   ├── history.py              # GET  /api/history
│   │   │   ├── users.py                # CRUD /api/users
│   │   │   ├── feedback.py             # POST /api/feedback
│   │   │   ├── results.py              # GET  /api/results
│   │   │   └── api_health.py           # GET  /api/health
│   │   └── services/
│   │       ├── llm_service.py          # LiteLLM unified LLM caller
│   │       ├── evaluation_service.py   # Hallucination scoring
│   │       └── mitigation_service.py   # RAG context retrieval
│   ├── app/
│   │   └── streamlit_app.py            # Streamlit frontend UI
│   ├── docker/
│   │   ├── Dockerfile
│   │   └── docker-compose.yml
│   └── tests/
│       └── test_api.py
│
└── reports/                            # Analysis reports and results
    └── README.md
```

For more details on each directory, please refer to the respective `README.md` files inside them.
