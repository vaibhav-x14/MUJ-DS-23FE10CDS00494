# OmniRAG Source Code & CLI Entry Points

This directory contains the production source code and executable entry points for the **OmniRAG** Multi-Hop Agentic RAG system.

---

## 📁 Directory Structure

```
code/
├── README.md               # Source code documentation and usage guide
├── run_pipeline.py         # CLI runner for end-to-end multi-hop RAG execution
├── benchmark.py            # Automated quantitative evaluation benchmark runner
├── serve_ui.py             # Interactive FastAPI web dashboard launcher
└── omnirag/                # (Symlinked / Core Engine Modules)
    ├── agent/              # Multi-hop DAG query planner & self-reflection
    ├── retrieval/          # Hybrid BM25Okapi + Dense vector + Light Graph-RAG
    ├── storage/            # SQLite persistent database & vector blob store
    ├── synthesis/          # Grounded citation synthesizer
    ├── auditor/            # Calibrated NLI hallucination auditor
    ├── llm/                # Gemini 3.5 Flash / 1.5 Pro client with fallback
    └── pipeline.py         # Unified end-to-end pipeline orchestrator
```

---

## 🚀 Execution Commands

### 1. Run Multi-Hop Query Pipeline (CLI)
```bash
python code/run_pipeline.py --query "How does the coherence time of neutral-atom quantum processors compare to superconducting qubits?"
```

### 2. Launch Interactive Web Dashboard
```bash
python code/serve_ui.py
# Open http://localhost:8000 in your browser
```

### 3. Run Quantitative Benchmark Suite
```bash
python code/benchmark.py
```

### 4. Run Automated Unit & Integration Tests
```bash
pytest tests/ -v
```
