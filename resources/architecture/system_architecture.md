# OmniRAG System Architecture Specification

## 1. End-to-End Pipeline Diagram

```mermaid
flowchart TD
    subgraph Client [User Interaction Layer]
        User([User Query]) --> WebUI[FastAPI + Glassmorphism UI]
        User --> CLI[CLI Runner / Script]
    end

    subgraph Planner [Agentic Reasoning Layer]
        WebUI --> DAGPlanner[Multi-Hop DAG Query Planner]
        CLI --> DAGPlanner
        DAGPlanner --> Hop1[Sub-Query 1]
        DAGPlanner --> Hop2[Sub-Query 2]
    end

    subgraph Retrieval [Hybrid Retrieval Layer]
        Hop1 --> BM25_1[BM25Okapi Lexical]
        Hop1 --> Dense_1[Gemini Dense Embedding Cosine]
        Hop2 --> BM25_2[BM25Okapi Lexical]
        Hop2 --> Dense_2[Gemini Dense Embedding Cosine]
        BM25_1 & Dense_1 --> RRF_1[Reciprocal Rank Fusion k=60]
        BM25_2 & Dense_2 --> RRF_2[Reciprocal Rank Fusion k=60]
    end

    subgraph Verification [Reflective Noise Filtering & Knowledge Graph]
        RRF_1 & RRF_2 --> CRAG[Self-RAG Document Grader]
        CRAG --> FilteredPassages[Graded & Relevant Chunks]
        FilteredPassages --> GraphTraverse[Light Graph-RAG Neighborhood Expansion]
    end

    subgraph Storage [Persistent Database Layer]
        SQLite[(SQLite Engine: omnirag.db)]
        SQLite <--> BM25_1 & BM25_2
        SQLite <--> Dense_1 & Dense_2
        SQLite <--> GraphTraverse
    end

    subgraph Generation [Synthesis & Audit Layer]
        GraphTraverse --> Synthesizer[LLM Citation Synthesizer]
        Synthesizer --> DraftAnswer[Answer with Citations]
        DraftAnswer --> Auditor[Calibrated NLI Hallucination Auditor]
        Auditor --> FinalOutput[Audited Answer + Telemetry + Graph Visual]
    end
```

## 2. Key Mathematical Formulations

- **Reciprocal Rank Fusion (RRF):**
  $$RRF(d) = \sum_{m \in \{\text{BM25}, \text{Dense}\}} \frac{1}{60 + \text{rank}_m(d)}$$

- **Cosine Similarity:**
  $$\text{sim}(\mathbf{q}, \mathbf{d}) = \frac{\mathbf{q} \cdot \mathbf{d}}{\|\mathbf{q}\|_2 \|\mathbf{d}\|_2}$$

- **Faithfulness Score:**
  $$\text{Faithfulness} = \frac{\sum_{i=1}^{N} \mathbb{I}(\text{verdict}_i = \text{ENTAILED})}{N}$$
