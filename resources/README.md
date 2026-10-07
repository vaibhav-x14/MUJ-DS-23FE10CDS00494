# OmniRAG Project Resources & Knowledge Assets

This directory contains external assets, benchmark evaluation scenarios, SQLite database schemas, and architectural reference specifications for the **OmniRAG** project.

---

## 📂 Subdirectories

- **`schema/`**: Production SQLite DDL schemas for documents, chunks, dense vector embedding blobs, and knowledge graph triplets.
- **`datasets/`**: Curated multi-hop benchmark question-answer pairs and technical corpus passages.
- **`architecture/`**: System architecture diagrams, DAG flow charts, and algorithmic definitions.

---

## 🗄️ Database Schema Details
The SQLite engine (`data/omnirag.db`) manages:
- `documents`: Document-level metadata, content hash (SHA-256), and timestamps.
- `document_chunks`: Sliding window tokenized passages (`chunk_index`, `content`).
- `embeddings`: High-dimensional vector blobs (Google Gemini 3072-dim or 768-dim mock).
- `graph_entities`: Canonical knowledge graph entity nodes.
- `graph_relations`: Directed predicate edges connecting entity nodes.
