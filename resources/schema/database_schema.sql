-- OmniRAG SQLite Schema Specification
-- Persistent Relational, Dense Vector Blob & Knowledge Graph Storage

-- 1. Documents Table
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    source TEXT,
    content TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 2. Document Chunks Table
CREATE TABLE IF NOT EXISTS document_chunks (
    id TEXT PRIMARY KEY,
    doc_id TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    token_count INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(doc_id) REFERENCES documents(id) ON DELETE CASCADE
);

-- 3. Embeddings Table
CREATE TABLE IF NOT EXISTS embeddings (
    chunk_id TEXT PRIMARY KEY,
    embedding_blob BLOB NOT NULL,
    dim INTEGER NOT NULL,
    model_name TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(chunk_id) REFERENCES document_chunks(id) ON DELETE CASCADE
);

-- 4. Knowledge Graph Entities Table
CREATE TABLE IF NOT EXISTS graph_entities (
    entity_name TEXT PRIMARY KEY,
    entity_type TEXT DEFAULT 'CONCEPT',
    doc_ids TEXT NOT NULL, -- JSON array of doc IDs
    frequency INTEGER DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 5. Knowledge Graph Relations Table
CREATE TABLE IF NOT EXISTS graph_relations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_entity TEXT NOT NULL,
    target_entity TEXT NOT NULL,
    relation_type TEXT NOT NULL,
    doc_id TEXT NOT NULL,
    weight REAL DEFAULT 1.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(source_entity) REFERENCES graph_entities(entity_name) ON DELETE CASCADE,
    FOREIGN KEY(target_entity) REFERENCES graph_entities(entity_name) ON DELETE CASCADE
);

-- Indexes for high-performance retrieval
CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON document_chunks(doc_id);
CREATE INDEX IF NOT EXISTS idx_graph_relations_source ON graph_relations(source_entity);
CREATE INDEX IF NOT EXISTS idx_graph_relations_target ON graph_relations(target_entity);
