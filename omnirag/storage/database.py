import json
import sqlite3
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from omnirag.core.schemas import Chunk, EntityNode, RelationEdge
from omnirag.core.logger import logger
from config.settings import settings, DATA_DIR

DEFAULT_DB_PATH = DATA_DIR / "omnirag.db"


class OmniRAGDatabase:
    """Production persistent storage layer for documents, chunks, dense embeddings,
    and knowledge graph entities/relations using SQLite.
    Removes in-memory limitations and enables scaling to unlimited documents.
    """

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA journal_mode = WAL;")  # High concurrency write-ahead logging
        return conn

    def _init_db(self):
        """Initializes database schema with indexing."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            # Documents Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    doc_id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    total_chunks INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            # Chunks Table with Vector Embedding BLOB
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    doc_id TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    metadata_json TEXT,
                    embedding_blob BLOB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
                );
            """)

            # Knowledge Graph Entities Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS graph_entities (
                    id TEXT PRIMARY KEY,
                    name TEXT UNIQUE NOT NULL,
                    entity_type TEXT NOT NULL,
                    aliases_json TEXT,
                    frequency INTEGER DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            # Knowledge Graph Relations Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS graph_relations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source TEXT NOT NULL,
                    target TEXT NOT NULL,
                    predicate TEXT NOT NULL,
                    evidence_snippet TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            # Performance Indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_rel_source ON graph_relations(source);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_rel_target ON graph_relations(target);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_rel_pair ON graph_relations(source, target);")
            conn.commit()

    def compute_hash(self, text: str) -> str:
        """Returns SHA-256 hash of document text for incremental change detection."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def is_document_current(self, doc_id: str, content_hash: str) -> bool:
        """Checks if document is already indexed and unchanged."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT content_hash FROM documents WHERE doc_id = ?", (doc_id,))
            row = cursor.fetchone()
            return row is not None and row["content_hash"] == content_hash

    def save_document_and_chunks(
        self,
        doc_id: str,
        filename: str,
        content_hash: str,
        chunks: List[Chunk],
        embeddings: Optional[List[List[float]]] = None,
    ) -> None:
        """Persists document and all its chunks (including binary vector embeddings) transactionally."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            # Upsert document record
            cursor.execute("""
                INSERT INTO documents (doc_id, filename, content_hash, total_chunks)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(doc_id) DO UPDATE SET
                    filename = excluded.filename,
                    content_hash = excluded.content_hash,
                    total_chunks = excluded.total_chunks,
                    created_at = CURRENT_TIMESTAMP;
            """, (doc_id, filename, content_hash, len(chunks)))

            # Delete old chunks if updating
            cursor.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))

            # Insert chunks with embedding blobs
            for idx, c in enumerate(chunks):
                emb_blob = None
                if embeddings and idx < len(embeddings):
                    emb_arr = np.array(embeddings[idx], dtype=np.float32)
                    emb_blob = emb_arr.tobytes()

                cursor.execute("""
                    INSERT INTO chunks (chunk_id, doc_id, chunk_index, text, metadata_json, embedding_blob)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    c.chunk_id,
                    doc_id,
                    idx,
                    c.text,
                    json.dumps(c.metadata),
                    emb_blob,
                ))
            conn.commit()

    def load_all_chunks(self) -> Tuple[List[Chunk], Optional[np.ndarray]]:
        """Loads all persisted chunks and their embedding vectors in a single high-speed query."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT chunk_id, doc_id, text, metadata_json, embedding_blob
                FROM chunks
                ORDER BY doc_id, chunk_index
            """)
            rows = cursor.fetchall()

            chunks: List[Chunk] = []
            emb_list: List[np.ndarray] = []
            has_embs = True

            for r in rows:
                meta = json.loads(r["metadata_json"]) if r["metadata_json"] else {}
                chunk = Chunk(
                    chunk_id=r["chunk_id"],
                    doc_id=r["doc_id"],
                    text=r["text"],
                    metadata=meta,
                )
                chunks.append(chunk)

                if r["embedding_blob"]:
                    arr = np.frombuffer(r["embedding_blob"], dtype=np.float32)
                    emb_list.append(arr)
                else:
                    has_embs = False

            embeddings = np.array(emb_list, dtype=np.float32) if (has_embs and emb_list) else None
            return chunks, embeddings

    def save_knowledge_graph(self, entities: List[EntityNode], relations: List[RelationEdge]) -> None:
        """Persists extracted entities and relationships into database."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            for e in entities:
                cursor.execute("""
                    INSERT INTO graph_entities (id, name, entity_type, aliases_json, frequency)
                    VALUES (?, ?, ?, ?, 1)
                    ON CONFLICT(name) DO UPDATE SET
                        frequency = frequency + 1,
                        aliases_json = excluded.aliases_json;
                """, (e.id, e.name, e.type, json.dumps(e.aliases)))

            for r in relations:
                # Check for existing relation to avoid duplicate edges
                cursor.execute("""
                    SELECT id FROM graph_relations
                    WHERE source = ? AND target = ? AND predicate = ?
                """, (r.source, r.target, r.predicate))
                if not cursor.fetchone():
                    cursor.execute("""
                        INSERT INTO graph_relations (source, target, predicate, evidence_snippet)
                        VALUES (?, ?, ?, ?)
                    """, (r.source, r.target, r.predicate, r.evidence_snippet))
            conn.commit()

    def load_knowledge_graph(self) -> Tuple[List[EntityNode], List[RelationEdge]]:
        """Loads the entire persisted knowledge graph from database."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, name, entity_type, aliases_json FROM graph_entities")
            e_rows = cursor.fetchall()

            entities = [
                EntityNode(
                    id=r["id"],
                    name=r["name"],
                    type=r["entity_type"],
                    aliases=json.loads(r["aliases_json"]) if r["aliases_json"] else [],
                )
                for r in e_rows
            ]

            cursor.execute("SELECT source, target, predicate, evidence_snippet FROM graph_relations")
            r_rows = cursor.fetchall()
            relations = [
                RelationEdge(
                    source=r["source"],
                    target=r["target"],
                    predicate=r["predicate"],
                    evidence_snippet=r["evidence_snippet"] or "",
                )
                for r in r_rows
            ]

            return entities, relations

    def get_stats(self) -> Dict[str, Any]:
        """Returns database storage counts."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT count(*) as c FROM documents")
            n_docs = cursor.fetchone()["c"]
            cursor.execute("SELECT count(*) as c FROM chunks")
            n_chunks = cursor.fetchone()["c"]
            cursor.execute("SELECT count(*) as c FROM graph_entities")
            n_entities = cursor.fetchone()["c"]
            cursor.execute("SELECT count(*) as c FROM graph_relations")
            n_relations = cursor.fetchone()["c"]

            return {
                "database_path": str(self.db_path),
                "total_documents": n_docs,
                "total_chunks": n_chunks,
                "total_graph_entities": n_entities,
                "total_graph_relations": n_relations,
            }
