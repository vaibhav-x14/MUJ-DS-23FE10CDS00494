import tempfile
from pathlib import Path
import numpy as np
import pytest
from omnirag.storage.database import OmniRAGDatabase
from omnirag.core.schemas import Chunk, EntityNode, RelationEdge


def test_database_crud():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_omnirag.db"
        db = OmniRAGDatabase(db_path=db_path)

        # 1. Test document and chunk persistence
        chunks = [
            Chunk(chunk_id="doc1_c1", doc_id="doc1", text="Quantum computing using neutral atoms."),
            Chunk(chunk_id="doc1_c2", doc_id="doc1", text="Optical tweezers trap rubidium atoms."),
        ]
        embs = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        content_hash = db.compute_hash("Full text of document")

        db.save_document_and_chunks("doc1", "doc1.txt", content_hash, chunks, embs)
        assert db.is_document_current("doc1", content_hash) is True
        assert db.is_document_current("doc1", "different_hash") is False

        # 2. Test chunk and embedding loading
        loaded_chunks, loaded_embs = db.load_all_chunks()
        assert len(loaded_chunks) == 2
        assert loaded_chunks[0].chunk_id == "doc1_c1"
        assert loaded_embs is not None
        assert loaded_embs.shape == (2, 3)

        # 3. Test knowledge graph persistence
        entities = [
            EntityNode(id="e1", name="QuEra", type="ORGANIZATION", aliases=["QuEra Computing"]),
            EntityNode(id="e2", name="Neutral Atom", type="TECHNOLOGY"),
        ]
        relations = [
            RelationEdge(source="QuEra", target="Neutral Atom", predicate="DEVELOPS"),
        ]
        db.save_knowledge_graph(entities, relations)

        loaded_entities, loaded_relations = db.load_knowledge_graph()
        assert len(loaded_entities) == 2
        assert len(loaded_relations) == 1
        assert loaded_relations[0].predicate == "DEVELOPS"

        stats = db.get_stats()
        assert stats["total_documents"] == 1
        assert stats["total_chunks"] == 2
        assert stats["total_graph_entities"] == 2
