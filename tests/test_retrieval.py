import pytest
from omnirag.core.schemas import Chunk
from omnirag.retrieval import (
    SemanticRecursiveChunker,
    BM25Retriever,
    DenseVectorRetriever,
    ReciprocalRankFusion,
    KnowledgeGraphRetriever,
)
from omnirag.llm import MockLLMClient


@pytest.fixture
def sample_chunks():
    return [
        Chunk(
            chunk_id="doc1_chunk_1",
            doc_id="doc1",
            text="Neutral-atom quantum architectures utilize optical tweezers and Rydberg states with long coherence times.",
        ),
        Chunk(
            chunk_id="doc2_chunk_1",
            doc_id="doc2",
            text="TSMC and Intel Foundry manufacture leading edge semiconductors using High-NA EUV lithography scanners.",
        ),
        Chunk(
            chunk_id="doc3_chunk_1",
            doc_id="doc3",
            text="Solid-state lithium batteries use ceramic LLZO electrolytes achieving 415 Wh/kg energy density.",
        ),
    ]


def test_bm25_retrieval_ranking(sample_chunks):
    retriever = BM25Retriever()
    retriever.index(sample_chunks)

    hits = retriever.retrieve("optical tweezers neutral atom", top_k=2)
    assert len(hits) > 0
    top_chunk, score = hits[0]
    assert top_chunk.chunk_id == "doc1_chunk_1"
    assert score > 0.0


def test_dense_retrieval_ranking(sample_chunks):
    client = MockLLMClient()
    retriever = DenseVectorRetriever(client)
    retriever.index(sample_chunks)

    hits = retriever.retrieve("semiconductor foundry EUV lithography", top_k=2)
    assert len(hits) > 0
    top_chunk, score = hits[0]
    assert top_chunk.chunk_id == "doc2_chunk_1"
    assert score > 0.0


def test_reciprocal_rank_fusion(sample_chunks):
    fuser = ReciprocalRankFusion(k=60)
    sparse_res = [(sample_chunks[0], 5.0), (sample_chunks[1], 2.0)]
    dense_res = [(sample_chunks[0], 0.9), (sample_chunks[2], 0.4)]

    fused = fuser.fuse(sparse_res, dense_res, top_k=3)
    assert len(fused) == 3
    # sample_chunks[0] is rank 1 in both, so it must be top
    assert fused[0].chunk_id == "doc1_chunk_1"
    assert fused[0].rrf_score > fused[1].rrf_score


def test_chunker_splitting():
    chunker = SemanticRecursiveChunker(chunk_size=50, chunk_overlap=10)
    text = ("Paragraph one discusses quantum processors.\n\n"
            "Paragraph two discusses superconducting transmon qubits and dilution refrigerators.\n\n"
            "Paragraph three discusses cryogenic cooling systems.")
    chunks = chunker.split_text(text, doc_id="test_doc")
    assert len(chunks) >= 1
    assert all(c.doc_id == "test_doc" for c in chunks)
