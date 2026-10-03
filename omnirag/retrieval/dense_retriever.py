from typing import List, Tuple, Optional
import numpy as np
from omnirag.core.schemas import Chunk
from omnirag.llm.base import BaseLLMClient
from config.settings import settings


class DenseVectorRetriever:
    """Semantic dense vector retriever using Cosine similarity."""

    def __init__(self, llm_client: BaseLLMClient, metric: str = "cosine"):
        self.llm_client = llm_client
        self.metric = metric or settings.retrieval.dense_vector.metric
        self.corpus: List[Chunk] = []
        self.embeddings: Optional[np.ndarray] = None

    def index(self, chunks: List[Chunk]) -> None:
        """Indexes chunks by generating and caching vector embeddings."""
        self.corpus = chunks
        if not chunks:
            self.embeddings = None
            return

        texts = [c.text for c in chunks]
        # Batch generate embeddings
        raw_embs = self.llm_client.get_embeddings(texts)
        emb_arr = np.array(raw_embs, dtype=np.float32)

        # L2-normalize vectors for fast cosine distance via dot product
        norms = np.linalg.norm(emb_arr, axis=1, keepdims=True)
        norms[norms == 0.0] = 1e-10
        self.embeddings = emb_arr / norms

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[Chunk, float]]:
        """Finds top_k nearest chunks to query vector."""
        if not self.corpus or self.embeddings is None:
            return []

        q_embs = self.llm_client.get_embeddings([query])
        if not q_embs:
            return []

        q_vec = np.array(q_embs[0], dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm > 0:
            q_vec = q_vec / q_norm

        # Cosine similarity matrix multiplication
        sim_scores = np.dot(self.embeddings, q_vec)

        ranked_indices = np.argsort(-sim_scores)[:top_k]

        results: List[Tuple[Chunk, float]] = []
        for idx in ranked_indices:
            score = float(sim_scores[idx])
            chunk_copy = self.corpus[idx].model_copy(update={"dense_score": round(score, 4)})
            results.append((chunk_copy, score))

        return results
