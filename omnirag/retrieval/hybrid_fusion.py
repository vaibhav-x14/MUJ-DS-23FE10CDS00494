from typing import List, Tuple, Dict
from omnirag.core.schemas import Chunk
from config.settings import settings


class ReciprocalRankFusion:
    """Combines BM25 sparse and dense vector rankings using Reciprocal Rank Fusion (RRF).
    Formula: RRF_score(d) = sum_{m in models} w_m / (k + rank_m(d))
    """

    def __init__(self, k: int = 60, sparse_weight: float = 0.5, dense_weight: float = 0.5):
        cfg = settings.retrieval.hybrid_fusion
        self.k = k or cfg.rrf_k
        self.sparse_weight = sparse_weight or cfg.sparse_weight
        self.dense_weight = dense_weight or cfg.dense_weight

    def fuse(
        self,
        sparse_results: List[Tuple[Chunk, float]],
        dense_results: List[Tuple[Chunk, float]],
        top_k: int = 5,
    ) -> List[Chunk]:
        """Fuses ranked lists from sparse and dense retrievers into a single unified list."""
        rrf_scores: Dict[str, float] = {}
        chunk_map: Dict[str, Chunk] = {}

        # Process Sparse (BM25) ranking
        for rank, (chunk, score) in enumerate(sparse_results):
            cid = chunk.chunk_id
            if cid not in chunk_map:
                chunk_map[cid] = chunk.model_copy()
            chunk_map[cid].bm25_score = score
            contribution = self.sparse_weight / (self.k + (rank + 1))
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + contribution

        # Process Dense ranking
        for rank, (chunk, score) in enumerate(dense_results):
            cid = chunk.chunk_id
            if cid not in chunk_map:
                chunk_map[cid] = chunk.model_copy()
            chunk_map[cid].dense_score = score
            contribution = self.dense_weight / (self.k + (rank + 1))
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + contribution

        # Sort by total RRF score
        sorted_cids = sorted(rrf_scores.keys(), key=lambda c: rrf_scores[c], reverse=True)

        final_chunks: List[Chunk] = []
        for cid in sorted_cids[:top_k]:
            c = chunk_map[cid]
            c.rrf_score = round(rrf_scores[cid], 5)
            final_chunks.append(c)

        return final_chunks
