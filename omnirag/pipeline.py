import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from omnirag.core.schemas import (
    Chunk,
    QueryPlan,
    KnowledgeGraphSnapshot,
    OmniRAGResult,
    LLMUsage,
)
from omnirag.core.logger import logger
from omnirag.llm import get_llm_client, BaseLLMClient
from omnirag.retrieval import (
    SemanticRecursiveChunker,
    BM25Retriever,
    DenseVectorRetriever,
    ReciprocalRankFusion,
    KnowledgeGraphRetriever,
)
from omnirag.agent import (
    QueryPlannerAgent,
    DocumentGraderAgent,
    GroundedSynthesizerAgent,
    HallucinationAuditorAgent,
)
from config.settings import settings, DOCS_DIR


class OmniRAGEngine:
    """Master Multi-Hop Agentic RAG Engine with Self-Reflection and Graph Augmentation."""

    def __init__(self, llm_client: Optional[BaseLLMClient] = None):
        self.llm_client = llm_client or get_llm_client()
        self.chunker = SemanticRecursiveChunker()
        self.sparse_retriever = BM25Retriever()
        self.dense_retriever = DenseVectorRetriever(self.llm_client)
        self.rrf_fuser = ReciprocalRankFusion()
        self.graph_retriever = KnowledgeGraphRetriever(self.llm_client)

        # Agentic modules
        self.planner = QueryPlannerAgent(self.llm_client)
        self.grader = DocumentGraderAgent(self.llm_client)
        self.synthesizer = GroundedSynthesizerAgent(self.llm_client)
        self.auditor = HallucinationAuditorAgent(self.llm_client)

        self.chunks: List[Chunk] = []
        self._is_indexed = False

    def index(self, docs_dir: Optional[Path] = None) -> int:
        """Loads corpus, creates chunks, and indexes sparse, dense, and graph representations."""
        target_dir = docs_dir or DOCS_DIR
        logger.info(f"Indexing documents from directory: {target_dir}")
        self.chunks = self.chunker.chunk_directory(target_dir)

        if not self.chunks:
            logger.warning(f"No document chunks found in {target_dir}")
            return 0

        logger.info(f"Chunked corpus into {len(self.chunks)} semantic passages.")

        # 1. Index BM25
        self.sparse_retriever.index(self.chunks)

        # 2. Index Dense vectors
        self.dense_retriever.index(self.chunks)

        # 3. Extract and index Knowledge Graph
        self.graph_retriever.extract_from_chunks(self.chunks)

        self._is_indexed = True
        logger.info("OmniRAG Indexing complete (BM25 + Dense + KnowledgeGraph ready).")
        return len(self.chunks)

    def query(self, user_query: str, verbose: bool = True) -> OmniRAGResult:
        """Executes full multi-hop agentic retrieval, synthesis, and hallucination audit."""
        start_time = time.perf_counter()
        if not self._is_indexed:
            self.index()

        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_latency_ms = 0.0

        def accumulate_usage(u: Optional[LLMUsage]):
            nonlocal total_prompt_tokens, total_completion_tokens, total_latency_ms
            if u:
                total_prompt_tokens += u.prompt_tokens
                total_completion_tokens += u.completion_tokens
                total_latency_ms += u.latency_ms

        # -------------------------------------------------------------
        # STEP 1: Multi-Hop Query Decomposition Planning
        # -------------------------------------------------------------
        plan = self.planner.plan(user_query)

        # -------------------------------------------------------------
        # STEP 2: Multi-Hop Hybrid Retrieval & Self-RAG Relevance Grading
        # -------------------------------------------------------------
        verified_chunks_map: Dict[str, Chunk] = {}
        query_entities: List[str] = []

        for hop in plan.hops:
            if hop.target_entity:
                query_entities.append(hop.target_entity)

            # Retrieve candidates via BM25
            sparse_hits = self.sparse_retriever.retrieve(
                hop.sub_query, top_k=settings.retrieval.hybrid_fusion.top_k_candidates
            )

            # Retrieve candidates via Dense Vector
            dense_hits = self.dense_retriever.retrieve(
                hop.sub_query, top_k=settings.retrieval.hybrid_fusion.top_k_candidates
            )

            # Reciprocal Rank Fusion
            fused_candidates = self.rrf_fuser.fuse(
                sparse_hits, dense_hits, top_k=settings.retrieval.hybrid_fusion.final_top_k
            )

            # Self-RAG relevance grading & noise filtering (Corrective RAG)
            graded_chunks = self.grader.filter_and_grade(hop.sub_query, fused_candidates)

            hop.retrieved_chunk_ids = [c.chunk_id for c in graded_chunks]
            hop.status = "EXECUTED"

            for c in graded_chunks:
                if c.chunk_id not in verified_chunks_map:
                    verified_chunks_map[c.chunk_id] = c
                else:
                    # Update scores if higher
                    if c.rrf_score > verified_chunks_map[c.chunk_id].rrf_score:
                        verified_chunks_map[c.chunk_id] = c

        all_evidence = list(verified_chunks_map.values())
        if not all_evidence:
            # Fallback to top-1 overall chunk if strict filters removed everything
            all_evidence = self.chunks[:1]

        # -------------------------------------------------------------
        # STEP 3: Knowledge Graph Expansion & Relational Traversal
        # -------------------------------------------------------------
        graph_snapshot = self.graph_retriever.query_subgraph(query_entities)

        # -------------------------------------------------------------
        # STEP 4: Grounded Multi-Hop Synthesis with Citations
        # -------------------------------------------------------------
        answer_text, synth_usage = self.synthesizer.synthesize(
            user_query=user_query,
            plan=plan,
            evidence_chunks=all_evidence,
            graph_snapshot=graph_snapshot,
        )
        accumulate_usage(synth_usage)

        # -------------------------------------------------------------
        # STEP 5: Self-Reflective Hallucination & Faithfulness Audit
        # -------------------------------------------------------------
        audit_report, audit_usage = self.auditor.audit(
            answer_text=answer_text,
            evidence_chunks=all_evidence,
        )
        accumulate_usage(audit_usage)

        total_exec_seconds = round(time.perf_counter() - start_time, 3)

        combined_usage = LLMUsage(
            prompt_tokens=total_prompt_tokens,
            completion_tokens=total_completion_tokens,
            total_tokens=total_prompt_tokens + total_completion_tokens,
            estimated_cost_usd=(total_prompt_tokens * 0.0000001) + (total_completion_tokens * 0.0000004),
            latency_ms=round(total_latency_ms, 2),
            model=getattr(self.llm_client, "model_name", "omnirag-llm"),
        )

        return OmniRAGResult(
            query=user_query,
            query_plan=plan,
            retrieved_chunks=all_evidence,
            knowledge_graph=graph_snapshot,
            synthesized_answer=answer_text,
            audit_report=audit_report,
            usage=combined_usage,
            execution_time_seconds=total_exec_seconds,
        )
