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
from omnirag.storage import OmniRAGDatabase
from config.settings import settings, DOCS_DIR


class OmniRAGEngine:
    """Master Multi-Hop Agentic RAG Engine with Self-Reflection, Graph Augmentation,
    and Persistent Database Storage (SQLite).
    """

    def __init__(self, llm_client: Optional[BaseLLMClient] = None, db: Optional[OmniRAGDatabase] = None):
        self.llm_client = llm_client or get_llm_client()
        self.db = db or OmniRAGDatabase()
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

    def index(self, docs_dir: Optional[Path] = None, force_reindex: bool = False) -> int:
        """Incrementally indexes documents into persistent SQLite database.
        Eliminates document limits and caches embeddings/knowledge graphs across restarts.
        """
        target_dir = docs_dir or DOCS_DIR
        logger.info(f"Synchronizing corpus with database from: {target_dir}")

        new_or_modified_docs = 0

        # Scan directory for new or updated files
        if target_dir.exists():
            for file_path in target_dir.glob("*.*"):
                if file_path.suffix.lower() in [".txt", ".md", ".json"]:
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            content = f.read()

                        doc_id = file_path.stem
                        content_hash = self.db.compute_hash(content)

                        if not force_reindex and self.db.is_document_current(doc_id, content_hash):
                            continue  # Up-to-date in database

                        new_or_modified_docs += 1
                        logger.info(f"Indexing new/updated document into database: {file_path.name}")
                        doc_chunks = self.chunker.split_text(content, doc_id=doc_id, metadata={"source_path": str(file_path)})

                        # Generate dense embeddings for chunks
                        chunk_texts = [c.text for c in doc_chunks]
                        embeddings = self.llm_client.get_embeddings(chunk_texts)

                        # Save document and chunks + vectors to SQLite
                        self.db.save_document_and_chunks(
                            doc_id=doc_id,
                            filename=file_path.name,
                            content_hash=content_hash,
                            chunks=doc_chunks,
                            embeddings=embeddings,
                        )

                        # Extract Knowledge Graph and persist
                        self.graph_retriever.extract_from_chunks(doc_chunks, db_handler=self.db)

                    except Exception as e:
                        logger.warning(f"Failed to process {file_path}: {e}")

        # Load all chunks and pre-computed embeddings from database
        self.chunks, cached_embeddings = self.db.load_all_chunks()

        if not self.chunks:
            logger.warning(f"No document chunks found in database or directory {target_dir}")
            return 0

        # Load BM25 index
        self.sparse_retriever.index(self.chunks)

        # Load Dense index with cached vector embeddings (0ms latency!)
        if cached_embeddings is not None and len(cached_embeddings) == len(self.chunks):
            self.dense_retriever.load_cached_embeddings(self.chunks, cached_embeddings)
        else:
            self.dense_retriever.index(self.chunks)

        # Load full Knowledge Graph from database
        kg_entities, kg_relations = self.db.load_knowledge_graph()
        self.graph_retriever.load_from_db(kg_entities, kg_relations)

        self._is_indexed = True
        stats = self.db.get_stats()
        logger.info(
            f"OmniRAG Database Ready: {stats['total_documents']} documents, {stats['total_chunks']} chunks, "
            f"{stats['total_graph_entities']} entities, {stats['total_graph_relations']} graph edges indexed."
        )
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
                    if c.rrf_score > verified_chunks_map[c.chunk_id].rrf_score:
                        verified_chunks_map[c.chunk_id] = c

        all_evidence = list(verified_chunks_map.values())
        if not all_evidence:
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
