from typing import List, Dict, Any, Tuple
from omnirag.core.schemas import Chunk, RelevanceGrade
from omnirag.llm.base import BaseLLMClient
from prompts.prompt_manager import prompt_catalog
from config.settings import settings
from omnirag.core.logger import logger


class DocumentGraderAgent:
    """Evaluates retrieved candidate chunks for relevance and filters noise (Corrective RAG)."""

    def __init__(self, llm_client: BaseLLMClient):
        self.llm_client = llm_client
        self.threshold = settings.agent.self_reflection.relevance_threshold

    def grade_chunk(self, sub_query: str, chunk: Chunk) -> RelevanceGrade:
        """Grades a single chunk against the sub-query."""
        sys_p, _ = prompt_catalog.render("document_grader", {})
        user_p = prompt_catalog.get_template("document_grader").render_user(
            sub_query=sub_query,
            chunk_id=chunk.chunk_id,
            chunk_text=chunk.text[:800],
        )

        try:
            data, _ = self.llm_client.generate_json(system_prompt=sys_p, user_prompt=user_p)
            score = float(data.get("relevance_score", 0.7))
            verdict = data.get("verdict", "RELEVANT" if score >= self.threshold else "IRRELEVANT")
            return RelevanceGrade(
                sub_query=sub_query,
                chunk_id=chunk.chunk_id,
                relevance_score=score,
                verdict=verdict,
                key_factual_points=data.get("key_factual_points", []),
                rationale=data.get("rationale", ""),
            )
        except Exception as e:
            logger.warning(f"Grading chunk {chunk.chunk_id} failed: {e}. Defaulting to relevant.")
            return RelevanceGrade(
                sub_query=sub_query,
                chunk_id=chunk.chunk_id,
                relevance_score=0.75,
                verdict="RELEVANT",
                rationale="Fallback default grade.",
            )

    def filter_and_grade(self, sub_query: str, candidates: List[Chunk]) -> List[Chunk]:
        """Filters candidate chunks, attaching relevance scores and discarding irrelevant noise."""
        verified_chunks: List[Chunk] = []

        for chunk in candidates:
            grade = self.grade_chunk(sub_query, chunk)
            chunk.relevance_score = grade.relevance_score
            chunk.relevance_verdict = grade.verdict

            if grade.relevance_score >= self.threshold or grade.verdict in ("RELEVANT", "PARTIALLY_RELEVANT"):
                verified_chunks.append(chunk)

        # Fallback: if all filtered out, retain top 1 candidate to avoid total starvation
        if not verified_chunks and candidates:
            candidates[0].relevance_score = 0.5
            candidates[0].relevance_verdict = "PARTIALLY_RELEVANT"
            verified_chunks.append(candidates[0])

        return verified_chunks
