import re
from typing import List, Dict, Any, Tuple
from omnirag.core.schemas import Chunk, AuditReport, AuditedClaim, LLMUsage
from omnirag.llm.base import BaseLLMClient
from prompts.prompt_manager import prompt_catalog
from config.settings import settings
from omnirag.core.logger import logger


class HallucinationAuditorAgent:
    """Audits generated answers against retrieved contexts to compute faithfulness scores."""

    def __init__(self, llm_client: BaseLLMClient):
        self.llm_client = llm_client
        self.faithfulness_threshold = settings.agent.synthesizer.faithfulness_threshold

    def audit(self, answer_text: str, evidence_chunks: List[Chunk]) -> Tuple[AuditReport, LLMUsage]:
        """Audits answer against source contexts."""
        logger.info("Performing faithfulness and hallucination audit on synthesized answer...")

        # Format source context text
        source_texts = "\n\n".join([f"[{c.chunk_id}]: {c.text}" for c in evidence_chunks])

        sys_p, _ = prompt_catalog.render("hallucination_auditor", {})
        user_p = prompt_catalog.get_template("hallucination_auditor").render_user(
            generated_answer=answer_text,
            source_contexts=source_texts[:4000],
        )

        try:
            data, usage = self.llm_client.generate_json(system_prompt=sys_p, user_prompt=user_p)

            raw_claims = data.get("audited_claims", [])
            claims_list: List[AuditedClaim] = []

            for rc in raw_claims:
                claims_list.append(
                    AuditedClaim(
                        claim=rc.get("claim", ""),
                        status=rc.get("status", "ENTAILED"),
                        citation=rc.get("citation"),
                        source_evidence_snippet=rc.get("source_evidence_snippet"),
                        audit_note=rc.get("audit_note", ""),
                    )
                )

            total = data.get("total_claims", len(claims_list))
            entailed = data.get("entailed_claims_count", sum(1 for c in claims_list if c.status == "ENTAILED"))
            unverified = data.get("unverified_claims_count", sum(1 for c in claims_list if c.status == "UNVERIFIED"))
            contradicted = data.get("contradicted_claims_count", sum(1 for c in claims_list if c.status == "CONTRADICTED"))

            faithfulness = float(data.get("faithfulness_score", (entailed / total) if total > 0 else 1.0))
            faithfulness = max(0.0, min(1.0, round(faithfulness, 3)))

            if faithfulness >= 0.90:
                risk = "LOW"
            elif faithfulness >= 0.70:
                risk = "MODERATE"
            else:
                risk = "HIGH"

            report = AuditReport(
                total_claims=total,
                entailed_claims_count=entailed,
                unverified_claims_count=unverified,
                contradicted_claims_count=contradicted,
                faithfulness_score=faithfulness,
                hallucination_risk=risk,
                audited_claims=claims_list,
                overall_summary=data.get("overall_summary", f"Faithfulness score: {faithfulness*100:.1f}%"),
            )
            return report, usage

        except Exception as e:
            logger.warning(f"Audit generation failed: {e}. Generating fallback report.")
            # Fallback heuristic calculation
            sentences = [s.strip() for s in re.split(r"[.!?]\s+", answer_text) if len(s.strip()) > 20]
            cited = [s for s in sentences if re.search(r"\[Doc:[^\]]+\]", s)]
            score = round(len(cited) / max(1, len(sentences)), 2)

            return AuditReport(
                total_claims=len(sentences),
                entailed_claims_count=len(cited),
                unverified_claims_count=len(sentences) - len(cited),
                faithfulness_score=score,
                hallucination_risk="LOW" if score >= 0.75 else "MODERATE",
                overall_summary="Heuristic citation-based audit completed.",
            ), LLMUsage()
