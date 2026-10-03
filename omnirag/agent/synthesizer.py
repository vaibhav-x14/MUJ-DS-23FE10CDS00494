from typing import List, Dict, Any, Tuple
from omnirag.core.schemas import Chunk, QueryPlan, KnowledgeGraphSnapshot, LLMUsage
from omnirag.llm.base import BaseLLMClient
from prompts.prompt_manager import prompt_catalog
from omnirag.core.logger import logger


class GroundedSynthesizerAgent:
    """Synthesizes multi-hop evidence and graph relationships into an authoritative grounded answer."""

    def __init__(self, llm_client: BaseLLMClient):
        self.llm_client = llm_client

    def synthesize(
        self,
        user_query: str,
        plan: QueryPlan,
        evidence_chunks: List[Chunk],
        graph_snapshot: KnowledgeGraphSnapshot,
    ) -> Tuple[str, LLMUsage]:
        """Synthesizes final answer with strict citation references."""
        logger.info(f"Synthesizing grounded answer for query: '{user_query}' using {len(evidence_chunks)} contexts")

        # Format reasoning trace
        trace_lines = [f"Plan Strategy: {plan.reasoning}"]
        for h in plan.hops:
            trace_lines.append(f"- Hop {h.hop_id}: {h.sub_query} (Target: {h.target_entity or 'General'})")
        reasoning_trace = "\n".join(trace_lines)

        # Format evidence contexts with exact citation tags
        context_blocks = []
        for c in evidence_chunks:
            tag = f"[Doc:{c.doc_id}:Chunk:{c.chunk_id.split('_')[-1]}]"
            context_blocks.append(f"{tag}\n{c.text}\n")
        evidence_contexts = "\n---\n".join(context_blocks)

        # Format graph triples
        graph_lines = []
        for e in graph_snapshot.edges[:6]:
            graph_lines.append(f"- ({e.source}) --[{e.predicate}]--> ({e.target})")
        graph_context = "\n".join(graph_lines) if graph_lines else "No explicit knowledge graph triples found."

        sys_p, _ = prompt_catalog.render("grounded_synthesis", {})
        user_p = prompt_catalog.get_template("grounded_synthesis").render_user(
            user_query=user_query,
            reasoning_trace=reasoning_trace,
            evidence_contexts=evidence_contexts,
            graph_context=graph_context,
        )

        answer_text, usage = self.llm_client.generate(system_prompt=sys_p, user_prompt=user_p)
        return answer_text, usage
