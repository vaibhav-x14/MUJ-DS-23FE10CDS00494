from typing import Dict, Any, List
from omnirag.core.schemas import QueryPlan, SubQueryHop
from omnirag.llm.base import BaseLLMClient
from prompts.prompt_manager import prompt_catalog
from omnirag.core.logger import logger


class QueryPlannerAgent:
    """Decomposes complex user queries into a Directed Acyclic Graph (DAG) of sub-queries."""

    def __init__(self, llm_client: BaseLLMClient):
        self.llm_client = llm_client

    def plan(self, user_query: str) -> QueryPlan:
        """Generates a structured multi-hop execution plan."""
        logger.info(f"Decomposing query: '{user_query}'")
        sys_p, user_p = prompt_catalog.render("query_decomposition", {"user_query": user_query})

        try:
            data, usage = self.llm_client.generate_json(system_prompt=sys_p, user_prompt=user_p)
            if isinstance(data, list):
                hops_data = data
                reasoning = "Multi-hop decomposition executed."
            elif isinstance(data, dict):
                hops_data = data.get("hops", [])
                reasoning = data.get("reasoning", "Multi-hop decomposition executed.")
            else:
                hops_data = []
                reasoning = "Direct fallback execution."

            hops: List[SubQueryHop] = []

            for h in hops_data:
                if not isinstance(h, dict):
                    continue
                hops.append(
                    SubQueryHop(
                        hop_id=h.get("hop_id", len(hops) + 1),
                        sub_query=h.get("sub_query", user_query),
                        target_entity=h.get("target_entity"),
                        depends_on=h.get("depends_on", []),
                    )
                )

            if not hops:
                hops = [SubQueryHop(hop_id=1, sub_query=user_query, target_entity="Root Query")]

            return QueryPlan(
                original_query=user_query,
                reasoning=reasoning,
                is_multihop=len(hops) > 1,
                hops=hops,
            )

        except Exception as e:
            logger.warning(f"Query planning fallback triggered due to: {e}")
            return QueryPlan(
                original_query=user_query,
                reasoning="Direct single-hop fallback execution.",
                is_multihop=False,
                hops=[SubQueryHop(hop_id=1, sub_query=user_query, target_entity="Direct Query")],
            )
