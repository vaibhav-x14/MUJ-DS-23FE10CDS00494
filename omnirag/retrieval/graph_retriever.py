import re
from typing import List, Dict, Any, Set, Tuple
import networkx as nx
from omnirag.core.schemas import Chunk, EntityNode, RelationEdge, KnowledgeGraphSnapshot
from omnirag.llm.base import BaseLLMClient
from prompts.prompt_manager import prompt_catalog
from omnirag.core.logger import logger


class KnowledgeGraphRetriever:
    """Extracts, indexes, and traverses an entity-relation knowledge graph using NetworkX."""

    def __init__(self, llm_client: BaseLLMClient):
        self.llm_client = llm_client
        self.graph = nx.MultiDiGraph()
        self.entities: Dict[str, EntityNode] = {}
        self.relations: List[RelationEdge] = []
        self.chunk_entity_map: Dict[str, Set[str]] = {}

    def extract_from_chunks(self, chunks: List[Chunk], max_chunks_to_sample: int = 6) -> None:
        """Extracts entities and relational triplets from corpus chunks using LLM."""
        logger.info(f"Extracting knowledge graph entities from {len(chunks)} chunks...")
        sys_p, _ = prompt_catalog.render("entity_graph_extractor", {"text": ""})

        # Process chunks to build graph
        for chunk in chunks[:max_chunks_to_sample]:
            user_p = prompt_catalog.get_template("entity_graph_extractor").render_user(text=chunk.text)
            try:
                data, _ = self.llm_client.generate_json(system_prompt=sys_p, user_prompt=user_p)
                raw_entities = data.get("entities", [])
                raw_relations = data.get("relationships", [])

                c_entities = set()

                for e in raw_entities:
                    name = e.get("name")
                    if not name:
                        continue
                    node = EntityNode(
                        id=name.lower().replace(" ", "_"),
                        name=name,
                        type=e.get("type", "CONCEPT"),
                        aliases=e.get("aliases", []),
                    )
                    self.entities[node.name.lower()] = node
                    self.graph.add_node(node.name, type=node.type, id=node.id)
                    c_entities.add(node.name)

                for r in raw_relations:
                    subj = r.get("subject")
                    pred = r.get("predicate", "RELATED_TO")
                    obj = r.get("object")
                    snip = r.get("evidence_snippet", "")
                    if subj and obj:
                        edge = RelationEdge(
                            source=subj,
                            target=obj,
                            predicate=pred,
                            evidence_snippet=snip,
                        )
                        self.relations.append(edge)
                        self.graph.add_edge(subj, obj, predicate=pred, evidence=snip)

                self.chunk_entity_map[chunk.chunk_id] = c_entities

            except Exception as e:
                logger.warning(f"Entity graph extraction failed for chunk {chunk.chunk_id}: {e}")

    def query_subgraph(self, query_entities: List[str], max_hops: int = 2) -> KnowledgeGraphSnapshot:
        """Finds sub-graph containing the query entities and their immediate neighborhood."""
        matched_nodes: Set[str] = set()
        matched_edges: List[RelationEdge] = []

        # Find nodes matching query entities
        target_nodes = set()
        for q_ent in query_entities:
            q_clean = q_ent.lower()
            for node_name in self.graph.nodes:
                if q_clean in node_name.lower() or node_name.lower() in q_clean:
                    target_nodes.add(node_name)

        if not target_nodes:
            # Fallback: pick top 4 nodes
            target_nodes = set(list(self.graph.nodes)[:4])

        matched_nodes.update(target_nodes)

        # 1-hop / 2-hop neighborhood expansion
        for node in target_nodes:
            # Outgoing edges
            if self.graph.has_node(node):
                for _, neighbor, data in self.graph.out_edges(node, data=True):
                    matched_nodes.add(neighbor)
                    matched_edges.append(
                        RelationEdge(
                            source=node,
                            target=neighbor,
                            predicate=data.get("predicate", "RELATED_TO"),
                            evidence_snippet=data.get("evidence", ""),
                        )
                    )
                # Incoming edges
                for neighbor, _, data in self.graph.in_edges(node, data=True):
                    matched_nodes.add(neighbor)
                    matched_edges.append(
                        RelationEdge(
                            source=neighbor,
                            target=node,
                            predicate=data.get("predicate", "RELATED_TO"),
                            evidence_snippet=data.get("evidence", ""),
                        )
                    )

        nodes_list = []
        for n in matched_nodes:
            node_data = self.graph.nodes.get(n, {})
            nodes_list.append(
                EntityNode(
                    id=n.lower().replace(" ", "_"),
                    name=n,
                    type=node_data.get("type", "CONCEPT"),
                )
            )

        return KnowledgeGraphSnapshot(nodes=nodes_list, edges=matched_edges)

    def format_for_synthesis(self, snapshot: KnowledgeGraphSnapshot) -> str:
        """Formats graph triples as concise text lines for context injection."""
        if not snapshot.edges:
            return "No explicit knowledge graph triples found."
        lines = []
        for e in snapshot.edges[:8]:
            lines.append(f"- ({e.source}) --[{e.predicate}]--> ({e.target})")
        return "\n".join(lines)
