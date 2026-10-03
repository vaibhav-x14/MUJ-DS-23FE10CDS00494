import re
from typing import List, Dict, Any, Set, Tuple, Optional
import networkx as nx
from omnirag.core.schemas import Chunk, EntityNode, RelationEdge, KnowledgeGraphSnapshot
from omnirag.llm.base import BaseLLMClient
from prompts.prompt_manager import prompt_catalog
from omnirag.core.logger import logger


class KnowledgeGraphRetriever:
    """Extracts, indexes, persists, and traverses an entity-relation knowledge graph using NetworkX & SQLite."""

    def __init__(self, llm_client: BaseLLMClient):
        self.llm_client = llm_client
        self.graph = nx.MultiDiGraph()
        self.entities: Dict[str, EntityNode] = {}
        self.relations: List[RelationEdge] = []
        self.chunk_entity_map: Dict[str, Set[str]] = {}

    def load_from_db(self, db_entities: List[EntityNode], db_relations: List[RelationEdge]) -> None:
        """Loads and rebuilds NetworkX graph from database records."""
        self.graph.clear()
        self.entities.clear()
        self.relations.clear()

        for node in db_entities:
            self.entities[node.name.lower()] = node
            self.graph.add_node(node.name, type=node.type, id=node.id)

        for edge in db_relations:
            self.relations.append(edge)
            self.graph.add_edge(edge.source, edge.target, predicate=edge.predicate, evidence=edge.evidence_snippet)

        logger.info(f"Loaded Knowledge Graph from database ({len(db_entities)} entities, {len(db_relations)} relationships).")

    def extract_from_chunks(
        self,
        chunks: List[Chunk],
        max_chunks: Optional[int] = None,
        db_handler: Optional[Any] = None,
    ) -> None:
        """Extracts entities and relational triplets from corpus chunks, persisting to database."""
        # If DB already has graph data, load first
        if db_handler:
            saved_entities, saved_relations = db_handler.load_knowledge_graph()
            if saved_entities:
                self.load_from_db(saved_entities, saved_relations)
                return

        target_chunks = chunks[:max_chunks] if max_chunks else chunks
        logger.info(f"Extracting knowledge graph entities across {len(target_chunks)} chunks...")
        sys_p, _ = prompt_catalog.render("entity_graph_extractor", {"text": ""})

        new_entities: List[EntityNode] = []
        new_relations: List[RelationEdge] = []

        for chunk in target_chunks:
            user_p = prompt_catalog.get_template("entity_graph_extractor").render_user(text=chunk.text)
            try:
                data, _ = self.llm_client.generate_json(system_prompt=sys_p, user_prompt=user_p)
                raw_entities = data.get("entities", [])
                raw_relations = data.get("relationships", [])

                c_entities = set()

                for e in raw_entities:
                    if isinstance(e, str):
                        name = e
                        etype = "CONCEPT"
                        aliases = []
                    elif isinstance(e, dict):
                        name = e.get("name")
                        etype = e.get("type", "CONCEPT")
                        aliases = e.get("aliases", [])
                    else:
                        continue

                    if not name:
                        continue
                    node = EntityNode(
                        id=name.lower().replace(" ", "_"),
                        name=name,
                        type=etype,
                        aliases=aliases,
                    )
                    self.entities[node.name.lower()] = node
                    self.graph.add_node(node.name, type=node.type, id=node.id)
                    c_entities.add(node.name)
                    new_entities.append(node)

                for r in raw_relations:
                    if not isinstance(r, dict):
                        continue
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
                        new_relations.append(edge)

                self.chunk_entity_map[chunk.chunk_id] = c_entities

            except Exception as e:
                logger.warning(f"Entity graph extraction failed for chunk {chunk.chunk_id}: {e}")

        # Persist extracted graph to database if provided
        if db_handler and (new_entities or new_relations):
            db_handler.save_knowledge_graph(new_entities, new_relations)
            logger.info("Persisted newly extracted knowledge graph to SQLite database.")

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
