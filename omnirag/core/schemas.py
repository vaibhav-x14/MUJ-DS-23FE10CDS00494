from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    bm25_score: float = 0.0
    dense_score: float = 0.0
    rrf_score: float = 0.0
    relevance_score: Optional[float] = None
    relevance_verdict: Optional[str] = None


class SubQueryHop(BaseModel):
    hop_id: int
    sub_query: str
    target_entity: Optional[str] = None
    depends_on: List[int] = Field(default_factory=list)
    retrieved_chunk_ids: List[str] = Field(default_factory=list)
    status: str = "PENDING"  # PENDING, EXECUTED, REWRITTEN


class QueryPlan(BaseModel):
    original_query: str
    reasoning: str
    is_multihop: bool = True
    hops: List[SubQueryHop] = Field(default_factory=list)


class RelevanceGrade(BaseModel):
    sub_query: str
    chunk_id: str
    relevance_score: float
    verdict: str  # RELEVANT, PARTIALLY_RELEVANT, IRRELEVANT
    key_factual_points: List[str] = Field(default_factory=list)
    rationale: str = ""


class EntityNode(BaseModel):
    id: str
    name: str
    type: str = "CONCEPT"
    aliases: List[str] = Field(default_factory=list)


class RelationEdge(BaseModel):
    source: str
    target: str
    predicate: str
    evidence_snippet: str = ""


class KnowledgeGraphSnapshot(BaseModel):
    nodes: List[EntityNode] = Field(default_factory=list)
    edges: List[RelationEdge] = Field(default_factory=list)


class AuditedClaim(BaseModel):
    claim: str
    status: str  # ENTAILED, UNVERIFIED, CONTRADICTED
    citation: Optional[str] = None
    source_evidence_snippet: Optional[str] = None
    audit_note: str = ""


class AuditReport(BaseModel):
    total_claims: int = 0
    entailed_claims_count: int = 0
    unverified_claims_count: int = 0
    contradicted_claims_count: int = 0
    faithfulness_score: float = 1.0
    hallucination_risk: str = "LOW"  # LOW, MODERATE, HIGH
    audited_claims: List[AuditedClaim] = Field(default_factory=list)
    overall_summary: str = ""


class LLMUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0
    latency_ms: float = 0.0
    model: str = ""


class OmniRAGResult(BaseModel):
    query: str
    query_plan: QueryPlan
    retrieved_chunks: List[Chunk] = Field(default_factory=list)
    knowledge_graph: KnowledgeGraphSnapshot = Field(default_factory=KnowledgeGraphSnapshot)
    synthesized_answer: str
    audit_report: AuditReport
    usage: LLMUsage
    execution_time_seconds: float = 0.0
