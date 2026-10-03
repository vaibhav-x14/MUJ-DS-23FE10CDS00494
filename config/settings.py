import os
from pathlib import Path
from typing import List, Optional
import yaml
from pydantic import BaseModel, Field

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "config" / "config.yaml"
DATA_DIR = BASE_DIR / "data"
DOCS_DIR = DATA_DIR / "documents"
PROMPTS_DIR = BASE_DIR / "prompts"


class SystemConfig(BaseModel):
    project_name: str = "OmniRAG"
    version: str = "1.0.0"
    environment: str = "production"
    log_level: str = "INFO"
    seed: int = 42


class LLMConfig(BaseModel):
    provider: str = "gemini"
    model_name: str = "gemini-2.0-flash"
    embedding_model: str = "text-embedding-004"
    temperature: float = 0.2
    top_p: float = 0.95
    max_output_tokens: int = 2048
    request_timeout_seconds: int = 30
    max_retries: int = 3
    retry_backoff_factor: float = 1.5
    fallback_to_mock: bool = True
    api_key: Optional[str] = None


class ChunkingConfig(BaseModel):
    strategy: str = "recursive_semantic"
    chunk_size: int = 400
    chunk_overlap: int = 80
    min_chunk_length: int = 50
    separators: List[str] = ["\n\n", "\n", ". ", "; ", " "]


class SparseBM25Config(BaseModel):
    enabled: bool = True
    k1: float = 1.5
    b: float = 0.75
    lowercase: bool = True
    remove_stopwords: bool = True


class DenseVectorConfig(BaseModel):
    enabled: bool = True
    metric: str = "cosine"
    dimension: int = 768


class HybridFusionConfig(BaseModel):
    method: str = "reciprocal_rank_fusion"
    rrf_k: int = 60
    sparse_weight: float = 0.5
    dense_weight: float = 0.5
    top_k_candidates: int = 8
    final_top_k: int = 4


class RetrievalConfig(BaseModel):
    chunking: ChunkingConfig = ChunkingConfig()
    sparse_bm25: SparseBM25Config = SparseBM25Config()
    dense_vector: DenseVectorConfig = DenseVectorConfig()
    hybrid_fusion: HybridFusionConfig = HybridFusionConfig()


class KnowledgeGraphConfig(BaseModel):
    enabled: bool = True
    max_hop_depth: int = 2
    entity_types: List[str] = ["ORGANIZATION", "TECHNOLOGY", "METRIC", "CONCEPT", "MATERIAL", "DATE"]
    min_cooccurrence_weight: int = 1


class PlannerConfig(BaseModel):
    max_hops: int = 3
    allow_parallel_hops: bool = True


class SelfReflectionConfig(BaseModel):
    enabled: bool = True
    relevance_threshold: float = 0.65
    min_acceptable_sources: int = 2
    enable_query_rewriting: bool = True


class SynthesizerConfig(BaseModel):
    require_citations: bool = True
    citation_format: str = "[Doc:{doc_id}:Chunk:{chunk_id}]"
    hallucination_audit: bool = True
    faithfulness_threshold: float = 0.75


class AgentConfig(BaseModel):
    planner: PlannerConfig = PlannerConfig()
    self_reflection: SelfReflectionConfig = SelfReflectionConfig()
    synthesizer: SynthesizerConfig = SynthesizerConfig()


class WebConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: List[str] = ["*"]
    enable_ui: bool = True


class AppConfig(BaseModel):
    system: SystemConfig = SystemConfig()
    llm: LLMConfig = LLMConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    knowledge_graph: KnowledgeGraphConfig = KnowledgeGraphConfig()
    agent: AgentConfig = AgentConfig()
    web: WebConfig = WebConfig()


def load_config(config_path: Optional[Path] = None) -> AppConfig:
    """Load YAML config with environment variable overrides."""
    target_path = config_path or CONFIG_PATH
    config_dict = {}

    if target_path.exists():
        with open(target_path, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f) or {}

    config = AppConfig(**config_dict)

    # Check environment variable overrides
    env_api_key = os.getenv("GEMINI_API_KEY")
    if env_api_key:
        config.llm.api_key = env_api_key

    if os.getenv("OMNIRAG_MODEL"):
        config.llm.model_name = os.getenv("OMNIRAG_MODEL")

    if os.getenv("OMNIRAG_USE_MOCK") and os.getenv("OMNIRAG_USE_MOCK").lower() in ("true", "1", "yes"):
        config.llm.fallback_to_mock = True
        config.llm.provider = "mock"

    return config


# Global singleton instance
settings = load_config()
