from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from omnirag.core.schemas import LLMUsage


class BaseLLMClient(ABC):
    """Abstract interface for LLM providers."""

    @abstractmethod
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> tuple[str, LLMUsage]:
        """Generate text completion from system and user prompt."""
        pass

    @abstractmethod
    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: Optional[Dict[str, Any]] = None,
        temperature: Optional[float] = None,
    ) -> tuple[Dict[str, Any], LLMUsage]:
        """Generate structured JSON response conforming to schema."""
        pass

    @abstractmethod
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate dense vector embeddings for a list of text strings."""
        pass
