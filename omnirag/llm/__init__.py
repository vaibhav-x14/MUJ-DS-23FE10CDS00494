import os
from typing import Optional
from config.settings import settings
from omnirag.llm.base import BaseLLMClient
from omnirag.llm.gemini_client import GeminiLLMClient
from omnirag.llm.mock_client import MockLLMClient
from omnirag.core.logger import logger


def get_llm_client(force_mock: Optional[bool] = None) -> BaseLLMClient:
    """Returns configured LLM client with automatic fallback."""
    api_key = os.getenv("GEMINI_API_KEY") or settings.llm.api_key
    use_mock = force_mock if force_mock is not None else (settings.llm.provider == "mock" or not api_key)

    if not use_mock and api_key:
        try:
            client = GeminiLLMClient(api_key=api_key)
            if client.is_available():
                return client
        except Exception as e:
            logger.warning(f"Failed to initialize Gemini client: {e}. Falling back to MockLLMClient.")

    return MockLLMClient()


__all__ = ["BaseLLMClient", "GeminiLLMClient", "MockLLMClient", "get_llm_client"]
