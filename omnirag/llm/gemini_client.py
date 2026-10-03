import json
import os
import re
import time
from typing import List, Dict, Any, Optional
from omnirag.llm.base import BaseLLMClient
from omnirag.core.schemas import LLMUsage
from omnirag.core.logger import logger
from config.settings import settings


class GeminiLLMClient(BaseLLMClient):
    """Production client for Google Gemini 1.5/2.0 API with retry, telemetry, and fallback."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        embedding_model: Optional[str] = None,
    ):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or settings.llm.api_key
        self.model_name = model_name or settings.llm.model_name
        self.embedding_model = embedding_model or settings.llm.embedding_model
        self.client = None

        if self.api_key:
            try:
                from google import genai
                self.client = genai.Client(api_key=self.api_key)
                logger.info(f"Initialized Google Gemini Client (Model: {self.model_name})")
            except Exception as e:
                logger.warning(f"Could not initialize google-genai client: {e}")
        else:
            logger.info("No GEMINI_API_KEY detected. Running in Offline Mock mode by default.")

    def is_available(self) -> bool:
        return self.client is not None and bool(self.api_key)

    def _estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Estimate cost in USD based on Gemini 2.0 Flash / 1.5 pricing."""
        # Approx: $0.10 per 1M prompt tokens, $0.40 per 1M completion tokens
        return (prompt_tokens * 0.0000001) + (completion_tokens * 0.0000004)

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> tuple[str, LLMUsage]:
        if not self.is_available():
            raise RuntimeError("Gemini client not initialized with valid API key")

        from google.genai import types

        temp = temperature if temperature is not None else settings.llm.temperature
        max_tok = max_tokens or settings.llm.max_output_tokens
        max_retries = settings.llm.max_retries
        backoff = settings.llm.retry_backoff_factor

        last_err = None
        start_time = time.perf_counter()

        for attempt in range(max_retries):
            try:
                config = types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=temp,
                    max_output_tokens=max_tok,
                )
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=user_prompt,
                    config=config,
                )
                latency_ms = (time.perf_counter() - start_time) * 1000

                prompt_tokens = getattr(response.usage_metadata, "prompt_token_count", 0) if hasattr(response, "usage_metadata") else 0
                completion_tokens = getattr(response.usage_metadata, "candidates_token_count", 0) if hasattr(response, "usage_metadata") else 0
                total_tokens = prompt_tokens + completion_tokens

                usage = LLMUsage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    estimated_cost_usd=self._estimate_cost(prompt_tokens, completion_tokens),
                    latency_ms=latency_ms,
                    model=self.model_name,
                )

                text = response.text or ""
                return text.strip(), usage

            except Exception as e:
                last_err = e
                logger.warning(f"Gemini API attempt {attempt+1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(backoff * (2 ** attempt))

        raise RuntimeError(f"Gemini API generation failed after {max_retries} attempts: {last_err}")

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: Optional[Dict[str, Any]] = None,
        temperature: Optional[float] = None,
    ) -> tuple[Dict[str, Any], LLMUsage]:
        if not self.is_available():
            raise RuntimeError("Gemini client not initialized with valid API key")

        from google.genai import types

        temp = temperature if temperature is not None else settings.llm.temperature
        max_retries = settings.llm.max_retries
        backoff = settings.llm.retry_backoff_factor

        last_err = None
        start_time = time.perf_counter()

        for attempt in range(max_retries):
            try:
                config = types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=temp,
                    response_mime_type="application/json",
                )
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=user_prompt,
                    config=config,
                )
                latency_ms = (time.perf_counter() - start_time) * 1000

                prompt_tokens = getattr(response.usage_metadata, "prompt_token_count", 0) if hasattr(response, "usage_metadata") else 0
                completion_tokens = getattr(response.usage_metadata, "candidates_token_count", 0) if hasattr(response, "usage_metadata") else 0
                total_tokens = prompt_tokens + completion_tokens

                usage = LLMUsage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    estimated_cost_usd=self._estimate_cost(prompt_tokens, completion_tokens),
                    latency_ms=latency_ms,
                    model=self.model_name,
                )

                raw_text = response.text or "{}"
                # Clean potential markdown fences
                cleaned = re.sub(r"^```json\s*", "", raw_text.strip(), flags=re.MULTILINE)
                cleaned = re.sub(r"^```\s*$", "", cleaned.strip(), flags=re.MULTILINE)
                data = json.loads(cleaned.strip())
                return data, usage

            except Exception as e:
                last_err = e
                logger.warning(f"Gemini JSON generation attempt {attempt+1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(backoff * (2 ** attempt))

        raise RuntimeError(f"Gemini API JSON generation failed after {max_retries} retries: {last_err}")

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        if not self.is_available() or not texts:
            return []

        try:
            # Batch embedding call via google-genai
            response = self.client.models.embed_content(
                model=self.embedding_model,
                contents=texts,
            )
            embeddings = []
            if hasattr(response, "embeddings"):
                for emb in response.embeddings:
                    embeddings.append(list(emb.values))
            elif hasattr(response, "embedding"):
                embeddings.append(list(response.embedding.values))
            return embeddings
        except Exception as e:
            logger.warning(f"Gemini embedding API failed: {e}. Falling back.")
            raise e
