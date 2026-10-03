import numpy as np
import pytest
from omnirag.llm import get_llm_client, MockLLMClient, GeminiLLMClient


def test_mock_llm_generation():
    client = MockLLMClient()
    text, usage = client.generate("system prompt", "Explain quantum computing")
    assert len(text) > 0
    assert usage.total_tokens > 0
    assert usage.model == "mock-gemini-simulator"


def test_mock_llm_json_generation():
    client = MockLLMClient()
    prompt = "USER QUESTION:\nHow does neutral atom compare to superconducting?"
    data, usage = client.generate_json(
        system_prompt="system",
        user_prompt=prompt,
    )
    assert isinstance(data, dict)
    assert "hops" in data
    assert len(data["hops"]) >= 1


def test_mock_llm_embeddings_normalization():
    client = MockLLMClient()
    texts = ["quantum optical tweezers", "semiconductor extreme ultraviolet lithography"]
    embs = client.get_embeddings(texts)

    assert len(embs) == 2
    assert len(embs[0]) == 768
    assert len(embs[1]) == 768

    # Verify L2 normalization
    norm0 = np.linalg.norm(np.array(embs[0]))
    norm1 = np.linalg.norm(np.array(embs[1]))
    assert pytest.approx(norm0, abs=1e-3) == 1.0
    assert pytest.approx(norm1, abs=1e-3) == 1.0


def test_gemini_client_fallback_without_key():
    # If no key, get_llm_client safely falls back to MockLLMClient
    client = get_llm_client(force_mock=True)
    assert isinstance(client, MockLLMClient)
