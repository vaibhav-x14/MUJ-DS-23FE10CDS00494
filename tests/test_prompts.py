import pytest
from prompts.prompt_manager import prompt_catalog


def test_prompt_catalog_discovery():
    templates = prompt_catalog.list_templates()
    expected_prompts = [
        "query_decomposition",
        "document_grader",
        "entity_graph_extractor",
        "grounded_synthesis",
        "hallucination_auditor",
    ]
    for p in expected_prompts:
        assert p in templates or f"{p}_planner" in templates or f"{p}_engine" in templates


def test_query_decomposition_render():
    sys_p, user_p = prompt_catalog.render(
        "query_decomposition",
        {"user_query": "Compare QuantumScape to Tesla batteries"}
    )
    assert len(sys_p) > 20
    assert "Compare QuantumScape to Tesla batteries" in user_p


def test_grounded_synthesis_render():
    sys_p, user_p = prompt_catalog.render(
        "grounded_synthesis",
        {
            "user_query": "Test Query",
            "reasoning_trace": "Hop 1 -> Hop 2",
            "evidence_contexts": "[Doc:1:Chunk:1] Some context",
            "graph_context": "- (A) --[REL]--> (B)",
        }
    )
    assert "[Doc:1:Chunk:1]" in user_p
    assert "Test Query" in user_p
