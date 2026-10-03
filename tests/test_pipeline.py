import pytest
from omnirag.pipeline import OmniRAGEngine
from omnirag.core.schemas import OmniRAGResult
from omnirag.llm.mock_client import MockLLMClient


@pytest.fixture(scope="module")
def engine():
    e = OmniRAGEngine(llm_client=MockLLMClient())
    e.index()
    return e


def test_pipeline_query_execution(engine):
    result = engine.query("How does neutral atom coherence time compare to superconducting qubits?")
    assert isinstance(result, OmniRAGResult)
    assert len(result.query_plan.hops) >= 1
    assert len(result.retrieved_chunks) >= 1
    assert len(result.synthesized_answer) > 50
    assert result.audit_report.faithfulness_score >= 0.7
    assert result.audit_report.hallucination_risk in ("LOW", "MODERATE")
    assert result.execution_time_seconds > 0.0


def test_pipeline_multihop_reasoning(engine):
    query = "Compare the energy density of QuantumScape's solid-state battery cells against conventional lithium-ion batteries"
    result = engine.query(query)
    assert len(result.query_plan.hops) >= 2
    # Verify citations present in synthesized answer
    assert "[Doc:" in result.synthesized_answer
