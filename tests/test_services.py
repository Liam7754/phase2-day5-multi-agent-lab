"""Tests for mock services."""

from multi_agent_research_lab.services.llm_client import LLMClient, MockLLMClient
from multi_agent_research_lab.services.search_client import MockSearchClient, SearchClient


def test_mock_llm_returns_content() -> None:
    """MockLLMClient should return non-empty content."""
    mock = MockLLMClient()
    response = mock.complete("You are a researcher.", "Tell me about AI agents.")
    assert response.content
    assert response.input_tokens and response.input_tokens > 0
    assert response.output_tokens and response.output_tokens > 0
    assert response.cost_usd is not None
    assert response.model == "mock-gpt-4o-mini"


def test_mock_llm_tracks_totals() -> None:
    """MockLLMClient should accumulate token counts."""
    mock = MockLLMClient()
    mock.complete("system", "user prompt 1")
    mock.complete("system", "user prompt 2")
    assert mock.call_count == 2
    assert mock.total_input_tokens > 0
    assert mock.total_output_tokens > 0


def test_mock_search_returns_sources() -> None:
    """MockSearchClient should return SourceDocument list."""
    mock = MockSearchClient()
    results = mock.search("multi-agent systems")
    assert len(results) > 0
    assert results[0].title
    assert results[0].snippet


def test_mock_search_respects_max_results() -> None:
    """MockSearchClient should respect max_results parameter."""
    mock = MockSearchClient()
    results = mock.search("multi-agent systems", max_results=2)
    assert len(results) <= 2


def test_llm_client_uses_mock_without_key() -> None:
    """LLMClient should use mock when no API key is configured."""
    client = LLMClient()
    response = client.complete("system", "user prompt")
    assert response.content
    assert response.model == "mock-gpt-4o-mini"


def test_search_client_uses_mock_without_key() -> None:
    """SearchClient should use mock when no Tavily key is configured."""
    client = SearchClient()
    results = client.search("test query")
    assert len(results) > 0
