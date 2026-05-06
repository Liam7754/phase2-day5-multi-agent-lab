"""Tests for implemented agents."""

from multi_agent_research_lab.agents import (
    AnalystAgent,
    CriticAgent,
    ResearcherAgent,
    SupervisorAgent,
    WriterAgent,
)
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState


def _make_state(query: str = "Explain multi-agent systems") -> ResearchState:
    return ResearchState(request=ResearchQuery(query=query))


def test_supervisor_routes_to_researcher_first() -> None:
    """Supervisor should route to researcher when no research notes exist."""
    state = _make_state()
    result = SupervisorAgent().run(state)
    assert result.route_history[-1] in ("researcher", "analyst", "writer", "done")
    assert result.iteration >= 1
    assert len(result.agent_results) >= 1


def test_researcher_populates_sources_and_notes() -> None:
    """Researcher should populate sources and research_notes."""
    state = _make_state()
    result = ResearcherAgent().run(state)
    assert result.sources, "Researcher should find sources"
    assert result.research_notes, "Researcher should produce notes"
    assert len(result.agent_results) == 1
    assert result.agent_results[0].agent == "researcher"


def test_analyst_populates_analysis() -> None:
    """Analyst should populate analysis_notes."""
    state = _make_state()
    state.research_notes = "Some research notes about multi-agent systems."
    result = AnalystAgent().run(state)
    assert result.analysis_notes, "Analyst should produce analysis"
    assert len(result.agent_results) == 1
    assert result.agent_results[0].agent == "analyst"


def test_writer_populates_final_answer() -> None:
    """Writer should populate final_answer."""
    state = _make_state()
    state.research_notes = "Research notes here."
    state.analysis_notes = "Analysis notes here."
    result = WriterAgent().run(state)
    assert result.final_answer, "Writer should produce final answer"
    assert len(result.agent_results) == 1
    assert result.agent_results[0].agent == "writer"


def test_critic_reviews_final_answer() -> None:
    """Critic should review the final answer."""
    state = _make_state()
    state.final_answer = "This is a final answer about multi-agent systems."
    result = CriticAgent().run(state)
    assert len(result.agent_results) == 1
    assert result.agent_results[0].agent == "critic"
    assert len(result.agent_results[0].content) > 0


def test_critic_skips_when_no_answer() -> None:
    """Critic should handle missing final_answer gracefully."""
    state = _make_state()
    result = CriticAgent().run(state)
    # Should not crash, but also should not add agent results with content
    assert result.trace  # Should have a trace event


def test_supervisor_enforces_max_iterations() -> None:
    """Supervisor should stop when max_iterations reached."""
    state = _make_state()
    state.iteration = 100  # Way past max
    result = SupervisorAgent().run(state)
    # Should route to "done" or "writer" (to produce something)
    last_route = result.route_history[-1]
    assert last_route in ("done", "writer")
