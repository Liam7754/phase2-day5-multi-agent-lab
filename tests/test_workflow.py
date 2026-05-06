"""Tests for the multi-agent workflow."""

from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.observability.run_logger import RunLogger


def test_workflow_produces_final_answer() -> None:
    """Full workflow should produce a final answer."""
    state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))
    workflow = MultiAgentWorkflow()
    result = workflow.run(state)
    assert result.final_answer, "Workflow should produce a final answer"
    assert result.iteration > 0
    assert result.route_history, "Should have route history"
    assert result.agent_results, "Should have agent results"


def test_workflow_with_run_logger(tmp_path) -> None:
    """Workflow should work with RunLogger integration."""
    state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))
    run_log = RunLogger(mode="multi-agent", query="test", logs_dir=tmp_path)
    workflow = MultiAgentWorkflow()
    result = workflow.run(state, run_logger=run_log)

    assert result.final_answer
    log_path = run_log.finalize(result)
    assert log_path.exists()


def test_workflow_build_returns_graph_def() -> None:
    """Build should return a graph definition dict."""
    workflow = MultiAgentWorkflow()
    graph_def = workflow.build()
    assert "nodes" in graph_def
    assert "edges" in graph_def
    assert "supervisor" in graph_def["nodes"]
    assert "researcher" in graph_def["nodes"]
