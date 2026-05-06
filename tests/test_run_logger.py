"""Tests for the JSON run logger."""

import json

from multi_agent_research_lab.core.schemas import BenchmarkMetrics, ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.run_logger import RunLogger, list_run_logs


def test_run_logger_creates_log_file(tmp_path) -> None:
    """RunLogger should create a JSON log file."""
    state = ResearchState(request=ResearchQuery(query="Test query for logging"))
    run_log = RunLogger(mode="test", query="Test query", logs_dir=tmp_path)

    event = run_log.log_agent_start("test_agent")
    run_log.log_agent_end(event, input_tokens=100, output_tokens=50, cost_usd=0.001)

    log_path = run_log.finalize(state)
    assert log_path.exists()

    data = json.loads(log_path.read_text(encoding="utf-8"))
    assert data["mode"] == "test"
    assert data["query"] == "Test query"
    assert data["summary"]["total_agents_invoked"] == 1
    assert data["summary"]["total_input_tokens"] == 100
    assert data["summary"]["total_output_tokens"] == 50


def test_run_logger_tracks_multiple_agents(tmp_path) -> None:
    """RunLogger should track multiple agent events."""
    state = ResearchState(request=ResearchQuery(query="Multi agent test"))
    run_log = RunLogger(mode="multi-agent", query="test", logs_dir=tmp_path)

    for agent in ["supervisor", "researcher", "analyst", "writer"]:
        event = run_log.log_agent_start(agent)
        run_log.log_agent_end(event, input_tokens=50, output_tokens=25, cost_usd=0.0005)

    log_path = run_log.finalize(state)
    data = json.loads(log_path.read_text(encoding="utf-8"))

    assert data["summary"]["total_agents_invoked"] == 4
    assert data["summary"]["agent_sequence"] == ["supervisor", "researcher", "analyst", "writer"]
    assert data["summary"]["total_llm_calls"] == 4


def test_run_logger_with_benchmark(tmp_path) -> None:
    """RunLogger should include benchmark metrics when provided."""
    state = ResearchState(request=ResearchQuery(query="Benchmark test"))
    metrics = BenchmarkMetrics(
        run_name="test",
        latency_seconds=1.5,
        estimated_cost_usd=0.005,
        quality_score=7.5,
    )
    run_log = RunLogger(mode="test", query="test", logs_dir=tmp_path)
    log_path = run_log.finalize(state, metrics)

    data = json.loads(log_path.read_text(encoding="utf-8"))
    assert "benchmark" in data
    assert data["benchmark"]["latency_seconds"] == 1.5
    assert data["benchmark"]["quality_score"] == 7.5


def test_list_run_logs(tmp_path) -> None:
    """list_run_logs should read all log files."""
    state = ResearchState(request=ResearchQuery(query="Test query"))

    for i in range(3):
        rl = RunLogger(mode="test", query=f"query_{i}", logs_dir=tmp_path)
        rl.finalize(state)

    logs = list_run_logs(tmp_path)
    assert len(logs) == 3
