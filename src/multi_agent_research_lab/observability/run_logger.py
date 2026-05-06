"""JSON-based run logger for full traceability.

Logs every run (baseline and multi-agent) to a JSON file in the logs/ directory.
Each run file includes:
- Run metadata (id, mode, timestamp, query)
- Agent execution timeline (which agent, when, duration, tokens)
- Token usage summary
- Cost breakdown
- Final state snapshot
- Errors and warnings
"""

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from multi_agent_research_lab.core.schemas import BenchmarkMetrics
from multi_agent_research_lab.core.state import ResearchState

logger = logging.getLogger(__name__)

LOGS_DIR = Path("logs")


class RunLogger:
    """Tracks a single run and writes it to a JSON log file."""

    def __init__(self, mode: str, query: str, logs_dir: Path = LOGS_DIR) -> None:
        self.run_id = str(uuid.uuid4())[:8]
        self.mode = mode  # "baseline" or "multi-agent"
        self.query = query
        self.logs_dir = logs_dir
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        self.started_at = datetime.now(timezone.utc)
        self.ended_at: datetime | None = None

        # Agent timeline
        self.agent_events: list[dict[str, Any]] = []

        # Token tracking
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost_usd = 0.0
        self.llm_calls = 0

        # Search tracking
        self.search_calls = 0
        self.total_sources_found = 0

        # Errors
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def log_agent_start(self, agent_name: str) -> dict[str, Any]:
        """Record agent execution start. Returns event dict to update on completion."""
        event = {
            "agent": agent_name,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "ended_at": None,
            "duration_ms": None,
            "input_tokens": 0,
            "output_tokens": 0,
            "cost_usd": 0.0,
            "status": "running",
            "details": {},
        }
        self.agent_events.append(event)
        logger.info("[Run %s] Agent '%s' started", self.run_id, agent_name)
        return event

    def log_agent_end(
        self,
        event: dict[str, Any],
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost_usd: float = 0.0,
        status: str = "success",
        details: dict[str, Any] | None = None,
    ) -> None:
        """Update agent event with completion data."""
        now = datetime.now(timezone.utc)
        started = datetime.fromisoformat(event["started_at"])
        duration_ms = (now - started).total_seconds() * 1000

        event["ended_at"] = now.isoformat()
        event["duration_ms"] = round(duration_ms, 2)
        event["input_tokens"] = input_tokens
        event["output_tokens"] = output_tokens
        event["cost_usd"] = round(cost_usd, 8)
        event["status"] = status
        if details:
            event["details"] = details

        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cost_usd += cost_usd
        self.llm_calls += 1

        logger.info(
            "[Run %s] Agent '%s' completed: %d+%d tokens, $%.6f, %.1fms, status=%s",
            self.run_id, event["agent"], input_tokens, output_tokens, cost_usd, duration_ms, status,
        )

    def log_search(self, query: str, num_results: int) -> None:
        """Record a search operation."""
        self.search_calls += 1
        self.total_sources_found += num_results
        logger.info("[Run %s] Search: query=%r, results=%d", self.run_id, query, num_results)

    def log_error(self, error: str) -> None:
        self.errors.append(error)
        logger.error("[Run %s] Error: %s", self.run_id, error)

    def log_warning(self, warning: str) -> None:
        self.warnings.append(warning)
        logger.warning("[Run %s] Warning: %s", self.run_id, warning)

    def finalize(
        self,
        state: ResearchState,
        metrics: BenchmarkMetrics | None = None,
    ) -> Path:
        """Write the complete run log to a JSON file and return its path."""
        self.ended_at = datetime.now(timezone.utc)
        total_duration_ms = (self.ended_at - self.started_at).total_seconds() * 1000

        log_data = {
            "run_id": self.run_id,
            "mode": self.mode,
            "query": self.query,
            "started_at": self.started_at.isoformat(),
            "ended_at": self.ended_at.isoformat(),
            "total_duration_ms": round(total_duration_ms, 2),
            "summary": {
                "total_agents_invoked": len(self.agent_events),
                "agent_sequence": [e["agent"] for e in self.agent_events],
                "total_llm_calls": self.llm_calls,
                "total_input_tokens": self.total_input_tokens,
                "total_output_tokens": self.total_output_tokens,
                "total_tokens": self.total_input_tokens + self.total_output_tokens,
                "total_cost_usd": round(self.total_cost_usd, 8),
                "total_search_calls": self.search_calls,
                "total_sources_found": self.total_sources_found,
                "iterations": state.iteration,
                "has_final_answer": state.final_answer is not None,
                "error_count": len(self.errors),
            },
            "agent_timeline": self.agent_events,
            "route_history": state.route_history,
            "trace_events": state.trace,
            "sources": [s.model_dump() for s in state.sources],
            "errors": self.errors + state.errors,
            "warnings": self.warnings,
            "state_snapshot": {
                "research_notes_length": len(state.research_notes) if state.research_notes else 0,
                "analysis_notes_length": len(state.analysis_notes) if state.analysis_notes else 0,
                "final_answer_length": len(state.final_answer) if state.final_answer else 0,
                "num_sources": len(state.sources),
                "num_agent_results": len(state.agent_results),
            },
        }

        if metrics:
            log_data["benchmark"] = {
                "run_name": metrics.run_name,
                "latency_seconds": metrics.latency_seconds,
                "estimated_cost_usd": metrics.estimated_cost_usd,
                "quality_score": metrics.quality_score,
                "notes": metrics.notes,
            }

        # Write to file
        timestamp = self.started_at.strftime("%Y%m%d_%H%M%S")
        filename = f"run_{timestamp}_{self.mode}_{self.run_id}.json"
        filepath = self.logs_dir / filename

        filepath.write_text(json.dumps(log_data, indent=2, ensure_ascii=False), encoding="utf-8")

        logger.info(
            "[Run %s] Log saved: %s (mode=%s, agents=%d, tokens=%d, cost=$%.6f, duration=%.1fms)",
            self.run_id, filepath, self.mode,
            len(self.agent_events),
            self.total_input_tokens + self.total_output_tokens,
            self.total_cost_usd,
            total_duration_ms,
        )

        return filepath


def list_run_logs(logs_dir: Path = LOGS_DIR) -> list[dict[str, Any]]:
    """Read and return all run logs as a list of dicts, sorted by timestamp."""
    if not logs_dir.exists():
        return []

    logs = []
    for f in sorted(logs_dir.glob("run_*.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            logs.append(data)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to read log %s: %s", f, e)
    return logs


def print_run_summary(log_data: dict[str, Any]) -> str:
    """Format a single run log into a readable summary."""
    summary = log_data.get("summary", {})
    lines = [
        f"Run ID: {log_data['run_id']}",
        f"Mode: {log_data['mode']}",
        f"Query: {log_data['query']}",
        f"Started: {log_data['started_at']}",
        f"Duration: {log_data['total_duration_ms']:.1f}ms",
        f"Agents invoked: {summary.get('total_agents_invoked', 0)}",
        f"Agent sequence: {' -> '.join(summary.get('agent_sequence', []))}",
        f"LLM calls: {summary.get('total_llm_calls', 0)}",
        f"Total tokens: {summary.get('total_tokens', 0)} "
        f"(input: {summary.get('total_input_tokens', 0)}, output: {summary.get('total_output_tokens', 0)})",
        f"Cost: ${summary.get('total_cost_usd', 0):.6f}",
        f"Sources found: {summary.get('total_sources_found', 0)}",
        f"Errors: {summary.get('error_count', 0)}",
        f"Has final answer: {summary.get('has_final_answer', False)}",
    ]
    return "\n".join(lines)
