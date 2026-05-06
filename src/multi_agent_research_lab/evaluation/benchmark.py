"""Benchmark for single-agent vs multi-agent comparison.

Measures latency, estimates token cost, checks citation coverage, and counts errors.
"""

import logging
import re
from time import perf_counter
from typing import Callable

from multi_agent_research_lab.core.schemas import BenchmarkMetrics
from multi_agent_research_lab.core.state import ResearchState

logger = logging.getLogger(__name__)

Runner = Callable[[str], ResearchState]


def _count_citations(text: str | None) -> int:
    """Count citation references like [1], [2], etc."""
    if not text:
        return 0
    return len(re.findall(r"\[\d+\]", text))


def _estimate_quality(state: ResearchState) -> float:
    """Heuristic quality score (0-10) based on completeness metrics."""
    score = 0.0

    # Has final answer (3 points)
    if state.final_answer:
        score += 3.0
        # Length bonus (up to 2 points)
        length = len(state.final_answer)
        if length > 200:
            score += min(2.0, length / 1000)

    # Has sources (2 points)
    if state.sources:
        score += min(2.0, len(state.sources) * 0.4)

    # Citation coverage (2 points)
    citations = _count_citations(state.final_answer)
    if citations > 0:
        score += min(2.0, citations * 0.5)

    # No errors (1 point)
    if not state.errors:
        score += 1.0

    return min(10.0, round(score, 1))


def _estimate_total_cost(state: ResearchState) -> float:
    """Sum up cost from all agent results."""
    return sum(
        r.metadata.get("cost_usd", 0) or 0
        for r in state.agent_results
    )


def _count_total_tokens(state: ResearchState) -> tuple[int, int]:
    """Return (total_input_tokens, total_output_tokens)."""
    input_tok = sum(r.metadata.get("input_tokens", 0) or 0 for r in state.agent_results)
    output_tok = sum(r.metadata.get("output_tokens", 0) or 0 for r in state.agent_results)
    return input_tok, output_tok


def run_benchmark(run_name: str, query: str, runner: Runner) -> tuple[ResearchState, BenchmarkMetrics]:
    """Run a benchmark with comprehensive metrics.

    Measures:
    - Latency (wall-clock seconds)
    - Estimated token cost (USD)
    - Quality score (heuristic 0-10)
    - Citation coverage
    - Error rate
    """
    logger.info("Starting benchmark: %s (query=%r)", run_name, query)

    started = perf_counter()
    state = runner(query)
    latency = perf_counter() - started

    # Compute metrics
    total_cost = _estimate_total_cost(state)
    quality = _estimate_quality(state)
    input_tokens, output_tokens = _count_total_tokens(state)
    citations = _count_citations(state.final_answer)

    notes_parts = [
        f"{state.iteration} iterations",
        f"{len(state.agent_results)} agent calls",
        f"{input_tokens + output_tokens} tokens (in:{input_tokens} out:{output_tokens})",
        f"{len(state.sources)} sources",
        f"{citations} citations",
        f"{len(state.errors)} errors",
    ]

    metrics = BenchmarkMetrics(
        run_name=run_name,
        latency_seconds=round(latency, 3),
        estimated_cost_usd=round(total_cost, 8),
        quality_score=quality,
        notes=", ".join(notes_parts),
    )

    logger.info(
        "Benchmark '%s' complete: latency=%.3fs, cost=$%.6f, quality=%.1f/10",
        run_name, latency, total_cost, quality,
    )

    return state, metrics
