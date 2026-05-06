"""Benchmark report rendering with rich analysis."""

from datetime import datetime, timezone

from multi_agent_research_lab.core.schemas import BenchmarkMetrics


def render_markdown_report(metrics: list[BenchmarkMetrics]) -> str:
    """Render benchmark metrics to a comprehensive markdown report.

    Includes:
    - Summary table
    - Comparative analysis
    - Recommendations
    """
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    lines = [
        "# Benchmark Report",
        "",
        f"**Generated:** {now}",
        f"**Runs compared:** {len(metrics)}",
        "",
        "## Summary Table",
        "",
        "| Run | Latency (s) | Cost (USD) | Quality (/10) | Notes |",
        "|---|---:|---:|---:|---|",
    ]

    for item in metrics:
        cost = f"${item.estimated_cost_usd:.6f}" if item.estimated_cost_usd is not None else "-"
        quality = f"{item.quality_score:.1f}" if item.quality_score is not None else "-"
        lines.append(f"| {item.run_name} | {item.latency_seconds:.3f} | {cost} | {quality} | {item.notes} |")

    # Comparative analysis
    if len(metrics) >= 2:
        baseline = next((m for m in metrics if "baseline" in m.run_name.lower()), metrics[0])
        multi = next((m for m in metrics if "multi" in m.run_name.lower()), metrics[-1])

        latency_ratio = multi.latency_seconds / baseline.latency_seconds if baseline.latency_seconds > 0 else 0
        cost_ratio = (
            (multi.estimated_cost_usd or 0) / (baseline.estimated_cost_usd or 1)
            if baseline.estimated_cost_usd and baseline.estimated_cost_usd > 0
            else 0
        )
        quality_diff = (multi.quality_score or 0) - (baseline.quality_score or 0)

        lines.extend([
            "",
            "## Comparative Analysis",
            "",
            f"### Latency",
            f"- Baseline: {baseline.latency_seconds:.3f}s",
            f"- Multi-agent: {multi.latency_seconds:.3f}s",
            f"- **Ratio:** {latency_ratio:.1f}x {'slower' if latency_ratio > 1 else 'faster'}",
            "",
            f"### Cost",
            f"- Baseline: ${baseline.estimated_cost_usd or 0:.6f}",
            f"- Multi-agent: ${multi.estimated_cost_usd or 0:.6f}",
            f"- **Ratio:** {cost_ratio:.1f}x",
            "",
            f"### Quality",
            f"- Baseline: {baseline.quality_score or 0:.1f}/10",
            f"- Multi-agent: {multi.quality_score or 0:.1f}/10",
            f"- **Improvement:** {'+' if quality_diff >= 0 else ''}{quality_diff:.1f} points",
            "",
            "## Key Observations",
            "",
            f"1. Multi-agent system is **{latency_ratio:.1f}x** {'slower' if latency_ratio > 1 else 'faster'} "
            f"than baseline due to multiple agent calls and LLM invocations.",
            f"2. Cost increases by **{cost_ratio:.1f}x** with multi-agent, reflecting additional LLM calls "
            f"for research, analysis, writing, and review.",
            f"3. Quality {'improves' if quality_diff > 0 else 'decreases'} by **{abs(quality_diff):.1f} points** "
            f"thanks to specialized agent roles and iterative refinement.",
            "",
            "## Trade-off Analysis",
            "",
            "| Factor | Single-Agent (Baseline) | Multi-Agent | Winner |",
            "|--------|------------------------|-------------|--------|",
            f"| Latency | {baseline.latency_seconds:.3f}s | {multi.latency_seconds:.3f}s | "
            f"{'Baseline' if baseline.latency_seconds < multi.latency_seconds else 'Multi-Agent'} |",
            f"| Cost | ${baseline.estimated_cost_usd or 0:.6f} | ${multi.estimated_cost_usd or 0:.6f} | "
            f"{'Baseline' if (baseline.estimated_cost_usd or 0) < (multi.estimated_cost_usd or 0) else 'Multi-Agent'} |",
            f"| Quality | {baseline.quality_score or 0:.1f}/10 | {multi.quality_score or 0:.1f}/10 | "
            f"{'Multi-Agent' if (multi.quality_score or 0) > (baseline.quality_score or 0) else 'Baseline'} |",
            "| Traceability | Limited | Full agent-level | Multi-Agent |",
            "| Debugging | Harder | Easier (per-agent) | Multi-Agent |",
            "",
            "## Recommendations",
            "",
            "- Use **single-agent** for simple queries where latency and cost matter most.",
            "- Use **multi-agent** for complex research tasks requiring multiple perspectives.",
            "- Consider **hybrid approach**: route simple queries to baseline, complex to multi-agent.",
            "",
        ])

    lines.extend([
        "---",
        f"*Report generated at {now}*",
        "",
    ])

    return "\n".join(lines)
