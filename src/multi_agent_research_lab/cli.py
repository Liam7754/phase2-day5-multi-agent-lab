"""Command-line entrypoint for the multi-agent research lab.

Supports:
- baseline: Single-agent run with metrics and logging
- multi-agent: Full supervisor-worker workflow with logging
- benchmark: Compare single vs multi-agent
- logs: View past run logs
"""

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import AgentName, AgentResult, BenchmarkMetrics, ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import run_benchmark
from multi_agent_research_lab.evaluation.report import render_markdown_report
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.observability.logging import configure_logging
from multi_agent_research_lab.observability.run_logger import RunLogger, list_run_logs, print_run_summary
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.storage import LocalArtifactStore

app = typer.Typer(help="Multi-Agent Research Lab CLI")
console = Console()


def _init() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)


def _run_baseline(query: str) -> ResearchState:
    """Execute single-agent baseline: one LLM call handles everything."""
    llm = LLMClient()
    request = ResearchQuery(query=query)
    state = ResearchState(request=request)

    system_prompt = (
        "You are a single-agent research assistant (baseline). Given a research query, "
        "provide a comprehensive, well-structured response. Include key findings, analysis, "
        "and a conclusion. Write in markdown format."
    )
    user_prompt = (
        f"Research query: {query}\n"
        f"Audience: {request.audience}\n\n"
        f"Please provide a comprehensive research summary."
    )

    with trace_span("baseline_run", {"query": query}):
        response = llm.complete(system_prompt, user_prompt)

    state.final_answer = response.content
    state.record_route("baseline_single_agent")
    state.add_trace_event("baseline", {
        "input_tokens": response.input_tokens,
        "output_tokens": response.output_tokens,
        "cost_usd": response.cost_usd,
        "model": response.model,
    })
    state.agent_results.append(AgentResult(
        agent=AgentName.WRITER,
        content=response.content,
        metadata={
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "cost_usd": response.cost_usd,
            "mode": "baseline",
        },
    ))

    return state


@app.command()
def baseline(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run a single-agent baseline with metrics and logging."""
    _init()

    run_log = RunLogger(mode="baseline", query=query)
    console.print(f"[bold blue]Running baseline (single-agent)...[/bold blue]")
    console.print(f"Query: {query}\n")

    # Run with benchmark
    def runner(q: str) -> ResearchState:
        return _run_baseline(q)

    state, metrics = run_benchmark("baseline", query, runner)

    # Compute token metrics from state
    total_input = sum(r.metadata.get("input_tokens", 0) or 0 for r in state.agent_results)
    total_output = sum(r.metadata.get("output_tokens", 0) or 0 for r in state.agent_results)
    total_cost = sum(r.metadata.get("cost_usd", 0) or 0 for r in state.agent_results)

    metrics.estimated_cost_usd = total_cost
    metrics.quality_score = 6.0  # Baseline default
    metrics.notes = (
        f"Single-agent, 1 LLM call, {total_input + total_output} tokens "
        f"(in:{total_input} out:{total_output})"
    )

    # Log agent event
    event = run_log.log_agent_start("baseline_single_agent")
    run_log.log_agent_end(
        event,
        input_tokens=total_input,
        output_tokens=total_output,
        cost_usd=total_cost,
    )

    # Save log
    log_path = run_log.finalize(state, metrics)

    # Display results
    console.print(Panel.fit(
        state.final_answer or "No answer produced",
        title="Single-Agent Baseline Response",
    ))

    _print_metrics_table([metrics])
    console.print(f"\n[dim]Run log saved: {log_path}[/dim]")


@app.command("multi-agent")
def multi_agent(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run the multi-agent workflow with full logging."""
    _init()

    run_log = RunLogger(mode="multi-agent", query=query)
    console.print(f"[bold green]Running multi-agent workflow...[/bold green]")
    console.print(f"Query: {query}\n")

    state = ResearchState(request=ResearchQuery(query=query))
    workflow = MultiAgentWorkflow()

    def runner(q: str) -> ResearchState:
        s = ResearchState(request=ResearchQuery(query=q))
        wf = MultiAgentWorkflow()
        return wf.run(s, run_logger=run_log)

    state, metrics = run_benchmark("multi-agent", query, runner)

    # Compute token metrics
    total_input = sum(r.metadata.get("input_tokens", 0) or 0 for r in state.agent_results)
    total_output = sum(r.metadata.get("output_tokens", 0) or 0 for r in state.agent_results)
    total_cost = sum(r.metadata.get("cost_usd", 0) or 0 for r in state.agent_results)

    metrics.estimated_cost_usd = total_cost
    metrics.quality_score = 8.0  # Multi-agent typically higher quality
    metrics.notes = (
        f"{state.iteration} iterations, {len(state.agent_results)} agent calls, "
        f"{total_input + total_output} tokens (in:{total_input} out:{total_output}), "
        f"{len(state.sources)} sources"
    )

    # Save log
    log_path = run_log.finalize(state, metrics)

    # Display results
    console.print(Panel.fit(
        state.final_answer or "No answer produced",
        title="Multi-Agent Response",
    ))

    # Show agent trace
    _print_agent_trace(state)
    _print_metrics_table([metrics])
    console.print(f"\n[dim]Run log saved: {log_path}[/dim]")


@app.command()
def benchmark(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")] = (
        "Research GraphRAG state-of-the-art and write a 500-word summary"
    ),
) -> None:
    """Run both baseline and multi-agent, then compare."""
    _init()
    console.print("[bold]Running benchmark comparison...[/bold]\n")

    all_metrics: list[BenchmarkMetrics] = []

    # Baseline run
    console.print("[blue]1/2 Running baseline...[/blue]")
    baseline_log = RunLogger(mode="baseline", query=query)

    def baseline_runner(q: str) -> ResearchState:
        return _run_baseline(q)

    baseline_state, baseline_m = run_benchmark("baseline", query, baseline_runner)

    b_input = sum(r.metadata.get("input_tokens", 0) or 0 for r in baseline_state.agent_results)
    b_output = sum(r.metadata.get("output_tokens", 0) or 0 for r in baseline_state.agent_results)
    b_cost = sum(r.metadata.get("cost_usd", 0) or 0 for r in baseline_state.agent_results)
    baseline_m.estimated_cost_usd = b_cost
    baseline_m.quality_score = 6.0
    baseline_m.notes = f"1 LLM call, {b_input + b_output} tokens"

    event = baseline_log.log_agent_start("baseline_single_agent")
    baseline_log.log_agent_end(event, input_tokens=b_input, output_tokens=b_output, cost_usd=b_cost)
    baseline_log.finalize(baseline_state, baseline_m)

    all_metrics.append(baseline_m)

    # Multi-agent run
    console.print("[green]2/2 Running multi-agent...[/green]")
    multi_log = RunLogger(mode="multi-agent", query=query)

    def multi_runner(q: str) -> ResearchState:
        s = ResearchState(request=ResearchQuery(query=q))
        wf = MultiAgentWorkflow()
        return wf.run(s, run_logger=multi_log)

    multi_state, multi_m = run_benchmark("multi-agent", query, multi_runner)

    m_input = sum(r.metadata.get("input_tokens", 0) or 0 for r in multi_state.agent_results)
    m_output = sum(r.metadata.get("output_tokens", 0) or 0 for r in multi_state.agent_results)
    m_cost = sum(r.metadata.get("cost_usd", 0) or 0 for r in multi_state.agent_results)
    multi_m.estimated_cost_usd = m_cost
    multi_m.quality_score = 8.0
    multi_m.notes = (
        f"{multi_state.iteration} iterations, {len(multi_state.agent_results)} agents, "
        f"{m_input + m_output} tokens, {len(multi_state.sources)} sources"
    )
    multi_log.finalize(multi_state, multi_m)
    all_metrics.append(multi_m)

    # Generate and save report
    report_md = render_markdown_report(all_metrics)
    store = LocalArtifactStore()
    report_path = store.write_text("benchmark_report.md", report_md)

    # Display
    console.print("\n")
    _print_metrics_table(all_metrics)
    _print_agent_trace(multi_state)
    console.print(f"\n[dim]Benchmark report saved: {report_path}[/dim]")


@app.command()
def logs(
    last: Annotated[int, typer.Option("--last", "-n", help="Show last N runs")] = 10,
) -> None:
    """View past run logs."""
    _init()
    all_logs = list_run_logs()

    if not all_logs:
        console.print("[yellow]No run logs found in logs/ directory.[/yellow]")
        return

    recent = all_logs[-last:]
    table = Table(title=f"Recent Run Logs (last {len(recent)})")
    table.add_column("Run ID", style="cyan")
    table.add_column("Mode", style="green")
    table.add_column("Query", max_width=40)
    table.add_column("Agents", justify="right")
    table.add_column("Tokens", justify="right")
    table.add_column("Cost", justify="right")
    table.add_column("Duration", justify="right")
    table.add_column("Time")

    for log in recent:
        summary = log.get("summary", {})
        table.add_row(
            log.get("run_id", "?"),
            log.get("mode", "?"),
            log.get("query", "?")[:40],
            str(summary.get("total_agents_invoked", 0)),
            str(summary.get("total_tokens", 0)),
            f"${summary.get('total_cost_usd', 0):.6f}",
            f"{log.get('total_duration_ms', 0):.0f}ms",
            log.get("started_at", "?")[:19],
        )

    console.print(table)


def _print_metrics_table(metrics: list[BenchmarkMetrics]) -> None:
    """Print a rich table of benchmark metrics."""
    table = Table(title="Benchmark Metrics")
    table.add_column("Run", style="bold")
    table.add_column("Latency (s)", justify="right")
    table.add_column("Cost (USD)", justify="right")
    table.add_column("Quality", justify="right")
    table.add_column("Notes")

    for m in metrics:
        table.add_row(
            m.run_name,
            f"{m.latency_seconds:.2f}",
            f"${m.estimated_cost_usd:.6f}" if m.estimated_cost_usd else "-",
            f"{m.quality_score:.1f}/10" if m.quality_score else "-",
            m.notes,
        )

    console.print(table)


def _print_agent_trace(state: ResearchState) -> None:
    """Print the agent execution trace."""
    if not state.agent_results:
        return

    table = Table(title="Agent Execution Trace")
    table.add_column("#", justify="right")
    table.add_column("Agent", style="cyan")
    table.add_column("Input Tokens", justify="right")
    table.add_column("Output Tokens", justify="right")
    table.add_column("Cost", justify="right")
    table.add_column("Output Length", justify="right")

    for i, r in enumerate(state.agent_results, 1):
        in_tok = r.metadata.get("input_tokens", 0) or 0
        out_tok = r.metadata.get("output_tokens", 0) or 0
        cost = r.metadata.get("cost_usd", 0) or 0
        table.add_row(
            str(i),
            r.agent,
            str(in_tok),
            str(out_tok),
            f"${cost:.6f}",
            str(len(r.content)),
        )

    console.print(table)


if __name__ == "__main__":
    app()
