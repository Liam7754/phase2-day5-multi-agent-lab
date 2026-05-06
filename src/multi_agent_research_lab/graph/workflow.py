"""Multi-agent workflow orchestration.

Implements the supervisor-worker pattern:
  Supervisor -> Researcher -> Analyst -> Writer -> Critic -> Done

Uses a simple loop-based approach (no LangGraph dependency required).
If LangGraph is available, it builds a proper state graph.
"""

import logging
import time

from multi_agent_research_lab.agents.analyst import AnalystAgent
from multi_agent_research_lab.agents.critic import CriticAgent
from multi_agent_research_lab.agents.researcher import ResearcherAgent
from multi_agent_research_lab.agents.supervisor import SupervisorAgent
from multi_agent_research_lab.agents.writer import WriterAgent
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.run_logger import RunLogger
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient

logger = logging.getLogger(__name__)


class MultiAgentWorkflow:
    """Builds and runs the multi-agent graph.

    The workflow follows the supervisor pattern:
    1. Supervisor decides which agent to run next
    2. Worker agent executes and updates shared state
    3. Loop back to supervisor until "done" or max iterations
    """

    def __init__(self) -> None:
        self._llm = LLMClient()
        self._search = SearchClient()

        self._agents = {
            "supervisor": SupervisorAgent(llm_client=self._llm),
            "researcher": ResearcherAgent(llm_client=self._llm, search_client=self._search),
            "analyst": AnalystAgent(llm_client=self._llm),
            "writer": WriterAgent(llm_client=self._llm),
            "critic": CriticAgent(llm_client=self._llm),
        }

    def build(self) -> dict:
        """Create the workflow graph definition.

        Returns a dict describing the graph nodes and edges for traceability.
        """
        graph_def = {
            "nodes": list(self._agents.keys()),
            "edges": [
                {"from": "supervisor", "to": "researcher", "condition": "no research_notes"},
                {"from": "supervisor", "to": "analyst", "condition": "no analysis_notes"},
                {"from": "supervisor", "to": "writer", "condition": "no final_answer"},
                {"from": "supervisor", "to": "critic", "condition": "has final_answer, not reviewed"},
                {"from": "supervisor", "to": "done", "condition": "all complete or max iterations"},
                {"from": "researcher", "to": "supervisor", "condition": "always"},
                {"from": "analyst", "to": "supervisor", "condition": "always"},
                {"from": "writer", "to": "supervisor", "condition": "always"},
                {"from": "critic", "to": "supervisor", "condition": "always"},
            ],
            "entry_point": "supervisor",
            "stop_condition": "route == 'done' or iteration >= max_iterations",
        }

        logger.info("Workflow graph built: %d nodes", len(graph_def["nodes"]))
        return graph_def

    def run(self, state: ResearchState, run_logger: RunLogger | None = None) -> ResearchState:
        """Execute the workflow loop and return final state.

        Loop:
        1. Supervisor picks next route
        2. Execute the chosen worker agent
        3. Repeat until "done" or max iterations reached
        """
        settings = get_settings()
        graph_def = self.build()

        state.add_trace_event("workflow_start", {
            "query": state.request.query,
            "max_iterations": settings.max_iterations,
            "timeout_seconds": settings.timeout_seconds,
            "graph": graph_def,
        })

        start_time = time.perf_counter()

        with trace_span("workflow_run", {"query": state.request.query}) as span:
            while True:
                elapsed = time.perf_counter() - start_time

                # Guardrail: timeout
                if elapsed > settings.timeout_seconds:
                    logger.warning("Timeout reached (%.1fs > %ds)", elapsed, settings.timeout_seconds)
                    state.errors.append(f"Timeout after {elapsed:.1f}s")
                    state.add_trace_event("timeout", {"elapsed": elapsed})
                    if not state.final_answer:
                        # Emergency: force writer
                        logger.info("Forcing writer due to timeout")
                        writer_event = run_logger.log_agent_start("writer") if run_logger else None
                        state = self._agents["writer"].run(state)
                        if writer_event and run_logger:
                            self._log_agent_completion(run_logger, writer_event, state, "writer")
                    break

                # Step 1: Supervisor decides
                sup_event = run_logger.log_agent_start("supervisor") if run_logger else None
                state = self._agents["supervisor"].run(state)
                if sup_event and run_logger:
                    self._log_agent_completion(run_logger, sup_event, state, "supervisor")

                # Get the latest route decision
                if not state.route_history:
                    logger.error("No route history after supervisor. Breaking.")
                    state.errors.append("Supervisor produced no route")
                    break

                route = state.route_history[-1]
                logger.info(
                    "Iteration %d: route=%s, elapsed=%.2fs",
                    state.iteration, route, elapsed,
                )

                # Step 2: Check stop condition
                if route == "done":
                    state.add_trace_event("workflow_done", {
                        "iterations": state.iteration,
                        "elapsed_seconds": elapsed,
                    })
                    break

                # Step 3: Execute worker agent
                if route not in self._agents:
                    logger.error("Unknown route: %s", route)
                    state.errors.append(f"Unknown route: {route}")
                    break

                worker_event = run_logger.log_agent_start(route) if run_logger else None
                try:
                    state = self._agents[route].run(state)
                    if worker_event and run_logger:
                        self._log_agent_completion(run_logger, worker_event, state, route)
                except Exception as e:
                    logger.error("Agent '%s' failed: %s", route, e)
                    state.errors.append(f"Agent {route} failed: {e}")
                    if worker_event and run_logger:
                        run_logger.log_agent_end(worker_event, status="error", details={"error": str(e)})
                    state.add_trace_event("agent_error", {"agent": route, "error": str(e)})

            # Record workflow completion
            total_elapsed = time.perf_counter() - start_time
            span["iterations"] = state.iteration
            span["total_seconds"] = total_elapsed
            span["has_answer"] = state.final_answer is not None

            state.add_trace_event("workflow_complete", {
                "iterations": state.iteration,
                "total_seconds": round(total_elapsed, 3),
                "route_history": state.route_history,
                "has_final_answer": state.final_answer is not None,
                "error_count": len(state.errors),
            })

        return state

    def _log_agent_completion(
        self,
        run_logger: RunLogger,
        event: dict,
        state: ResearchState,
        agent_name: str,
    ) -> None:
        """Extract token info from the latest agent result and log it."""
        # Find the most recent result for this agent
        latest = None
        for r in reversed(state.agent_results):
            if r.agent == agent_name:
                latest = r
                break

        input_tokens = latest.metadata.get("input_tokens", 0) if latest else 0
        output_tokens = latest.metadata.get("output_tokens", 0) if latest else 0
        cost_usd = latest.metadata.get("cost_usd", 0.0) if latest else 0.0

        run_logger.log_agent_end(
            event,
            input_tokens=input_tokens or 0,
            output_tokens=output_tokens or 0,
            cost_usd=cost_usd or 0.0,
            details={"agent": agent_name, "iteration": state.iteration},
        )
