"""Supervisor / router agent.

Inspects current state and decides which worker agent should run next.
Enforces max iterations and failure fallback.
"""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a supervisor/router agent. Your job is to decide which worker agent "
    "should run next based on the current state of the research workflow.\n\n"
    "Available agents:\n"
    "- researcher: Searches for sources and creates research notes\n"
    "- analyst: Analyzes research notes and extracts insights\n"
    "- writer: Synthesizes final answer from research and analysis\n"
    "- critic: Reviews and fact-checks the final answer\n"
    "- done: All work is complete\n\n"
    "Respond with ONLY the agent name (one word). Do not explain."
)


class SupervisorAgent(BaseAgent):
    """Decides which worker should run next and when to stop."""

    name = "supervisor"

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self._llm = llm_client or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Update `state.route_history` with the next route.

        Routing policy:
        1. If no research_notes -> send to researcher
        2. If no analysis_notes -> send to analyst
        3. If no final_answer -> send to writer
        4. If final_answer exists but not reviewed -> send to critic
        5. Otherwise -> done
        Enforces max_iterations. Falls back to writer if stuck.
        """
        settings = get_settings()

        with trace_span("supervisor_decision", {"iteration": state.iteration}) as span:
            # Guardrail: max iterations
            if state.iteration >= settings.max_iterations:
                logger.warning(
                    "Max iterations (%d) reached. Forcing completion.",
                    settings.max_iterations,
                )
                route = "done"
                if not state.final_answer:
                    route = "writer"  # Force writer to produce something
                state.record_route(route)
                state.add_trace_event("supervisor", {
                    "decision": route,
                    "reason": "max_iterations_reached",
                    "iteration": state.iteration,
                })
                span["route"] = route
                return state

            # Build state summary for LLM
            state_summary = (
                f"Current state:\n"
                f"- Query: {state.request.query}\n"
                f"- Iteration: {state.iteration}\n"
                f"- Route history: {state.route_history}\n"
                f"- research_notes: {'present (' + str(len(state.research_notes)) + ' chars)' if state.research_notes else 'None'}\n"
                f"- analysis_notes: {'present (' + str(len(state.analysis_notes)) + ' chars)' if state.analysis_notes else 'None'}\n"
                f"- final_answer: {'present (' + str(len(state.final_answer)) + ' chars)' if state.final_answer else 'None'}\n"
                f"- Sources found: {len(state.sources)}\n"
                f"- Errors: {state.errors}\n"
            )

            response = self._llm.complete(_SYSTEM_PROMPT, state_summary)

            # Parse the route from LLM response
            raw_route = response.content.strip().lower()
            valid_routes = {"researcher", "analyst", "writer", "critic", "done"}
            route = raw_route if raw_route in valid_routes else self._fallback_route(state)

            logger.info(
                "Supervisor decision: iteration=%d, route=%s (raw=%r)",
                state.iteration, route, raw_route,
            )

            state.record_route(route)
            state.add_trace_event("supervisor", {
                "decision": route,
                "raw_llm_response": raw_route,
                "iteration": state.iteration,
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "cost_usd": response.cost_usd,
            })
            state.agent_results.append(AgentResult(
                agent=AgentName.SUPERVISOR,
                content=f"Route to: {route}",
                metadata={
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "cost_usd": response.cost_usd,
                },
            ))

            span["route"] = route

        return state

    def _fallback_route(self, state: ResearchState) -> str:
        """Deterministic fallback when LLM gives an invalid response."""
        if not state.research_notes:
            return "researcher"
        if not state.analysis_notes:
            return "analyst"
        if not state.final_answer:
            return "writer"
        if "critic" not in state.route_history:
            return "critic"
        return "done"
