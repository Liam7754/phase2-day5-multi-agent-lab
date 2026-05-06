"""Analyst agent.

Extracts key claims, compares viewpoints, flags weak evidence, and produces analysis notes.
"""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are an analyst agent. Your job is to critically analyze research notes and "
    "produce structured insights. You should:\n"
    "- Extract and evaluate key claims (strong, moderate, weak evidence)\n"
    "- Compare different viewpoints and identify consensus/disagreements\n"
    "- Create comparative tables where appropriate\n"
    "- Flag gaps in evidence or areas needing more research\n"
    "- Highlight the most important insight\n"
    "Write your analysis in markdown format."
)


class AnalystAgent(BaseAgent):
    """Turns research notes into structured insights."""

    name = "analyst"

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self._llm = llm_client or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.analysis_notes`.

        Steps:
        1. Gather research notes and source summaries
        2. Use LLM to produce critical analysis
        """
        with trace_span("analyst_run") as span:
            if not state.research_notes:
                logger.warning("Analyst called without research notes. Using sources directly.")

            # Build context
            source_summary = "\n".join(
                f"- [{s.title}]({s.url}): {s.snippet[:120]}..."
                for s in state.sources[:10]
            ) if state.sources else "No sources available."

            user_prompt = (
                f"Query: {state.request.query}\n"
                f"Audience: {state.request.audience}\n\n"
                f"Research Notes:\n{state.research_notes or 'No research notes available.'}\n\n"
                f"Sources ({len(state.sources)}):\n{source_summary}\n\n"
                f"Please provide a critical analysis with evidence evaluation and key insights."
            )

            response = self._llm.complete(_SYSTEM_PROMPT, user_prompt)
            state.analysis_notes = response.content

            logger.info(
                "Analyst completed: %d chars of analysis, %d tokens",
                len(response.content),
                (response.input_tokens or 0) + (response.output_tokens or 0),
            )

            state.add_trace_event("analyst", {
                "analysis_length": len(response.content),
                "had_research_notes": state.research_notes is not None,
                "source_count": len(state.sources),
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "cost_usd": response.cost_usd,
            })
            state.agent_results.append(AgentResult(
                agent=AgentName.ANALYST,
                content=response.content,
                metadata={
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "cost_usd": response.cost_usd,
                },
            ))

            span["analysis_length"] = len(response.content)

        return state
