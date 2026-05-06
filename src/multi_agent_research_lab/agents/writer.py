"""Writer agent.

Synthesizes research notes and analysis into a clear, well-structured final answer.
"""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a writer agent. Your job is to synthesize research notes and analysis "
    "into a clear, well-structured final response. You should:\n"
    "- Write in a clear, engaging style appropriate for the target audience\n"
    "- Include an introduction, main body with key sections, and conclusion\n"
    "- Reference sources with numbered citations [1], [2], etc.\n"
    "- Include a References section at the end\n"
    "- Keep the response focused and concise (around 500 words)\n"
    "Write in markdown format."
)


class WriterAgent(BaseAgent):
    """Produces final answer from research and analysis notes."""

    name = "writer"

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self._llm = llm_client or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.final_answer`.

        Steps:
        1. Gather all available notes (research + analysis)
        2. Use LLM to synthesize final answer with citations
        """
        with trace_span("writer_run") as span:
            source_refs = "\n".join(
                f"[{i+1}] {s.title} - {s.url}"
                for i, s in enumerate(state.sources)
            ) if state.sources else "No sources available."

            user_prompt = (
                f"Query: {state.request.query}\n"
                f"Audience: {state.request.audience}\n\n"
                f"Research Notes:\n{state.research_notes or 'No research notes.'}\n\n"
                f"Analysis:\n{state.analysis_notes or 'No analysis notes.'}\n\n"
                f"Available Sources:\n{source_refs}\n\n"
                f"Please write a comprehensive final response with citations."
            )

            response = self._llm.complete(_SYSTEM_PROMPT, user_prompt)
            state.final_answer = response.content

            logger.info(
                "Writer completed: %d chars final answer, %d tokens",
                len(response.content),
                (response.input_tokens or 0) + (response.output_tokens or 0),
            )

            state.add_trace_event("writer", {
                "answer_length": len(response.content),
                "had_research": state.research_notes is not None,
                "had_analysis": state.analysis_notes is not None,
                "source_count": len(state.sources),
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "cost_usd": response.cost_usd,
            })
            state.agent_results.append(AgentResult(
                agent=AgentName.WRITER,
                content=response.content,
                metadata={
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "cost_usd": response.cost_usd,
                },
            ))

            span["answer_length"] = len(response.content)

        return state
