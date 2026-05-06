"""Critic agent.

Reviews the final answer for fact-checking, citation coverage, and quality.
"""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a critic/reviewer agent. Your job is to review a final answer and provide:\n"
    "- Fact-check results: verify claims against source material\n"
    "- Citation coverage: what percentage of claims have citations\n"
    "- Quality assessment: overall score out of 10 with justification\n"
    "- Specific recommendations for improvement\n"
    "Write your review in markdown format."
)


class CriticAgent(BaseAgent):
    """Optional fact-checking and safety-review agent."""

    name = "critic"

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self._llm = llm_client or LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Validate final answer and append findings.

        Steps:
        1. Review final_answer against research_notes and sources
        2. Produce a quality review
        3. Append review to agent_results (does not overwrite final_answer)
        """
        with trace_span("critic_run") as span:
            if not state.final_answer:
                logger.warning("Critic called without final answer. Skipping review.")
                state.add_trace_event("critic", {"skipped": True, "reason": "no_final_answer"})
                return state

            source_refs = "\n".join(
                f"[{i+1}] {s.title}: {s.snippet[:100]}"
                for i, s in enumerate(state.sources)
            ) if state.sources else "No sources available."

            user_prompt = (
                f"Query: {state.request.query}\n\n"
                f"Final Answer to Review:\n{state.final_answer}\n\n"
                f"Research Notes:\n{state.research_notes or 'N/A'}\n\n"
                f"Source Material:\n{source_refs}\n\n"
                f"Please provide a thorough review with fact-checking and quality score."
            )

            response = self._llm.complete(_SYSTEM_PROMPT, user_prompt)

            logger.info(
                "Critic completed: %d chars review, %d tokens",
                len(response.content),
                (response.input_tokens or 0) + (response.output_tokens or 0),
            )

            state.add_trace_event("critic", {
                "review_length": len(response.content),
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "cost_usd": response.cost_usd,
            })
            state.agent_results.append(AgentResult(
                agent=AgentName.CRITIC,
                content=response.content,
                metadata={
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "cost_usd": response.cost_usd,
                },
            ))

            span["review_length"] = len(response.content)

        return state
