"""Researcher agent.

Searches for sources, filters results, captures citations, and produces research notes.
"""

import logging

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import trace_span
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a research agent. Your job is to synthesize search results into "
    "clear, well-organized research notes. Include key findings, cite sources, "
    "and highlight areas needing further investigation.\n"
    "Write comprehensive research notes in markdown format."
)


class ResearcherAgent(BaseAgent):
    """Collects sources and creates concise research notes."""

    name = "researcher"

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        search_client: SearchClient | None = None,
    ) -> None:
        self._llm = llm_client or LLMClient()
        self._search = search_client or SearchClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.sources` and `state.research_notes`.

        Steps:
        1. Search for sources using the query
        2. Compile source snippets into context
        3. Use LLM to synthesize research notes
        """
        with trace_span("researcher_run", {"query": state.request.query}) as span:
            # Step 1: Search
            logger.info("Researcher searching for: %s", state.request.query)
            sources = self._search.search(
                state.request.query,
                max_results=state.request.max_sources,
            )
            state.sources.extend(sources)

            # Step 2: Build context from sources
            source_context = "\n\n".join(
                f"**[{i+1}] {s.title}**\nURL: {s.url}\n{s.snippet}"
                for i, s in enumerate(sources)
            )

            user_prompt = (
                f"Research query: {state.request.query}\n"
                f"Target audience: {state.request.audience}\n\n"
                f"Search results ({len(sources)} sources):\n\n{source_context}\n\n"
                f"Please synthesize these into comprehensive research notes."
            )

            # Step 3: LLM synthesis
            response = self._llm.complete(_SYSTEM_PROMPT, user_prompt)
            state.research_notes = response.content

            logger.info(
                "Researcher completed: %d sources, %d chars of notes, %d tokens",
                len(sources), len(response.content),
                (response.input_tokens or 0) + (response.output_tokens or 0),
            )

            state.add_trace_event("researcher", {
                "sources_found": len(sources),
                "notes_length": len(response.content),
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "cost_usd": response.cost_usd,
            })
            state.agent_results.append(AgentResult(
                agent=AgentName.RESEARCHER,
                content=response.content,
                metadata={
                    "sources_found": len(sources),
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "cost_usd": response.cost_usd,
                },
            ))

            span["sources_found"] = len(sources)
            span["notes_length"] = len(response.content)

        return state
