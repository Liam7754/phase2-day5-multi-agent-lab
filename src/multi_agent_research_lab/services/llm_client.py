"""LLM client abstraction with mock implementation.

Production note: agents should depend on this interface instead of importing an SDK directly.
When no API key is available, MockLLMClient provides realistic responses for development/testing.
"""

import logging
import random
import time
from dataclasses import dataclass, field

from multi_agent_research_lab.core.config import get_settings

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
#  Cost estimation constants (per 1K tokens, approximate for gpt-4o-mini)
# --------------------------------------------------------------------------- #
INPUT_COST_PER_1K = 0.00015
OUTPUT_COST_PER_1K = 0.0006


@dataclass(frozen=True)
class LLMResponse:
    content: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None
    model: str = "mock"
    latency_ms: float = 0.0


# --------------------------------------------------------------------------- #
#  Mock knowledge base for realistic responses
# --------------------------------------------------------------------------- #
_MOCK_RESEARCH_NOTES = {
    "default": (
        "## Research Notes\n\n"
        "### Key Findings\n"
        "1. **Multi-agent systems** divide complex tasks among specialized agents, each with "
        "a clear role (researcher, analyst, writer). This separation of concerns improves "
        "output quality by 20-35% compared to monolithic prompts (Smith et al., 2024).\n\n"
        "2. **Graph-based orchestration** (e.g., LangGraph) enables conditional routing and "
        "iterative refinement loops. Supervisor patterns are the most common architecture, "
        "where a router agent delegates to worker agents.\n\n"
        "3. **Shared state** is critical for agent coordination. Pydantic models provide "
        "type-safe state management that prevents data corruption between agent handoffs.\n\n"
        "4. **Guardrails** such as max iterations, timeout enforcement, and output validation "
        "prevent runaway costs and hallucination cascades.\n\n"
        "### Sources Consulted\n"
        "- Anthropic: Building Effective Agents (2024)\n"
        "- OpenAI Agents SDK Documentation (2024)\n"
        "- LangGraph Concepts & Tutorials (2024)\n"
        "- Microsoft AutoGen Research Paper (2023)\n"
        "- CrewAI Framework Documentation (2024)\n"
    ),
    "graphrag": (
        "## Research Notes on GraphRAG\n\n"
        "### Key Findings\n"
        "1. **GraphRAG** (Graph Retrieval-Augmented Generation) combines knowledge graphs with "
        "LLM generation. Microsoft Research introduced this approach to handle complex queries "
        "that require multi-hop reasoning across documents.\n\n"
        "2. **Architecture**: Documents are first indexed into a knowledge graph using entity "
        "extraction and relationship mapping. At query time, relevant subgraphs are retrieved "
        "and used as context for the LLM.\n\n"
        "3. **Advantages over traditional RAG**: GraphRAG handles 'global' queries (summarizing "
        "themes across a corpus) significantly better, with up to 70% improvement in "
        "comprehensiveness scores.\n\n"
        "4. **Community detection**: The Leiden algorithm is used to create hierarchical "
        "community summaries, enabling multi-resolution querying.\n\n"
        "5. **Cost considerations**: Indexing is expensive (10-100x more tokens than chunk-based "
        "RAG), but query-time performance is comparable.\n\n"
        "### Sources Consulted\n"
        "- Microsoft Research: GraphRAG Paper (2024)\n"
        "- LangChain GraphRAG Integration Guide\n"
        "- Neo4j + LLM Knowledge Graph Tutorial\n"
        "- From Local to Global: GraphRAG Approach (Edge et al., 2024)\n"
    ),
}

_MOCK_ANALYSIS_NOTES = {
    "default": (
        "## Analysis\n\n"
        "### Strength of Evidence\n"
        "- **Strong**: Multi-agent architectures improve task decomposition and output quality. "
        "Multiple independent sources confirm 20-35% quality improvement.\n"
        "- **Moderate**: Supervisor/router patterns are industry-standard but newer patterns "
        "(hierarchical, debate-based) show promise in specific domains.\n"
        "- **Weak**: Cost estimates vary widely depending on model choice and task complexity.\n\n"
        "### Comparative Analysis\n"
        "| Aspect | Single-Agent | Multi-Agent |\n"
        "|--------|-------------|-------------|\n"
        "| Latency | Lower (1 LLM call) | Higher (3-5 calls) |\n"
        "| Quality | Good for simple tasks | Better for complex research |\n"
        "| Cost | ~$0.002/query | ~$0.008/query |\n"
        "| Traceability | Limited | Full agent-level trace |\n\n"
        "### Key Insight\n"
        "Multi-agent systems excel when tasks require diverse expertise or iterative "
        "refinement. For simple Q&A, single-agent is more cost-effective.\n"
    ),
    "graphrag": (
        "## Analysis of GraphRAG\n\n"
        "### Strength of Evidence\n"
        "- **Strong**: GraphRAG outperforms traditional RAG on global/thematic queries. "
        "Microsoft's evaluation shows 70% comprehensiveness improvement.\n"
        "- **Moderate**: Community detection via Leiden algorithm is effective but "
        "parameter-sensitive. Results depend on resolution settings.\n"
        "- **Weak**: Long-term maintenance costs of knowledge graphs are not well-studied.\n\n"
        "### Trade-off Analysis\n"
        "| Aspect | Traditional RAG | GraphRAG |\n"
        "|--------|----------------|----------|\n"
        "| Indexing Cost | Low | High (10-100x) |\n"
        "| Query Latency | Fast | Comparable |\n"
        "| Global Queries | Poor | Excellent |\n"
        "| Local Queries | Good | Good |\n"
        "| Maintenance | Simple | Complex |\n\n"
        "### Key Insight\n"
        "GraphRAG is best suited for knowledge-intensive domains where users need "
        "thematic summaries. For simple factual lookup, traditional RAG suffices.\n"
    ),
}

_MOCK_FINAL_ANSWERS = {
    "default": (
        "# Multi-Agent Research Systems: State of the Art\n\n"
        "## Introduction\n"
        "Multi-agent systems represent a paradigm shift in how we build LLM-powered "
        "applications. Rather than relying on a single monolithic prompt, these systems "
        "decompose complex tasks across specialized agents that collaborate through "
        "shared state.\n\n"
        "## Architecture\n"
        "The dominant architecture is the **Supervisor pattern**, where a routing agent "
        "delegates work to specialized workers:\n"
        "- **Researcher**: Gathers and filters sources\n"
        "- **Analyst**: Extracts insights and compares viewpoints\n"
        "- **Writer**: Synthesizes a coherent final response\n\n"
        "Graph-based orchestration frameworks like LangGraph enable conditional routing, "
        "iterative refinement, and robust error handling [1].\n\n"
        "## Benefits\n"
        "Studies show multi-agent approaches improve output quality by 20-35% on complex "
        "research tasks compared to single-agent baselines [2]. Key advantages include:\n"
        "- Better task decomposition and role specialization\n"
        "- Full traceability of which agent contributed what\n"
        "- Easier debugging and iterative improvement\n\n"
        "## Challenges\n"
        "Multi-agent systems introduce higher latency (3-5 LLM calls vs 1) and increased "
        "cost (~4x). Careful guardrails are needed: max iteration limits, timeout enforcement, "
        "and output validation prevent runaway costs [3].\n\n"
        "## Conclusion\n"
        "For complex research and analysis tasks, multi-agent systems offer meaningful "
        "quality improvements. For simple queries, single-agent remains more efficient.\n\n"
        "## References\n"
        "[1] LangGraph Documentation, 2024\n"
        "[2] Anthropic: Building Effective Agents, 2024\n"
        "[3] Microsoft AutoGen Research, 2023\n"
    ),
    "graphrag": (
        "# GraphRAG: From Local to Global Retrieval-Augmented Generation\n\n"
        "## Introduction\n"
        "GraphRAG is an advanced retrieval-augmented generation technique developed by "
        "Microsoft Research that combines knowledge graphs with LLM-based generation "
        "to handle complex, multi-hop queries.\n\n"
        "## How It Works\n"
        "1. **Indexing Phase**: Documents are processed to extract entities and relationships, "
        "forming a knowledge graph. Community detection (Leiden algorithm) creates hierarchical "
        "summaries at multiple resolutions.\n"
        "2. **Query Phase**: User queries are matched against graph communities. Relevant "
        "subgraphs and community summaries provide rich context for generation.\n\n"
        "## Key Advantages\n"
        "GraphRAG dramatically outperforms traditional RAG on 'global' queries that require "
        "synthesizing themes across an entire corpus [1]. Evaluations show up to 70% "
        "improvement in comprehensiveness scores.\n\n"
        "## Trade-offs\n"
        "The primary cost is indexing: GraphRAG uses 10-100x more tokens during the "
        "indexing phase compared to chunk-based RAG. However, query-time performance "
        "remains comparable [2].\n\n"
        "## Conclusion\n"
        "GraphRAG is best suited for knowledge-intensive domains requiring thematic "
        "analysis. For simple factual retrieval, traditional RAG remains more cost-effective.\n\n"
        "## References\n"
        "[1] Edge et al., 'From Local to Global: A GraphRAG Approach', 2024\n"
        "[2] Microsoft Research GraphRAG Documentation, 2024\n"
    ),
}

_MOCK_CRITIC_NOTES = (
    "## Critic Review\n\n"
    "### Fact-Check Results\n"
    "- Claims about quality improvements (20-35%): **Verified** - consistent with published benchmarks\n"
    "- Cost estimates: **Approximate** - actual costs depend on model and prompt length\n"
    "- Architecture descriptions: **Accurate** - matches current framework documentation\n\n"
    "### Citation Coverage\n"
    "- Total claims made: 8\n"
    "- Claims with citations: 6 (75%)\n"
    "- Unsupported claims: 2 (cost estimates, latency comparisons)\n\n"
    "### Recommendations\n"
    "- Add specific benchmark numbers for latency claims\n"
    "- Include date-stamps for rapidly evolving framework references\n"
    "- Overall quality: **7.5/10** - well-structured with minor gaps\n"
)

_MOCK_SUPERVISOR_DECISIONS = {
    "no_research": "researcher",
    "no_analysis": "analyst",
    "no_answer": "writer",
    "needs_review": "critic",
    "done": "done",
}

_MOCK_BASELINE_ANSWER = (
    "# Research Summary (Single-Agent Baseline)\n\n"
    "Multi-agent systems in AI divide complex tasks among specialized agents. "
    "The supervisor pattern routes tasks to researcher, analyst, and writer agents. "
    "Key benefits include better quality (20-35% improvement), full traceability, "
    "and easier debugging. Trade-offs include higher latency and cost (3-5x). "
    "GraphRAG enhances retrieval by combining knowledge graphs with LLM generation, "
    "excelling at global queries but requiring 10-100x more indexing tokens. "
    "For production systems, guardrails like max iterations and timeout enforcement "
    "are essential to prevent runaway costs.\n"
)


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 characters per token."""
    return max(1, len(text) // 4)


def _estimate_cost(input_tokens: int, output_tokens: int) -> float:
    return (input_tokens / 1000) * INPUT_COST_PER_1K + (output_tokens / 1000) * OUTPUT_COST_PER_1K


def _pick_topic(prompt: str) -> str:
    """Detect topic from prompt text for mock response selection."""
    lower = prompt.lower()
    if "graphrag" in lower or "graph rag" in lower:
        return "graphrag"
    return "default"


class MockLLMClient:
    """Mock LLM client that returns realistic responses without an API key."""

    def __init__(self) -> None:
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost_usd = 0.0
        self.call_count = 0

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        start = time.perf_counter()

        # Simulate small latency
        time.sleep(random.uniform(0.05, 0.15))

        topic = _pick_topic(user_prompt)
        content = self._generate_mock_response(system_prompt, user_prompt, topic)

        input_tokens = _estimate_tokens(system_prompt + user_prompt)
        output_tokens = _estimate_tokens(content)
        cost = _estimate_cost(input_tokens, output_tokens)
        latency_ms = (time.perf_counter() - start) * 1000

        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cost_usd += cost
        self.call_count += 1

        logger.info(
            "MockLLM call #%d: input=%d tokens, output=%d tokens, cost=$%.6f, latency=%.1fms",
            self.call_count, input_tokens, output_tokens, cost, latency_ms,
        )

        return LLMResponse(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            model="mock-gpt-4o-mini",
            latency_ms=latency_ms,
        )

    def _generate_mock_response(self, system_prompt: str, user_prompt: str, topic: str) -> str:
        sys_lower = system_prompt.lower()

        if "supervisor" in sys_lower or "router" in sys_lower or "routing" in sys_lower:
            return self._mock_supervisor_response(user_prompt)
        if "research" in sys_lower and "analy" not in sys_lower:
            return _MOCK_RESEARCH_NOTES.get(topic, _MOCK_RESEARCH_NOTES["default"])
        if "analy" in sys_lower:
            return _MOCK_ANALYSIS_NOTES.get(topic, _MOCK_ANALYSIS_NOTES["default"])
        if "writ" in sys_lower or "synthe" in sys_lower:
            return _MOCK_FINAL_ANSWERS.get(topic, _MOCK_FINAL_ANSWERS["default"])
        if "critic" in sys_lower or "fact" in sys_lower or "review" in sys_lower:
            return _MOCK_CRITIC_NOTES
        if "baseline" in sys_lower or "single" in sys_lower:
            return _MOCK_BASELINE_ANSWER

        # Fallback
        return _MOCK_FINAL_ANSWERS.get(topic, _MOCK_FINAL_ANSWERS["default"])

    def _mock_supervisor_response(self, user_prompt: str) -> str:
        lower = user_prompt.lower()
        if "research_notes: none" in lower or "research_notes: null" in lower or "no research" in lower:
            return "researcher"
        if "analysis_notes: none" in lower or "analysis_notes: null" in lower or "no analysis" in lower:
            return "analyst"
        if "final_answer: none" in lower or "final_answer: null" in lower or "no final" in lower:
            return "writer"
        if "final_answer:" in lower and "critic" not in lower:
            return "critic"
        return "done"


class LLMClient:
    """Provider-agnostic LLM client. Uses mock when no API key is configured."""

    def __init__(self) -> None:
        settings = get_settings()
        self._mock = MockLLMClient()
        self._use_mock = not settings.openai_api_key
        if self._use_mock:
            logger.info("No OPENAI_API_KEY found. Using MockLLMClient for all completions.")

    @property
    def mock(self) -> MockLLMClient:
        return self._mock

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Return a model completion. Falls back to mock when no API key is available."""
        if self._use_mock:
            return self._mock.complete(system_prompt, user_prompt)

        # Real OpenAI path (not used without API key)
        try:
            from openai import OpenAI
            settings = get_settings()
            client = OpenAI(api_key=settings.openai_api_key)
            response = client.chat.completions.create(
                model=settings.openai_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
            )
            msg = response.choices[0].message.content or ""
            usage = response.usage
            input_tokens = usage.prompt_tokens if usage else 0
            output_tokens = usage.completion_tokens if usage else 0
            cost = _estimate_cost(input_tokens, output_tokens)
            return LLMResponse(
                content=msg,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost,
                model=settings.openai_model,
            )
        except Exception as e:
            logger.warning("OpenAI call failed, falling back to mock: %s", e)
            return self._mock.complete(system_prompt, user_prompt)
