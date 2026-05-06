"""Search client abstraction with mock implementation.

When no Tavily API key is available, MockSearchClient returns realistic mock results.
"""

import logging
import random
import time

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import SourceDocument

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
#  Mock search database
# --------------------------------------------------------------------------- #
_MOCK_SOURCES: dict[str, list[dict[str, str]]] = {
    "default": [
        {
            "title": "Building Effective Agents - Anthropic",
            "url": "https://www.anthropic.com/engineering/building-effective-agents",
            "snippet": "Multi-agent systems decompose complex tasks into specialized roles. "
            "The supervisor pattern delegates research, analysis, and writing to dedicated agents, "
            "improving output quality by 20-35% over monolithic approaches.",
        },
        {
            "title": "OpenAI Agents SDK - Orchestration Guide",
            "url": "https://developers.openai.com/docs/guides/agents",
            "snippet": "Agent handoffs enable seamless task delegation between specialized agents. "
            "The SDK supports tool calling, guardrails, and tracing for production deployments.",
        },
        {
            "title": "LangGraph Concepts - LangChain",
            "url": "https://langchain-ai.github.io/langgraph/concepts/",
            "snippet": "LangGraph provides a framework for building stateful, multi-actor applications. "
            "Conditional edges enable dynamic routing based on agent output and shared state.",
        },
        {
            "title": "Microsoft AutoGen: Multi-Agent Conversations",
            "url": "https://arxiv.org/abs/2308.08155",
            "snippet": "AutoGen enables next-gen LLM applications via multi-agent conversation framework. "
            "Agents can be customized with different roles, LLMs, tools, and human involvement.",
        },
        {
            "title": "CrewAI - Framework for AI Agent Teams",
            "url": "https://docs.crewai.com/",
            "snippet": "CrewAI orchestrates role-playing AI agents working together on complex tasks. "
            "Supports hierarchical and sequential process flows with memory and tool sharing.",
        },
        {
            "title": "Multi-Agent Systems for Production LLM Applications",
            "url": "https://arxiv.org/abs/2402.01680",
            "snippet": "Production multi-agent systems require careful attention to cost management, "
            "latency budgets, error handling, and observability. Guardrails prevent cascading failures.",
        },
    ],
    "graphrag": [
        {
            "title": "From Local to Global: A GraphRAG Approach - Microsoft Research",
            "url": "https://arxiv.org/abs/2404.16130",
            "snippet": "GraphRAG uses knowledge graph generation and community summaries to answer "
            "global queries over entire text corpora, outperforming traditional RAG by 70% on comprehensiveness.",
        },
        {
            "title": "GraphRAG: Unlocking LLM Discovery on Narrative Private Data",
            "url": "https://www.microsoft.com/en-us/research/blog/graphrag/",
            "snippet": "Microsoft's GraphRAG approach combines entity extraction, relationship mapping, "
            "and community detection to build hierarchical knowledge graphs from unstructured text.",
        },
        {
            "title": "LangChain GraphRAG Integration",
            "url": "https://python.langchain.com/docs/use_cases/graph/",
            "snippet": "LangChain provides built-in support for knowledge graph construction and querying. "
            "Integration with Neo4j, Amazon Neptune, and other graph databases is supported.",
        },
        {
            "title": "Knowledge Graphs for RAG - Neo4j Developer Guide",
            "url": "https://neo4j.com/developer/rag/",
            "snippet": "Neo4j enables graph-powered RAG pipelines. Entity extraction plus relationship "
            "modeling creates rich context for LLM generation beyond simple vector similarity search.",
        },
        {
            "title": "Evaluation of GraphRAG vs Traditional RAG",
            "url": "https://arxiv.org/abs/2405.12345",
            "snippet": "Comparative evaluation shows GraphRAG excels at global/thematic queries but "
            "traditional RAG is more cost-effective for simple factual lookups. Indexing costs are 10-100x higher.",
        },
    ],
    "customer support": [
        {
            "title": "Multi-Agent Customer Support Systems",
            "url": "https://example.com/customer-support-agents",
            "snippet": "Multi-agent customer support uses specialized agents for intent classification, "
            "knowledge retrieval, response generation, and quality assurance. Reduces resolution time by 40%.",
        },
        {
            "title": "Single vs Multi-Agent for Customer Service",
            "url": "https://example.com/single-vs-multi",
            "snippet": "Single-agent handles simple queries well but struggles with complex multi-step "
            "issues. Multi-agent architectures improve first-contact resolution from 65% to 85%.",
        },
        {
            "title": "Production Guardrails for Support Agents",
            "url": "https://example.com/guardrails",
            "snippet": "Essential guardrails include: PII detection, sentiment monitoring, escalation triggers, "
            "response length limits, and hallucination detection for customer-facing agents.",
        },
    ],
    "guardrails": [
        {
            "title": "Production Guardrails for LLM Agents - Best Practices",
            "url": "https://example.com/llm-guardrails",
            "snippet": "Key guardrails: max iterations (prevent infinite loops), timeout enforcement, "
            "output validation (schema checking), content filtering, and cost budgets per request.",
        },
        {
            "title": "NeMo Guardrails - NVIDIA",
            "url": "https://github.com/NVIDIA/NeMo-Guardrails",
            "snippet": "NeMo Guardrails provides programmable safety rails for LLM applications. "
            "Supports topical rails, safety rails, and custom action chains.",
        },
        {
            "title": "Guardrails AI Framework",
            "url": "https://www.guardrailsai.com/",
            "snippet": "Guardrails AI enables output validation, structured generation, and retry logic "
            "for LLM outputs. Integrates with OpenAI, Anthropic, and local models.",
        },
    ],
}


def _pick_topic(query: str) -> str:
    lower = query.lower()
    if "graphrag" in lower or "graph rag" in lower:
        return "graphrag"
    if "customer" in lower and "support" in lower:
        return "customer support"
    if "guardrail" in lower:
        return "guardrails"
    return "default"


class MockSearchClient:
    """Returns realistic mock search results without an API key."""

    def __init__(self) -> None:
        self.total_searches = 0

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        time.sleep(random.uniform(0.02, 0.08))  # Simulate latency
        self.total_searches += 1

        topic = _pick_topic(query)
        raw = _MOCK_SOURCES.get(topic, _MOCK_SOURCES["default"])
        results = [
            SourceDocument(
                title=s["title"],
                url=s["url"],
                snippet=s["snippet"],
                metadata={"source": "mock", "topic": topic},
            )
            for s in raw[:max_results]
        ]

        logger.info("MockSearch #%d: query=%r, topic=%s, results=%d", self.total_searches, query, topic, len(results))
        return results


class SearchClient:
    """Provider-agnostic search client. Uses mock when no Tavily key is configured."""

    def __init__(self) -> None:
        settings = get_settings()
        self._mock = MockSearchClient()
        self._use_mock = not settings.tavily_api_key
        if self._use_mock:
            logger.info("No TAVILY_API_KEY found. Using MockSearchClient.")

    @property
    def mock(self) -> MockSearchClient:
        return self._mock

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        if self._use_mock:
            return self._mock.search(query, max_results)

        try:
            from tavily import TavilyClient
            settings = get_settings()
            client = TavilyClient(api_key=settings.tavily_api_key)
            response = client.search(query=query, max_results=max_results)
            return [
                SourceDocument(
                    title=r.get("title", ""),
                    url=r.get("url"),
                    snippet=r.get("content", ""),
                    metadata={"source": "tavily"},
                )
                for r in response.get("results", [])
            ]
        except Exception as e:
            logger.warning("Tavily search failed, falling back to mock: %s", e)
            return self._mock.search(query, max_results)
