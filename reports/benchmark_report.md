# Benchmark Report

**Generated:** 2026-05-06 04:41:37 UTC
**Runs compared:** 2

## Summary Table

| Run | Latency (s) | Cost (USD) | Quality (/10) | Notes |
|---|---:|---:|---:|---|
| baseline | 0.113 | $0.000142 | 6.0 | 1 LLM call, 304 tokens |
| multi-agent | 0.909 | $0.001199 | 8.0 | 5 iterations, 9 agents, 4810 tokens, 5 sources |

## Comparative Analysis

### Latency
- Baseline: 0.113s
- Multi-agent: 0.909s
- **Ratio:** 8.0x slower

### Cost
- Baseline: $0.000142
- Multi-agent: $0.001199
- **Ratio:** 8.5x

### Quality
- Baseline: 6.0/10
- Multi-agent: 8.0/10
- **Improvement:** +2.0 points

## Key Observations

1. Multi-agent system is **8.0x** slower than baseline due to multiple agent calls and LLM invocations.
2. Cost increases by **8.5x** with multi-agent, reflecting additional LLM calls for research, analysis, writing, and review.
3. Quality improves by **2.0 points** thanks to specialized agent roles and iterative refinement.

## Trade-off Analysis

| Factor | Single-Agent (Baseline) | Multi-Agent | Winner |
|--------|------------------------|-------------|--------|
| Latency | 0.113s | 0.909s | Baseline |
| Cost | $0.000142 | $0.001199 | Baseline |
| Quality | 6.0/10 | 8.0/10 | Multi-Agent |
| Traceability | Limited | Full agent-level | Multi-Agent |
| Debugging | Harder | Easier (per-agent) | Multi-Agent |

## Recommendations

- Use **single-agent** for simple queries where latency and cost matter most.
- Use **multi-agent** for complex research tasks requiring multiple perspectives.
- Consider **hybrid approach**: route simple queries to baseline, complex to multi-agent.

---
*Report generated at 2026-05-06 04:41:37 UTC*
