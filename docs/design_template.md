# Design Template

## Problem

Xay dung he thong nghien cuu tu dong co kha nang: nhan cau hoi nghien cuu tu nguoi dung, tim kiem va thu thap nguon tai lieu, phan tich va danh gia chat luong thong tin, tong hop thanh bai viet co cau truc voi trich dan, va kiem tra chat luong dau ra. He thong can trace duoc toan bo qua trinh xu ly.

## Why multi-agent?

Single-agent khong du vi:
1. **Qua tai vai tro**: Mot agent phai vua tim kiem, vua phan tich, vua viet - dan den chat luong thap hon 20-35% so voi chuyen mon hoa.
2. **Kho trace**: Khong biet phan nao cua output den tu dau, khong debug duoc khi sai.
3. **Khong co iterative refinement**: Single-agent chi chay 1 lan, khong co vong lap supervisor de kiem tra va bo sung.
4. **Khong co quality gate**: Khong co critic/reviewer de fact-check truoc khi tra ket qua.

## Agent roles

| Agent | Responsibility | Input | Output | Failure mode |
|---|---|---|---|---|
| Supervisor | Routing decision: chon agent tiep theo hoac dung lai | ResearchState (toan bo) | Route decision (researcher/analyst/writer/critic/done) | Fallback to deterministic routing khi LLM response khong hop le |
| Researcher | Tim kiem nguon, loc ket qua, ghi research notes | Query + max_sources | sources[], research_notes | Return empty sources, log warning |
| Analyst | Phan tich research notes, so sanh viewpoints, danh gia evidence | research_notes + sources | analysis_notes | Skip khi khong co research_notes, phan tich truc tiep tu sources |
| Writer | Tong hop final answer voi citations tu research + analysis | research_notes + analysis_notes + sources | final_answer | Viet tu bat ky thong tin nao co san |
| Critic | Fact-check, citation coverage, quality score | final_answer + research_notes + sources | Review appended to agent_results | Skip khi khong co final_answer |

## Shared state

| Field | Type | Purpose |
|---|---|---|
| `request` | ResearchQuery | Luu query goc, max_sources, audience |
| `iteration` | int | Dem so vong lap de enforce max_iterations |
| `route_history` | list[str] | Track agent nao da chay, thu tu nao |
| `sources` | list[SourceDocument] | Tat ca nguon tim duoc, chia se giua agents |
| `research_notes` | str | Output cua Researcher, input cho Analyst |
| `analysis_notes` | str | Output cua Analyst, input cho Writer |
| `final_answer` | str | Output cua Writer, input cho Critic |
| `agent_results` | list[AgentResult] | Lich su day du moi agent (content + metadata + tokens) |
| `trace` | list[dict] | Event log cho observability |
| `errors` | list[str] | Loi xay ra trong qua trinh chay |

## Routing policy

```
START -> Supervisor
Supervisor -> Researcher  (khi research_notes == None)
Supervisor -> Analyst     (khi analysis_notes == None)  
Supervisor -> Writer      (khi final_answer == None)
Supervisor -> Critic      (khi final_answer co nhung chua review)
Supervisor -> DONE        (khi tat ca hoan thanh hoac max_iterations)

Researcher -> Supervisor (luon luon)
Analyst    -> Supervisor (luon luon)
Writer     -> Supervisor (luon luon)
Critic     -> Supervisor (luon luon)
```

Stop conditions:
- Route == "done"
- iteration >= max_iterations (default 6)
- elapsed_time > timeout_seconds (default 60s)

## Guardrails

- **Max iterations**: 6 (configurable via MAX_ITERATIONS env var). Khi dat max, force writer neu chua co final_answer, hoac dung.
- **Timeout**: 60s (configurable via TIMEOUT_SECONDS). Force writer khi timeout.
- **Retry**: LLMClient co fallback tu OpenAI -> MockLLM khi API fail.
- **Fallback**: Supervisor co deterministic fallback routing khi LLM response khong hop le.
- **Validation**: Pydantic schemas cho tat ca input/output. ResearchQuery enforce min_length=5, max_sources 1-20.

## Benchmark plan

| Query | Metric | Expected Baseline | Expected Multi-Agent |
|---|---|---|---|
| "Research GraphRAG state-of-the-art and write a 500-word summary" | Latency | < 0.5s | 0.5-2s |
| | Cost | ~$0.0002 | ~$0.0012 |
| | Quality | 5-6/10 | 7-8/10 |
| | Citations | 0 | 3-5 |
| "Compare single-agent and multi-agent workflows" | Latency | < 0.5s | 0.5-2s |
| | Agent calls | 1 | 8-10 |
| "Summarize production guardrails for LLM agents" | Error rate | 0 | 0 |
| | Sources | 0 | 3-6 |

Metrics tracked per run:
- `latency_seconds`: Wall-clock time
- `estimated_cost_usd`: Sum of all LLM call costs
- `quality_score`: Heuristic 0-10 based on completeness, citations, errors
- `total_tokens`: Input + output tokens across all agents
- `citation_count`: Number of [N] references in final answer
- `error_count`: Number of errors during execution
