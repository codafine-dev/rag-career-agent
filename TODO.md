# RAG Career Agent — TODO

Tracked roadmap for turning this scaffold into a portfolio-grade AI application.
Priorities are ordered against target-role requirements (applied AI engineer:
agents, evaluation, fine-tuning) — see conversation notes from 2026-08-26.

## Done

- [x] Domain models (`Ticket`, `Comment`) with Pydantic validation
- [x] Ingestion: JSON loader + normalizing parser (`TicketParseError` on invalid records)
- [x] MongoDB repositories: tickets, chunks, usage, admin tokens (idempotent upserts)
- [x] Field-aware chunker (summary / implementation / comments, overlap splitting)
- [x] OpenAI embeddings provider behind `EmbeddingsProvider` Protocol + pricing table
- [x] Cosine-similarity retriever with metadata filters, `min_score`, `max_per_ticket`
- [x] Daily cost budget cap for embedding spend
- [x] Unit tests for every module (mongomock + fake embeddings provider)

## Priority 1 — Agent orchestration (critical) — DONE

- [x] Tool definitions wrapping existing functions (`search_chunks`, `get_ticket`,
      `list_tickets`) for function/tool calling — plus `remember_fact` (`app/agent/tools.py`)
- [x] Orchestration loop: plan → call tool → observe → iterate, with max-step guard
      (`app/agent/orchestrator.py`, `AgentRunner`)
- [x] Short-term memory (conversation state within a session) (`app/agent/memory.py`)
- [x] Long-term memory (persisted across sessions) (`app/agent/memory.py` +
      `app/db/memory_repository.py`)
- [x] Error recovery: tool-handler exceptions are caught and fed back to the model
      as an observation (it can retry/adjust rather than crash the run); the planner
      call itself retries with backoff on transient failures before failing the run
- [x] State checkpointing: `AgentRunState` is saved after every step
      (`app/db/agent_run_repository.py`), so an interrupted run can be resumed by
      reloading the last checkpoint

Chat/embeddings providers: OpenAI (paid, already wired) and Gemini (free-tier via
AI Studio) both implemented behind the same `ChatProvider`/`EmbeddingsProvider`
protocols (`app/llm/chat.py`, `app/llm/embeddings.py`) — swappable without touching
the agent loop.

Test coverage: every module above has tests (`tests/test_orchestrator.py`,
`test_agent_tools.py`, `test_agent_memory.py`, `test_agent_models.py`,
`test_agent_run_repository.py`, `test_memory_repository.py`, `test_chat.py`,
`test_embeddings.py`). 86/86 passing, `ruff check` clean.

## Priority 2 — Evaluation & benchmarking (critical)

- [ ] Golden Q&A dataset derived from ticket resolutions/implementation notes
- [ ] Retrieval metrics: recall@k, MRR, nDCG
- [ ] Answer faithfulness / citation accuracy checks
- [ ] LLM-as-judge scoring, with documented failure modes and spot-check protocol
- [ ] Regression suite runnable in CI; track scores per commit/change

## Priority 3 — Grounded generation

- [ ] LLM answer synthesis over retrieved chunks (chat completions provider)
- [ ] Mandatory inline citations back to ticket IDs / chunk source fields
- [ ] Refuse to answer when retrieval confidence is insufficient
- [ ] Prompt/version management

## Priority 4 — Retrieval relevance upgrades

- [ ] Replace brute-force scan with ANN vector search (Mongo Atlas Vector Search
      preferred — no new infrastructure)
- [ ] Hybrid search: BM25 keyword + dense vectors fused via RRF
- [ ] Cross-encoder reranking of top candidates
- [ ] Query rewriting/expansion before retrieval
- [ ] Measure each change against the eval harness (Priority 2) — no blind tuning

## Priority 5 — Fine-tuning & synthetic data (critical)

- [ ] Synthetic query–chunk pair generation from tickets via LLM
- [ ] Fine-tune small embedding model (sentence-transformers +
      MultipleNegativesRankingLoss); compare vs. baseline on eval harness
- [ ] Document cost/benefit trade-off of fine-tuning vs. prompting/retrieval

## Priority 6 — Production & LLMOps

- [ ] FastAPI service exposing ask/index endpoints
- [ ] Retries, timeouts, graceful degradation on model/API failure
- [ ] Per-request latency + token/cost logging
- [ ] Dockerfile + docker-compose (app + MongoDB)
- [ ] Expand README: architecture, setup, usage
- [ ] Expand Privacy section: secrets handling, raw ticket content policy

## Nice-to-have

- [ ] Multilingual sample tickets + cross-language retrieval test queries
- [ ] Streaming responses
- [ ] CI pipeline (pytest + ruff) on push
