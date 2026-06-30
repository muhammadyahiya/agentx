# Research: Agentic-AI Scaffolders & Enterprise Features (2025–2026)

Why this exists: to ground `agentx-kit`'s feature set in what the leading agent
frameworks and scaffolders actually ship, and what enterprises demand to run
agents in production. Survey conducted June 2026 across PyPI, GitHub, and docs.

## The market has three layers
1. **Libraries** you import — pydantic-ai, smolagents, marvin/controlflow, griptape, autogen/ag2.
2. **CLI scaffolders / codegen** — CrewAI CLI, LangGraph CLI, AgentStack, create-llama, FastAgency (cookiecutter), agno (`ag`). **`agentx-kit` lives here.**
3. **Runtimes / platforms** — agno AgentOS, LangGraph Platform, CrewAI AMP, Griptape Cloud.

A good scaffolder (layer 2) should generate a project that already talks to layer 3.

## The dominant 2025–2026 trend
Convergence on **OpenTelemetry GenAI semantic conventions** (`gen_ai.*` spans:
model, token usage, finish reason, tool calls) as the telemetry standard, with
**LangSmith / Langfuse / Arize Phoenix / Logfire** as interchangeable backends.
Token usage is standardized; **cost is derived** from tokens (not yet a standard
span attribute). Every serious framework now emits OTel spans.

## Per-tool highlights
| Tool | Scaffolds? | Standout / enterprise notes |
|---|---|---|
| **CrewAI CLI** (`crewai create`) | Yes, **uv-based** | `agents.yaml`/`tasks.yaml` config-as-data; fn + LLM **guardrails** w/ retries; `output_pydantic`; Flows; telemetry opt-out `OTEL_SDK_DISABLED=true`; tracing/RBAC in paid AMP |
| **LangGraph CLI** (`langgraph new`) | Yes, **uv-based** | `langgraph.json` manifest (deps, graphs, env, store, checkpointer, auth, http); `dev`/`build`/`up`/`deploy`; checkpointing, **HITL `interrupt`**, streaming, LangSmith tracing, Docker first-class |
| **create-llama** | Yes (use-case wizard) | provider `--ask-models`; vector-store choice; structured outputs; **HITL** first-class; async/SSE; `llama_deploy.yml`; LlamaIndex-locked, no eval/CI/Docker |
| **AgentStack** | Yes, framework-agnostic | **incremental codegen** (`generate agent/task`, `tools add`); AgentOps observability + **cost/token tracking** by default; `agentstack.json` w/ `telemetry_opt_out`; no Docker/CI/deploy yet |
| **agno** (ex-phidata) | Yes (`ag`) | most batteries-included: OTel tracing+audit logs, **evals (4 dims)**, **guardrails (PII/jailbreak/moderation)**, HITL, **JWT RBAC + multi-tenant**, Docker, telemetry opt-out `AGNO_TELEMETRY=false` |
| **pydantic-ai** | No (library) | **structured outputs** (3 modes, retry-on-fail, streamed validation); **`FallbackModel`**; **`UsageLimits`** (token/cost budgets); **pydantic-evals** (LLMJudge over OTel); first-party Logfire/OTel; durable execution (Temporal/DBOS/Prefect) |
| **FastAgency** | Yes (cookiecutter) | best deploy story: Mesop/FastAPI/NATS, Docker+compose+devcontainer+nginx, auth, Fly.io/Azure/AWS scripts; no obs/eval/guardrails |
| **smolagents** (HF) | No | CodeAgent; **sandboxed exec** (E2B/Docker/Wasm); native OTel→Langfuse/Phoenix/MLflow; token tracking; `final_answer_checks` guardrails |
| **griptape** | template repo | **Off-Prompt** task memory; Rulesets (guardrails); **Eval Engine** (0–1 score + reason); structured output w/ re-prompt; OTel observability drivers |
| **autogen / ag2** | AutoGen Studio (GUI) | multi-agent chat; OTel (GenAI conventions, opt-out env); AutoGenBench eval CLI; Docker code exec; HITL UserProxyAgent |

**Reference layouts to study:** `langchain-ai/new-langgraph-project` (clean uv + CI + tests) and `wassim249/fastapi-langgraph-agent-production-ready-template` (full enterprise stack: FastAPI+LangGraph, Langfuse, Prometheus/Grafana, circular LLM fallback, tenacity retries, JWT, rate limiting, Docker Compose).

## Top 14 enterprise features (ranked) → agentx-kit status
| # | Feature | agentx-kit |
|---|---|---|
| 1 | **OTel GenAI tracing, pluggable backend** | ✅ `agentx.observability` (Langfuse/OTLP, opt-out) + generated `observability.py` |
| 2 | **Structured outputs (Pydantic) + retry** | ✅ `agentx.structured.with_structured_output` |
| 3 | **Retries / fallbacks / circuit-breakers** | ✅ `agentx.reliability` (provider fallback + retry) |
| 4 | **Cost & token budgets** | ✅ `agentx.reliability.UsageLimits` + usage callback |
| 5 | **Provider-agnostic config + `--ask-models`** | ✅ core factory + wizard (pre-existing) |
| 6 | **Guardrails (input/output)** | ✅ `agentx.guardrails` (PII/banned/length/schema) + generated `guardrails.py` |
| 7 | **HITL / approval on high-risk actions** | ➖ via LangGraph `interrupt` (documented); roadmap for generated wiring |
| 8 | **Async + streaming** | ✅ generated FastAPI `server.py` (SSE) |
| 9 | **Eval harness wired into CI** | ✅ generated `evals/` (LLM-as-judge) + CI eval job |
| 10 | **FastAPI serving + Docker + Compose** | ✅ generated `server.py`, `Dockerfile`, `docker-compose.yml` |
| 11 | **12-factor secrets/config (pydantic-settings)** | ✅ generated `settings.py` (`BaseSettings`, `SecretStr`) |
| 12 | **CI/CD (GitHub Actions: lint+type+test+eval)** | ✅ generated `.github/workflows/ci.yml` |
| 13 | **RBAC / JWT auth + audit logs** | ➖ roadmap (high effort; multi-tenant) |
| 14 | **Anonymous, opt-out telemetry** | ✅ `AGENTX_TELEMETRY=false` honored everywhere |

**Roadmap (honorable mentions):** durable execution (Temporal/DBOS), incremental
codegen (`agentx generate agent`), `copier update`-style template propagation,
sandboxed code execution, RBAC/multi-tenant, devcontainer.

## Generated-project skeleton we converge on
`uv`-managed; `.env.example` + `pydantic-settings`; config-as-data
(`prompts.json`, optional `agents.yaml`); an `agentx.json` manifest (à la
`langgraph.json`/`agentstack.json`); `observability.py` + `guardrails.py`;
optional `server.py` (FastAPI/SSE), `Dockerfile`+`docker-compose.yml`,
`.github/workflows/ci.yml`, and `evals/` runnable locally and in CI.

### Sources (selected)
PyPI/GitHub/docs for: crewai, langgraph-cli (`new-langgraph-project`),
create-llama, agentstack, agno, pydantic-ai (+pydantic-evals), fastagency,
smolagents, griptape, autogen/ag2; `wassim249/fastapi-langgraph-agent-production-ready-template`;
`GoogleCloudPlatform/agent-starter-pack`; OpenTelemetry GenAI semantic conventions.
