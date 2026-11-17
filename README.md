# repo-intel-agent

An autonomous, local-first codebase intelligence and security auditing agent built with **LangGraph**, **LangChain**, **Pydantic**, and **Ollama**.

## Architecture

The agent uses a cyclic ReAct state graph (`StateGraph`) with a deterministic read-enforcement guardrail to prevent local LLM hallucinations before generating the final report:

```text
[Investigator (LLM)] ──(tool_calls)──▶ [ToolNode: list_repo_files / read_file / search_code]
        ▲   ▲                                              │
        │   └───────────────(observations)─────────────────┘
        │
        ├──(insufficient reads)──▶ [Guardrail: enforce_reads]
        │
  (verified reads)
        ▼
[Finalizer (Pydantic Schema)] ──▶ audit_report.md
