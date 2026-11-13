# repo-intel-agent

An autonomous, local-first codebase intelligence and security auditing agent built with **LangGraph**, **LangChain**, **Pydantic**, and **Ollama**.

## Architecture

The agent uses a cyclic ReAct state graph (`StateGraph`) powered by local LLM tool calling:

```text
[Investigator (LLM)] ──(tool_calls)──> [ToolNode: list_repo_files / read_file / search_code]
        ▲                                              │
        └────────────────(observations)────────────────┘
        │
  (no tool calls)
        ▼
[Finalizer (Pydantic Schema)] ──> audit_report.md
