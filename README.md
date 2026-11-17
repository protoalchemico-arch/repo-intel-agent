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
```

## Setup & Installation

1. **Pull the local model in Ollama:**
   ```bash
   ollama pull qwen2.5:7b
   ```

2. **Clone and install dependencies:**
   ```bash
   git clone [https://github.com/protoalchemico-arch/repo-intel-agent.git](https://github.com/protoalchemico-arch/repo-intel-agent.git)
   cd repo-intel-agent
   python -m venv .venv
   source .venv/bin/activate
   pip install langchain-ollama langchain-core langgraph pydantic
   ```

## Usage

Run the auditor against any local repository directory:

```bash
python main.py /path/to/target-repo
```

## Sample Output (`audit_report.md`)

```markdown
# Repository Intelligence Audit: `repo-intel-agent`

## Summary
An autonomous, local-first codebase intelligence and security agent.

**Overall Severity:** `LOW`

## Files Inspected
- `main.py`

## Key Findings
- The `read_file` tool does not perform any validation or sanitization on the file content before returning it, which could expose sensitive information if an attacker were to manipulate the input path.
- There is no explicit handling of potential errors or exceptions in the tools, which could lead to unexpected behavior if an invalid file path is provided.
- The guardrails ensure that the agent reads files but do not perform a comprehensive security check on the content being read.

## Recommendations
- Implement input validation and sanitization in the `read_file` tool to ensure only valid file paths are processed.
- Add robust error handling in all tools to manage potential issues such as file not found or permission errors.
- Define and enforce security policies for the files being read, ensuring that only authorized content is processed.
```
