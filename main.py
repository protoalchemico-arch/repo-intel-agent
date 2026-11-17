import sys
from pathlib import Path
from typing import Annotated, TypedDict, List, Literal
from pydantic import BaseModel, Field

from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage, ToolMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

REPO_ROOT = Path(".").resolve()
IGNORE_DIRS = {".venv", "venv", ".git", "__pycache__", "node_modules", ".idea", ".vscode", "dist", "build"}
IGNORE_FILES = {"audit_report.md", ".env", ".env.local", "credentials.json", "secrets.yaml", "secrets.yml"}
CODE_EXTS = {".py", ".sh", ".js", ".ts", ".go", ".rs"}
DOC_CONFIG_EXTS = {".yaml", ".yml", ".toml", ".md", ".txt", ".json"}
ALLOWED_EXTS = CODE_EXTS | DOC_CONFIG_EXTS


def _is_ignored(p: Path) -> bool:
    return (
        p.is_symlink()
        or any(part in IGNORE_DIRS for part in p.parts)
        or p.name in IGNORE_FILES
        or p.name.startswith(".env")
    )


def _discover_files() -> List[str]:
    code_files, other_files = [], []
    for p in REPO_ROOT.rglob("*"):
        if _is_ignored(p):
            continue
        if p.is_file() and p.suffix in ALLOWED_EXTS:
            rel = str(p.relative_to(REPO_ROOT))
            if p.suffix in CODE_EXTS:
                code_files.append(rel)
            else:
                other_files.append(rel)
    return sorted(code_files) + sorted(other_files)


# --- 1. Agent Tools ---
@tool
def list_repo_files() -> str:
    """List up to 60 key source, config, and documentation files in the target repository."""
    files = _discover_files()
    if not files:
        return "No supported files found."
    if len(files) > 60:
        return "\n".join(files[:60]) + f"\n... ({len(files) - 60} more files; use search_code to locate specific modules)"
    return "\n".join(files)


@tool
def read_file(relative_path: str) -> str:
    """Read the contents of a specific file in the repository."""
    target = (REPO_ROOT / relative_path).resolve()
    if not target.is_relative_to(REPO_ROOT) or not target.is_file() or _is_ignored(target):
        return f"Error: File '{relative_path}' does not exist, is ignored, or is outside the repository."
    content = target.read_text(encoding="utf-8", errors="ignore")
    return content[:6000]


@tool
def search_code(keyword: str) -> str:
    """Search for a specific keyword or function name across all repository source files."""
    matches = []
    for p in REPO_ROOT.rglob("*"):
        if _is_ignored(p):
            continue
        if p.is_file() and p.suffix in ALLOWED_EXTS:
            for idx, line in enumerate(p.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
                if keyword.lower() in line.lower():
                    matches.append(f"{p.relative_to(REPO_ROOT)}:{idx}: {line.strip()}")
                    if len(matches) >= 25:
                        return "\n".join(matches)
    return "\n".join(matches) if matches else f"No matches found for '{keyword}'."


tools = [list_repo_files, read_file, search_code]


# --- 2. Structured Output Schema ---
class RepoAuditReport(BaseModel):
    project_summary: str = Field(description="Technical overview of what this repository does.")
    severity: Literal["LOW", "MEDIUM", "HIGH"] = Field(description="Overall code/security risk severity.")
    files_inspected: List[str] = Field(description="Exact relative paths of files read.")
    findings: List[str] = Field(description="Concrete bugs, security risks, or architectural observations citing actual functions/lines.")
    recommendations: List[str] = Field(description="Actionable engineering improvements.")


# --- 3. LangGraph State & Nodes ---
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    guardrail_retries: int
    final_report: dict


llm = ChatOllama(model="qwen2.5:7b", temperature=0)
llm_with_tools = llm.bind_tools(tools)


def _get_read_paths(messages: List[BaseMessage]) -> List[str]:
    paths = []
    for m in messages:
        if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
            for tc in m.tool_calls:
                if tc["name"] == "read_file" and "relative_path" in tc["args"]:
                    paths.append(tc["args"]["relative_path"])
    return paths


def investigator_node(state: AgentState) -> AgentState:
    print("[Investigator] Reasoning...")
    list_calls = sum(
        1 for m in state["messages"]
        if isinstance(m, ToolMessage) and m.name == "list_repo_files"
    )
    msgs = state["messages"]
    if list_calls >= 2:
        unread = [f for f in _discover_files() if f not in _get_read_paths(msgs)][:4]
        msgs = msgs + [
            HumanMessage(content=(
                f"Do NOT call `list_repo_files` again. Call `read_file` on one of these unread files now: {unread}"
            ))
        ]
    response = llm_with_tools.invoke(msgs)
    return {"messages": [response]}


def enforce_reads_node(state: AgentState) -> AgentState:
    retries = state.get("guardrail_retries", 0) + 1
    already_read = set(_get_read_paths(state["messages"]))
    unread = [f for f in _discover_files() if f not in already_read]

    if retries >= 2 or not unread:
        print("[Guardrail] Auto-reading remaining target files before finalizing...")
        injected = []
        for path in unread[:2]:
            content = (REPO_ROOT / path).read_text(encoding="utf-8", errors="ignore")[:6000]
            injected.append(HumanMessage(content=f"Contents of `{path}`:\n```\n{content}\n```"))
        return {"messages": injected, "guardrail_retries": retries}

    print(f"[Guardrail] Nudge {retries}/2 — directing agent to read: {unread[:2]}")
    return {
        "guardrail_retries": retries,
        "messages": [
            HumanMessage(content=(
                f"STOP. You still need to inspect actual files. Call `read_file` right now on: {unread[:2]}"
            ))
        ],
    }


def route_after_investigator(state: AgentState) -> str:
    last_msg = state["messages"][-1]
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        for tc in last_msg.tool_calls:
            print(f"  -> Tool: {tc['name']}({tc['args']})")
        return "tools"

    read_count = sum(
        1 for m in state["messages"]
        if isinstance(m, ToolMessage) and m.name == "read_file" and not str(m.content).startswith("Error:")
    )
    min_needed = min(2, len(_discover_files()))
    if read_count < min_needed and state.get("guardrail_retries", 0) < 2:
        return "enforce_reads"
    return "finalize"


def finalizer_node(state: AgentState) -> AgentState:
    print("[Finalizer] Generating structured Markdown audit...")
    structured_llm = llm.with_structured_output(RepoAuditReport)
    prompt = state["messages"] + [
        HumanMessage(content=(
            "Synthesize the final RepoAuditReport based STRICTLY on the file contents provided above. "
            "Do not invent file names or functions that were not read."
        ))
    ]
    report: RepoAuditReport = structured_llm.invoke(prompt)
    return {"final_report": report.model_dump()}


# --- 4. Compile Graph ---
workflow = StateGraph(AgentState)
workflow.add_node("investigator", investigator_node)
workflow.add_node("tools", ToolNode(tools))
workflow.add_node("enforce_reads", enforce_reads_node)
workflow.add_node("finalizer", finalizer_node)

workflow.set_entry_point("investigator")
workflow.add_conditional_edges(
    "investigator",
    route_after_investigator,
    {
        "tools": "tools",
        "enforce_reads": "enforce_reads",
        "finalize": "finalizer",
    },
)
workflow.add_edge("tools", "investigator")
workflow.add_edge("enforce_reads", "investigator")
workflow.add_edge("finalizer", END)

app = workflow.compile()


def save_markdown(report: dict, target_name: str, output_path: Path) -> None:
    md = f"""# Repository Intelligence Audit: `{target_name}`

## Summary
{report['project_summary']}

**Overall Severity:** `{report['severity']}`

## Files Inspected
{chr(10).join(f"- `{f}`" for f in report['files_inspected'])}

## Key Findings
{chr(10).join(f"- {item}" for item in report['findings'])}

## Recommendations
{chr(10).join(f"- {rec}" for rec in report['recommendations'])}
"""
    output_path.write_text(md, encoding="utf-8")


if __name__ == "__main__":
    target_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    REPO_ROOT = Path(target_dir).resolve()

    initial_messages = [
        SystemMessage(content=(
            "You are an autonomous codebase intelligence and security auditor.\n"
            "1. Call `list_repo_files` once.\n"
            "2. Call `read_file` on at least 2 real files from that list.\n"
            "3. Ground every finding strictly in the code you read."
        )),
        HumanMessage(content=f"Perform a technical and security audit of the repository: {REPO_ROOT.name}")
    ]

    result = app.invoke(
        {"messages": initial_messages, "guardrail_retries": 0, "final_report": {}},
        config={"recursion_limit": 25},
    )
    rep = result["final_report"]

    out_file = Path("audit_report.md")
    save_markdown(rep, REPO_ROOT.name, out_file)
    print(f"\nAudit complete! Report written to {out_file.resolve()}\n")
    print(out_file.read_text(encoding="utf-8"))
