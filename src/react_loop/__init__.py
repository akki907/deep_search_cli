"""A ReAct (reason + act) loop built explicitly with LangGraph."""

from react_loop.chat import ChatSession, Command, parse_command, run_chat
from react_loop.console import (
    get_console,
    print_error,
    print_trace,
    setup_logging,
)
from react_loop.graph import (
    DEFAULT_SYSTEM_PROMPT,
    build_react_graph,
    should_continue,
)
from react_loop.llm import ReActModel, ScriptedChatModel, build_llm
from react_loop.runner import ReActRunner, final_answer, run_react, trace
from react_loop.state import AgentState
from react_loop.streaming import (
    FinalEvent,
    ReActStreamRunner,
    TokenEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from react_loop.tools import (
    ALL_TOOLS,
    calculator,
    get_current_time,
    search_knowledge_base,
    web_search,
    wikipedia_search,
    wikipedia_summary,
)

__all__ = [
    "ALL_TOOLS",
    "DEFAULT_SYSTEM_PROMPT",
    "AgentState",
    "ChatSession",
    "Command",
    "FinalEvent",
    "ReActModel",
    "ReActRunner",
    "ReActStreamRunner",
    "ScriptedChatModel",
    "TokenEvent",
    "ToolCallEvent",
    "ToolResultEvent",
    "build_llm",
    "build_react_graph",
    "calculator",
    "final_answer",
    "get_console",
    "get_current_time",
    "parse_command",
    "print_error",
    "print_trace",
    "run_chat",
    "run_react",
    "search_knowledge_base",
    "setup_logging",
    "should_continue",
    "trace",
    "web_search",
    "wikipedia_search",
    "wikipedia_summary",
]

__version__ = "0.1.0"
