"""Builds the ReAct loop as an explicit LangGraph state graph.

The loop is three pieces:

1. ``agent``   - calls the model, which either answers or requests tools.
2. ``tools``   - executes the requested tools and appends ToolMessages.
3. routing     - sends control to ``tools`` while the model is asking for
   tools, otherwise ends the run.

That ``agent -> tools -> agent`` cycle is the ReAct loop: observe, reason,
act, repeat until the model stops asking for tools.
"""

from collections.abc import Sequence
from langchain_core.runnables import RunnableConfig
from typing import Any, Literal

from langchain_core.messages import SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from react_loop.llm import ReActModel
from react_loop.state import AgentState

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant.\n"
    "Work in small steps: call a tool when it gives you information you "
    "need, then use the tool output to decide the next step.\n"
    "When you can answer, reply with the final answer only."
)


def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
    """Route to ``tools`` while the model keeps requesting tool calls.

    The graph stops as soon as the model returns a message without tool
    calls, which is the model's way of saying it is done.
    """
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return "tools"
    return END


def build_react_graph(
    llm: ReActModel,
    tools: Sequence[Any],
    system_prompt: str | None = None,
    checkpointer: Any | None = None,
    name: str = "react_agent",
) -> Any:
    """Compile the ReAct graph.

    Args:
        llm: A model exposing ``bind_tools`` and ``invoke``.
        tools: Tools to expose. ``None`` builds a chat-only loop.
        system_prompt: Instruction placed before the messages.
        checkpointer: Optional LangGraph checkpointer for memory/threading.
        name: Compiled graph name.

    Returns:
        A compiled LangGraph graph.

    Note:
        LangGraph's ``recursion_limit`` (default 25) is the backstop that
        stops an agent from looping forever; ``invoke`` callers can raise it
        or lower it per call.

    """
    tools = list(tools)
    prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
    model = llm.bind_tools(tools) if tools else llm

    def call_model(state: AgentState, config: RunnableConfig) -> dict[str, list[Any]]:
        # Insert the system prompt fresh each turn so it is not duplicated
        # as the message history grows.
        messages = [SystemMessage(content=prompt), *state["messages"]]
        
        # Log the agent's thought process
        import logging
        session_id = config.get("configurable", {}).get("thread_id", "unknown")
        logger = logging.getLogger(f"react_loop.session.{session_id}")
        logger.info(f"Agent call with {len(messages)} messages")
        
        response = model.invoke(messages)
        logger.info(f"Agent response: {response.content} (Tool calls: {len(getattr(response, 'tool_calls', []))})")
        return {"messages": [response], "steps": 1}

    builder: StateGraph = StateGraph(AgentState)
    builder.add_node("agent", call_model)
    if tools:
        builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "agent")
    if tools:
        # While the model requests tools, loop back through the tools node.
        builder.add_conditional_edges(
            "agent", should_continue, {"tools": "tools", END: END}
        )
        builder.add_edge("tools", "agent")
    else:
        # No tools: a single model call is the whole run.
        builder.add_edge("agent", END)
    return builder.compile(
        checkpointer=checkpointer,
        name=name,
        interrupt_before=["tools"] if tools else None,
    )
