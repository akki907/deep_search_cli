"""Convenience wrappers around the compiled ReAct graph."""

import uuid
from collections.abc import Sequence
from typing import Any

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage

from react_loop.graph import build_react_graph
from react_loop.llm import ReActModel, build_llm
from react_loop.tools import ALL_TOOLS


class ReActRunner:
    """Runs a compiled ReAct graph and reports the final answer.

    Args:
        llm: A model exposing ``bind_tools`` and ``invoke``.
        tools: Tools to expose. Defaults to the bundled demo tools.
        system_prompt: Optional system instruction.
        checkpointer: Optional checkpointer. When given, :meth:`run` uses a
            fresh ``thread_id`` per call unless one is passed.
        name: Name given to the compiled graph.

    """

    def __init__(
        self,
        llm: ReActModel,
        tools: Sequence[Any] | None = None,
        system_prompt: str | None = None,
        checkpointer: Any | None = None,
        name: str = "react_agent",
    ) -> None:
        """Compile the ReAct graph for the given model and tools."""
        self.graph = build_react_graph(
            llm=llm,
            tools=ALL_TOOLS if tools is None else tools,
            system_prompt=system_prompt,
            checkpointer=checkpointer,
            name=name,
        )
        self.checkpointer = checkpointer

    def run(
        self,
        question: str,
        *,
        recursion_limit: int = 25,
        thread_id: str | None = None,
    ) -> dict[str, Any]:
        """Run one question end to end and return the final graph state.

        Args:
            question: The user message to start from.
            recursion_limit: Maximum agent/tool rounds. LangGraph raises
                ``GraphRecursionError`` if the agent keeps requesting tools.
            thread_id: Conversation to append to. Defaults to a fresh id, so
                each call is independent; pass a fixed id to keep memory
                across turns.

        Returns:
            The final state, with ``messages`` and the ``steps`` count.

        """
        config: dict[str, Any] = {"recursion_limit": recursion_limit}
        if self.checkpointer is not None:
            config["configurable"] = {"thread_id": thread_id or uuid.uuid4().hex}
        start = {"messages": [HumanMessage(content=question)]}
        return self.graph.invoke(start, config)


def final_answer(messages: list[AnyMessage]) -> str:
    """Return the text of the last assistant message, or an empty string."""
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            return str(message.content)
    return ""


def trace(messages: list[AnyMessage]) -> list[str]:
    """Render the loop as readable lines, one per message.

    Useful for seeing the reason/act/observe cycle in a terminal.
    """
    lines: list[str] = []
    for message in messages:
        if isinstance(message, HumanMessage):
            lines.append(f"user      > {message.content}")
        elif isinstance(message, AIMessage):
            if message.tool_calls:
                for call in message.tool_calls:
                    lines.append(f"assistant> call {call['name']}({call['args']})")
            else:
                lines.append(f"assistant> {message.content}")
        elif isinstance(message, ToolMessage):
            text = str(message.content).replace(chr(10), " ")
            lines.append(f"tool     < {message.name}: {text[:200]}")
    return lines


def run_react(
    question: str,
    llm: ReActModel | None = None,
    tools: Sequence[Any] | None = None,
    system_prompt: str | None = None,
    *,
    recursion_limit: int = 25,
    verbose: bool = True,
) -> dict[str, Any]:
    """Build a runner, answer one question, and print the trace.

    Args:
        question: The user message to start from.
        llm: Model to use. Built from the environment when omitted.
        tools: Tools to expose. Defaults to the bundled demo tools.
        system_prompt: Optional system instruction.
        recursion_limit: Maximum agent/tool rounds.
        verbose: Print the trace as it is produced.

    Returns:
        The final graph state.

    """
    runner = ReActRunner(
        llm or build_llm(), tools=tools, system_prompt=system_prompt
    )
    result = runner.run(question, recursion_limit=recursion_limit)
    if verbose:
        for line in trace(result["messages"]):
            print(line)
    return result
