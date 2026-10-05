"""Streaming support for the ReAct loop.

:class:`ReActStreamRunner` wraps a compiled graph and turns its LangGraph
``astream`` output into simple, typed events, so a caller can render tokens as
they arrive without knowing LangGraph's payload shapes.

Events are yielded in this order for a typical run::

    ToolCallEvent    the model asked for a tool
    ToolResultEvent  the tool answered
    TokenEvent       a fragment of the final answer
    FinalEvent       the run is over
"""
from collections.abc import AsyncIterator, Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import uuid
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage

from react_loop.graph import build_react_graph
from react_loop.llm import ReActModel
from react_loop.tools import ALL_TOOLS


@dataclass
class ToolCallEvent:
    """The model requested a tool."""

    name: str
    args: dict[str, Any]
    id: str = ""


@dataclass
class ToolResultEvent:
    """A tool finished and returned output."""

    name: str
    content: str
    id: str = ""


@dataclass
class TokenEvent:
    """A fragment of the assistant's answer."""

    text: str


@dataclass
class FinalEvent:
    """The run finished. Carries the full result."""

    messages: list[AnyMessage]
    answer: str
    steps: int


StreamEvent = ToolCallEvent | ToolResultEvent | TokenEvent | FinalEvent


async def stream_events(
    events: AsyncIterator[StreamEvent],
    console: Any,
) -> None:
    """Render a stream of events to the console in real-time.

    Tool calls and results are printed as panels; tokens are streamed
    as a continuous block of text.
    """
    for event in events:
        if isinstance(event, ToolCallEvent):
            if hasattr(console, "print"):
                console.print(f"[bold yellow]tool call[/] {event.name}({event.args})")
            else:
                print(f"TOOL CALL: {event.name}({event.args})")
        elif isinstance(event, ToolResultEvent):
            if hasattr(console, "print"):
                console.print(f"[bold blue]tool result[/] {event.name}: {event.content[:200]}...")
            else:
                print(f"TOOL RESULT: {event.name}: {event.content[:200]}...")
        elif isinstance(event, TokenEvent):
            if hasattr(console, "print"):
                console.print(event.text, end="")
            else:
                print(event.text, end="")
        elif isinstance(event, FinalEvent):
            pass

StreamEvent = ToolCallEvent | ToolResultEvent | TokenEvent | FinalEvent

def _token_of(chunk: Any) -> str:
    """Pull the text fragment out of a streamed chunk."""
    if isinstance(chunk, str):
        return chunk
    content = getattr(chunk, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # Some providers send content as a list of blocks.
        return "".join(
            block.get("text", "") for block in content if isinstance(block, dict)
        )
    return ""


class ReActStreamRunner:
    """Runs the ReAct graph and yields typed stream events.

    Unlike :class:`~react_loop.runner.ReActRunner`, this never blocks until the
    whole run is done. Tokens are yielded as the model produces them.

    Args:
        llm: A model exposing ``bind_tools`` and ``invoke``.
        tools: Tools to expose. Defaults to the bundled demo tools.
        system_prompt: Optional system instruction.
        checkpointer: Optional checkpointer.
        name: Name given to the compiled graph.

    """

    def __init__(
        self,
        llm: ReActModel,
        tools: Sequence[Any] | None = None,
        system_prompt: str | None = None,
        checkpointer: Any | None = None,
        name: str = "react_stream_agent",
    ) -> None:
        """Compile the ReAct graph for streaming."""
        self.graph = build_react_graph(
            llm=llm,
            tools=ALL_TOOLS if tools is None else tools,
            system_prompt=system_prompt,
            checkpointer=checkpointer,
            name=name,
        )
        self.checkpointer = checkpointer

    def _config(self, recursion_limit: int, thread_id: str | None = None) -> dict[str, Any]:
        config: dict[str, Any] = {"recursion_limit": recursion_limit}
        if self.checkpointer is not None:
            config["configurable"] = {"thread_id": thread_id or uuid.uuid4().hex}
        return config

    async def astream(
        self,
        question: str,
        *,
        recursion_limit: int = 25,
        thread_id: str | None = None,
        approval_callback: Callable[[list[Any]], str | list[Any]] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Run one question, yielding events as they happen, with HITL approval.

        Args:
            question: The user message to start from.
            recursion_limit: Maximum agent/tool rounds.
            thread_id: Conversation to append to.
            approval_callback: A callback called before tool execution.
        """
        config = self._config(recursion_limit, thread_id)
        start = {"messages": [HumanMessage(content=question)]}
        messages: list[AnyMessage] = []
        steps = 0
        
        # Initial run to reach the first interrupt or end
        async for mode, payload in self.graph.astream(
            start, config=config, stream_mode=["messages", "updates"]
        ):
            if mode == "messages":
                chunk, meta = payload
                if meta.get("langgraph_node") == "agent":
                    text = _token_of(chunk)
                    if text: yield TokenEvent(text=text)
            elif mode == "updates":
                for node, update in payload.items():
                    if isinstance(update, dict):
                        steps += int(update.get("steps") or 0)
                        for event in _events_from_update(node, update.get("messages", [])):
                            yield event
                        messages.extend(update.get("messages", []))

        while True:
            snapshot = self.graph.get_state(config)
            if not snapshot.next or "tools" not in snapshot.next:
                break

            if approval_callback:
                last_msg = snapshot.values["messages"][-1]
                tool_calls = getattr(last_msg, "tool_calls", [])
                decision = await approval_callback(tool_calls)
                
                if decision == "approve":
                    pass
                elif decision == "cancel":
                    cancel_msgs = [ToolMessage(tool_call_id=tc["id"], content="Cancelled by user") for tc in tool_calls]
                    self.graph.update_state(config, {"messages": cancel_msgs})
                elif isinstance(decision, list):
                    new_msg = AIMessage(content=last_msg.content, tool_calls=decision)
                    self.graph.update_state(config, {"messages": [new_msg]})
                else:
                    self.graph.update_state(config, {"messages": [ToolMessage(tool_call_id=tool_calls[0]["id"], content="Cancelled")]})

            async for mode, payload in self.graph.astream(
                None, config=config, stream_mode=["messages", "updates"]
            ):
                if mode == "messages":
                    chunk, meta = payload
                    if meta.get("langgraph_node") == "agent":
                        text = _token_of(chunk)
                        if text: yield TokenEvent(text=text)
                elif mode == "updates":
                    for node, update in payload.items():
                        if isinstance(update, dict):
                            steps += int(update.get("steps") or 0)
                            for event in _events_from_update(node, update.get("messages", [])):
                                yield event
                            messages.extend(update.get("messages", []))

        answer = ""
        for message in reversed(messages):
            if isinstance(message, AIMessage) and not message.tool_calls:
                answer = str(message.content)
                break
        yield FinalEvent(messages=messages, answer=answer, steps=steps)

def _events_from_update(node: str, new_messages: list[AnyMessage]) -> Iterator[StreamEvent]:
    """Turn one node's new messages into stream events."""
    for message in new_messages:
        if isinstance(message, AIMessage):
            for call in message.tool_calls or []:
                yield ToolCallEvent(
                    name=call["name"], args=call.get("args", {}), id=call.get("id", "")
                )
        elif isinstance(message, ToolMessage):
            yield ToolResultEvent(
                name=message.name or "tool",
                content=str(message.content),
                id=message.tool_call_id,
            )


def collect_text(events: Sequence[StreamEvent]) -> str:
    """Join the token text from a list of events."""
    return "".join(e.text for e in events if isinstance(e, TokenEvent))


__all__ = [
    "FinalEvent",
    "ReActStreamRunner",
    "StreamEvent",
    "TokenEvent",
    "ToolCallEvent",
    "ToolResultEvent",
    "collect_text",
]
