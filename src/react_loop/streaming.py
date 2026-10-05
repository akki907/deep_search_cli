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

import uuid
from collections.abc import AsyncIterator, Iterator, Sequence
from dataclasses import dataclass
from typing import Any

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
        self, question: str, *, recursion_limit: int = 25, thread_id: str | None = None
    ) -> AsyncIterator[StreamEvent]:
        """Run one question, yielding events as they happen.

        Args:
            question: The user message to start from.
            recursion_limit: Maximum agent/tool rounds.
            thread_id: Conversation to append to. Defaults to a fresh id, so
                each call is independent; pass a fixed id to keep memory
                across turns.

        Yields:
            :class:`ToolCallEvent`, :class:`ToolResultEvent`,
            :class:`TokenEvent`, and finally one :class:`FinalEvent`.

        """
        start = {"messages": [HumanMessage(content=question)]}
        messages: list[AnyMessage] = []
        steps = 0
        async for mode, payload in self.graph.astream(
            start,
            config=self._config(recursion_limit, thread_id),
            stream_mode=["messages", "updates"],
        ):
            if mode == "messages":
                chunk, meta = payload
                # The messages stream also carries ToolMessage output. Only the
                # agent node produces answer text, so filter on the node name.
                if meta.get("langgraph_node") != "agent":
                    continue
                text = _token_of(chunk)
                if text:
                    yield TokenEvent(text=text)
            elif mode == "updates":
                for node, update in payload.items():
                    if not isinstance(update, dict):
                        continue
                    # The agent node reports steps as an int via the reducer.
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
