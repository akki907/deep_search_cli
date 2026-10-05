"""An interactive chat session for the ReAct loop.

The one-shot CLI answers a single question per process, so every run starts
from an empty history. :class:`ChatSession` instead holds one conversation
alive across turns: it keeps a stable ``thread_id`` against the LangGraph
checkpointer, so the model sees what was said earlier.

The read-eval-print loop is kept free of terminal handling. It takes a
``read_line`` callable and a ``turn`` coroutine, so the same code serves the
CLI, the tests, and any other front end.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any

from langchain_core.messages import AnyMessage

from react_loop.console import print_error, print_plain_trace
from react_loop.runner import final_answer
from react_loop.streaming import StreamEvent

if TYPE_CHECKING:
    from react_loop.runner import ReActRunner
    from react_loop.streaming import ReActStreamRunner

#: Shown before every question, keeping the user's own lines visually separate
#: from the panels the agent prints.
PROMPT = "you > "

#: Commands understood at the prompt. Kept in one place so that /help and the
#: dispatch in :func:`run_chat` cannot drift apart.
COMMANDS: dict[str, str] = {
    "/help": "Show these commands.",
    "/exit": "Leave the session. Ctrl-D does the same.",
    "/quit": "Alias for /exit.",
    "/clear": "Forget the conversation and start a new thread.",
    "/trace": "Replay the conversation so far.",
    "/research": "Conduct an exhaustive deep research on the provided topic.",
}

EXIT_COMMANDS = frozenset({"/exit", "/quit"})

HELP_TEXT = "\n".join([f"{name:<8} {text}" for name, text in COMMANDS.items()])


@dataclass(frozen=True)
class Command:
    """A slash command typed at the prompt."""

    name: str
    arg: str = ""


def parse_command(line: str) -> Command | None:
    """Return the command in ``line``, or ``None`` when it is ordinary text.

    A line that merely starts with a slash still counts as a command, so a
    typo such as ``/cleer`` is reported instead of being sent to the model.
    """
    stripped = line.strip()
    if not stripped.startswith("/"):
        return None
    name, _, arg = stripped.partition(" ")
    return Command(name=name.lower(), arg=arg.strip())


def banner(provider: str, model: str | None) -> str:
    """Return the header shown when the session starts."""
    return "\n".join(
        [
            f"react-loop   provider: {provider}   model: {model or 'provider default'}",
            "Type /help for commands, /exit to quit.",
            "",
        ]
    )


class ChatSession:
    """One conversation, held in a single LangGraph thread.

    Args:
        runner: A :class:`~react_loop.runner.ReActRunner` or
            :class:`~react_loop.streaming.ReActStreamRunner`. History is only
            kept when it was built with a ``checkpointer``.
        recursion_limit: Step budget for each turn.
        thread_id: Conversation id. A random one is used when omitted.

    """

    def __init__(
        self,
        runner: ReActRunner | ReActStreamRunner,
        *,
        recursion_limit: int = 25,
        thread_id: str | None = None,
    ) -> None:
        """Start a session on a fresh thread."""
        self.runner = runner
        self.recursion_limit = recursion_limit
        self._thread_id = thread_id or uuid.uuid4().hex

    @property
    def thread_id(self) -> str:
        """The conversation this session writes to."""
        return self._thread_id

    @property
    def remembers(self) -> bool:
        """Whether the runner is able to store history at all."""
        return getattr(self.runner, "checkpointer", None) is not None

    def reset(self) -> None:
        """Start a new thread, dropping every earlier turn.

        The old thread is simply abandoned, which is cheaper and less error
        prone than deleting entries from the checkpointer.
        """
        self._thread_id = uuid.uuid4().hex

    def messages(self) -> list[AnyMessage]:
        """Return every message in the current thread, oldest first."""
        if not self.remembers:
            return []
        state = self.runner.graph.get_state({"configurable": {"thread_id": self._thread_id}})
        return list((state.values or {}).get("messages", []))

    def ask(self, question: str) -> str:
        """Run one turn to completion and return the final answer."""
        result = self.runner.run(
            question, recursion_limit=self.recursion_limit, thread_id=self._thread_id
        )
        return final_answer(result["messages"])

    async def aask(
        self,
        question: str,
        approval_callback: Callable[[list[Any]], str | list[Any]] | None = None,
    ) -> str:
        """Run one turn with optional HITL approval and return the final answer."""
        result = await self.runner.run_async(
            question,
            config={"configurable": {"thread_id": self._thread_id}},
            approval_callback=approval_callback,
            recursion_limit=self.recursion_limit,
        )
        return final_answer(result["messages"])
    async def astream(
        self,
        question: str,
        approval_callback: Callable[[list[Any]], str | list[Any]] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Run one turn, yielding stream events as they happen."""
        events = self.runner.astream(
            question,
            recursion_limit=self.recursion_limit,
            thread_id=self._thread_id,
            approval_callback=approval_callback,
        )
        async for event in events:
            yield event


async def run_chat(
    session: ChatSession,
    *,
    read_line: Callable[[str], Any],
    turn: Callable[[ChatSession, str], Any],
    write: Callable[[str], None] = print,
    show_trace: Callable[[list[AnyMessage]], None] | None = None,
    on_error: Callable[[BaseException], None] = print_error,
) -> int:
    """Read questions until ``/exit`` or end of input.

    Args:
        session: The conversation to drive.
        read_line: Called with the prompt, returns one line of input, and
            raises ``EOFError`` at end of input.
        turn: Awaitable that runs one question and renders it. A failure
            here is reported and the session continues.
        write: Prints session-level text such as help.
        show_trace: Renders the conversation history for ``/trace``.
        on_error: Reports a turn that failed.

    Returns:
        A process exit code, always 0.

    """
    render_trace = show_trace or partial(print_plain_trace)
    while True:
        try:
            line = await read_line(PROMPT)
        except (EOFError, KeyboardInterrupt):
            # Ctrl-D and Ctrl-C both mean "we are done", not "something broke".
            write("")
            return 0

        text = line.strip()
        if not text:
            continue

        command = parse_command(text)
        if command is not None:
            if command.name in EXIT_COMMANDS:
                return 0
            if command.name == "/help":
                write(HELP_TEXT)
                continue
            if command.name == "/clear":
                session.reset()
                write("Conversation cleared.")
                continue
            if command.name == "/trace":
                render_trace(session.messages())
                continue
            if command.name == "/research":
                if not command.arg:
                    write("Please provide a topic: /research <topic>")
                    continue
                
                write(f"🔍 Deep researching: {command.arg}...")
                research_prompt = (
                    "You are a professional deep research agent. Your goal is to provide "
                    "an exhaustive, detailed report on the topic. Do not stop at the first "
                    "satisfactory answer. Search multiple sources, dig into details, "
                    "and synthesize a comprehensive final answer."
                )
                text = f"[RESEARCH MODE]: {research_prompt}\n\nTopic: {command.arg}"
                # Do NOT continue; let it fall through to the turn() call
            else:
                write(f"Unknown command {command.name}. Type /help for the list.")
                continue
        try:
            await turn(session, text)
        except Exception as exc:
            on_error(exc)
