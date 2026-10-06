"""Tests for the terminal rendering. All of these run offline."""

import asyncio

from langchain_core.messages import AIMessage
from rich.console import Console

from react_loop.console import stream_events
from react_loop.llm import ScriptedChatModel
from react_loop.streaming import FinalEvent, ReActStreamRunner, TokenEvent
from react_loop.tools import calculator


class FakeStatus:
    """Stands in for the Rich spinner and notes the order of events."""

    def __init__(self, journal: list[str]) -> None:
        self.journal = journal

    def start(self) -> None:
        self.journal.append("spinner start")

    def stop(self) -> None:
        self.journal.append("spinner stop")


class FakeConsole:
    """Records what the renderer prints, and when, relative to the spinner."""

    is_terminal = True
    no_color = False

    def __init__(self) -> None:
        self.journal: list[str] = []

    def status(self, *_args, **_kwargs) -> FakeStatus:
        return FakeStatus(self.journal)

    def print(self, text: str = "", **_kwargs) -> None:
        self.journal.append(f"print {text}" if text else "print ")


def render(console: FakeConsole) -> None:
    """Stream a canned answer that calls no tools at all."""
    runner = ReActStreamRunner(
        ScriptedChatModel(script=[AIMessage(content="the answer is 42")]), tools=[calculator]
    )

    async def go() -> str:
        return await stream_events(runner.astream("q"), console)

    asyncio.run(go())


def test_spinner_is_settled_before_the_answer_is_printed():
    """A turn with no tool calls must still take the spinner down.

    Nothing else stops it on such a turn, so if it is still live when the
    tokens print, its redraw overwrites the answer and the user sees nothing.
    """
    console = FakeConsole()
    render(console)

    first_print = next(i for i, line in enumerate(console.journal) if line.startswith("print "))
    settled = [i for i, line in enumerate(console.journal) if line == "spinner stop"]

    assert settled, "the spinner was never stopped"
    assert min(settled) < first_print


def test_stream_renders_the_completed_answer_as_markdown():
    console = Console(record=True, force_terminal=False, width=80)

    async def events():
        yield TokenEvent(text="# Findings\n\n**SQLite** is embedded.")
        yield FinalEvent(messages=[], answer="# Findings\n\n**SQLite** is embedded.", steps=1)

    asyncio.run(stream_events(events(), console))

    output = console.export_text()
    assert "Findings" in output
    assert "SQLite is embedded." in output
    assert "**SQLite**" not in output
