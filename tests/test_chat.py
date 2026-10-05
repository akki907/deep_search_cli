"""Tests for the interactive chat session. All of these run offline."""

import asyncio
import io
import sys

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from react_loop.chat import COMMANDS, ChatSession, parse_command, run_chat
from react_loop.llm import ScriptedChatModel
from react_loop.runner import ReActRunner

#: What record() hands back: what the session wrote, what /trace showed, exit code.
SessionLog = tuple[list[str], list[list[AnyMessage]], int]


def echo_history(messages: list[AnyMessage]) -> AIMessage:
    """Answer with how many questions the model has been asked so far."""
    humans = [m.content for m in messages if isinstance(m, HumanMessage)]
    return AIMessage(content=f"I have heard {len(humans)} questions")


def make_session(**kwargs) -> ChatSession:
    """Build a session over a model that reports how much it remembers."""
    runner = ReActRunner(ScriptedChatModel(script=echo_history), checkpointer=InMemorySaver())
    return ChatSession(runner, **kwargs)


def feed(lines: list[str]):
    """Return a read_line that yields ``lines`` and then raises EOFError."""
    remaining = iter(lines)

    def read_line(_prompt: str) -> str:
        try:
            return next(remaining)
        except StopIteration:
            raise EOFError from None

    return read_line


def record(session: ChatSession, lines: list[str], turn=None) -> SessionLog:
    """Drive one session, returning what it wrote, /trace showed, and the code."""
    written: list[str] = []
    traced: list[list[AnyMessage]] = []

    async def ask(session_: ChatSession, question: str) -> None:
        written.append(session_.ask(question))

    code = asyncio.run(
        run_chat(
            session,
            read_line=feed(lines),
            turn=turn or ask,
            write=written.append,
            show_trace=lambda messages: traced.append(messages),
            on_error=lambda exc: written.append(f"error: {exc}"),
        )
    )
    return written, traced, code


def test_session_remembers_earlier_turns():
    """The point of a session: the model sees the previous turns."""
    session = make_session()

    assert session.ask("first question") == "I have heard 1 questions"
    assert session.ask("second question") == "I have heard 2 questions"


def test_clear_starts_a_new_thread():
    """/clear must drop the history, not just hide it."""
    session = make_session()
    session.ask("first question")
    before = session.thread_id

    session.reset()

    assert session.thread_id != before
    assert session.ask("after clearing") == "I have heard 1 questions"


def test_session_history_lists_every_turn():
    session = make_session()
    session.ask("first question")
    session.ask("second question")

    contents = [str(m.content) for m in session.messages()]

    assert contents == [
        "first question",
        "I have heard 1 questions",
        "second question",
        "I have heard 2 questions",
    ]


def test_chat_stops_at_end_of_input():
    """Ctrl-D leaves the session cleanly rather than raising."""
    written, _, code = record(make_session(), ["a question"])

    assert code == 0
    # The trailing blank is the newline left behind when input ends.
    assert written == ["I have heard 1 questions", ""]


def test_exit_command_ends_the_session():
    """Nothing after /exit should reach the model."""
    written, _, code = record(make_session(), ["/exit", "never asked"])

    assert code == 0
    assert written == []


def test_blank_lines_are_ignored():
    written, _, code = record(make_session(), ["", "   ", "a question", ""])

    assert code == 0
    assert written == ["I have heard 1 questions", ""]


def test_unknown_command_is_reported_not_asked():
    """/cleer is a typo, not a question for the model."""
    written, _, code = record(make_session(), ["/cleer", "/exit"])

    assert code == 0
    assert written == ["Unknown command /cleer. Type /help for the list."]


def test_help_lists_every_command():
    written, _, code = record(make_session(), ["/help", "/exit"])

    assert code == 0
    assert len(written) == 1
    assert all(name in written[0] for name in COMMANDS)


def test_trace_shows_the_conversation_so_far():
    _, traced, code = record(make_session(), ["a question", "/trace", "/exit"])

    assert code == 0
    assert len(traced) == 1
    assert [str(m.content) for m in traced[0]] == ["a question", "I have heard 1 questions"]


def test_a_failed_turn_does_not_end_the_session():
    """A model or network error is reported, and the user can keep going."""
    session = make_session()
    asked: list[str] = []

    async def turn(sess: ChatSession, question: str) -> None:
        asked.append(question)
        if question == "boom":
            raise RuntimeError("the model stopped early")
        sess.ask(question)

    written, _, code = record(session, ["boom", "still there"], turn=turn)

    assert code == 0
    assert asked == ["boom", "still there"]
    assert written == ["error: the model stopped early", ""]


def test_only_a_leading_slash_is_a_command():
    """A slash inside a sentence must not be read as a command."""
    assert parse_command("What is 1 / 2?") is None
    assert parse_command("   ") is None
    assert parse_command("/Trace").name == "/trace"


def test_cli_chat_needs_a_terminal():
    """Without a question and without a tty, refuse instead of hanging."""
    from react_loop.__main__ import main

    assert main([]) == 2


def test_cli_chat_session_answers_offline(monkeypatch, capsys):
    """The no-argument path runs a real session and returns the answers."""
    from react_loop.__main__ import main

    class Tty(io.StringIO):
        def isatty(self) -> bool:
            return True

    monkeypatch.setattr(sys, "stdin", Tty())
    for name in ("LLM_PROVIDER", "REACT_MODEL", "LLM_MODEL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(
        "react_loop.__main__.build_llm",
        lambda **_kwargs: ScriptedChatModel(script=echo_history),
    )

    lines = iter(["first question", "second question", "/exit"])

    def ask(_prompt: str = "") -> str:
        try:
            return next(lines)
        except StopIteration:
            raise EOFError from None

    monkeypatch.setattr("builtins.input", ask)

    assert main(["--plain"]) == 0
    out = capsys.readouterr().out

    assert "I have heard 1 questions" in out
    # The second answer is only right if the first turn was remembered.
    assert "I have heard 2 questions" in out
