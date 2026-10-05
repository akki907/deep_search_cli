"""Rich rendering for the ReAct loop.

The plain-text :func:`react_loop.runner.trace` output stays the default, so
piping to a file or a test keeps working. This module adds colour, panels, and
structured logging for a real terminal.
"""

import asyncio
import logging
import os
import sys
from collections.abc import AsyncIterator
from typing import Any

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage
from rich.console import Console, Group
from rich.logging import RichHandler
from rich.markdown import Markdown
from rich.panel import Panel

from react_loop.runner import trace
from react_loop.streaming import (
    FinalEvent,
    TokenEvent,
    ToolCallEvent,
    ToolResultEvent,
)

#: Tool results are long. Clip them in the panel view only.
TOOL_PREVIEW_CHARS = 400


#: UI Style Guide
COLOR_USER = "cyan"
COLOR_AGENT = "green"
COLOR_TOOL_CALL = "yellow"
COLOR_TOOL_RESULT = "blue"
COLOR_DEBUG = "magenta"
COLOR_ERROR = "red"

STYLE_BOLD = "bold"
STYLE_DIM = "dim"

log = logging.getLogger("react_loop")


def get_console(force_terminal: bool | None = None) -> Console:
    """Return a Rich console.

    Colour is disabled when ``NO_COLOR`` is set or the output is redirected, so
    redirected output stays clean ASCII.
    """
    return Console(
        force_terminal=force_terminal,
        no_color=bool(os.getenv("NO_COLOR")),
        soft_wrap=False,
    )


def setup_logging(level: str | int = "INFO", *, force_terminal: bool | None = None) -> None:
    """Route the package logger through Rich.

    Args:
        level: Logging level name such as ``DEBUG`` or an int.
        force_terminal: Passed to :class:`rich.console.Console`.

    """
    import logging
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    handler = RichHandler(
        console=get_console(force_terminal),
        show_time=False,
        show_path=False,
        rich_tracebacks=True,
        markup=True,
    )
    handler.setFormatter(logging.Formatter("%(message)s"))
    root = logging.getLogger("react_loop")
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)
    root.propagate = False


def _panel_for(message: AnyMessage) -> Panel | None:
    """Return a styled panel for one message, or None to skip it."""
    if isinstance(message, HumanMessage):
        return Panel(
            str(message.content),
            title="[bold blue]user[/]",
            border_style="blue",
            padding=(0, 1),
        )
    if isinstance(message, AIMessage):
        if message.tool_calls:
            calls = "\n".join(
                f"[bold]{call['name']}[/]([cyan]{call['args']}[/])" for call in message.tool_calls
            )
            return Panel(calls, title="[bold magenta]assistant · tool call[/]", border_style="magenta")
        return Panel(
            Markdown(str(message.content)),
            title="[bold green]assistant[/]",
            border_style="green",
        )
    if isinstance(message, ToolMessage):
        text = str(message.content)
        clipped = (
            text[:TOOL_PREVIEW_CHARS] + "..."
            if len(text) > TOOL_PREVIEW_CHARS
            else text
        )
        return Panel(
            clipped,
            title=f"[bold yellow]tool · {message.name}[/]",
            border_style="yellow",
        )
    return None


def render_messages(messages: list[AnyMessage]) -> Group:
    """Build a renderable group of panels, one per message."""
    panels = [p for p in (_panel_for(m) for m in messages) if p is not None]
    return Group(*panels)


def print_trace(messages: list[AnyMessage], console: Console | None = None) -> None:
    """Print the loop as Rich panels."""
    (console or get_console()).print(render_messages(messages))


def print_error(exc: BaseException, console: Console | None = None) -> None:
    """Print an error in a red panel."""
    (console or get_console()).print(
        Panel(str(exc), title="[bold red]error[/]", border_style="red")
    )


def print_plain_trace(messages: list[AnyMessage], console: Console | None = None) -> None:
    """Print the original plain-text trace, highlighted by Rich.

    Used when output is redirected, so logs and test assertions keep the
    ``user >`` / ``assistant>`` / ``tool <`` format of :func:`trace`.
    """
    con = console or get_console(force_terminal=False)
    styles = {"user": "blue", "assistant": "green", "tool": "yellow"}
    for line in trace(messages):
        style = next((s for k, s in styles.items() if line.startswith(k)), "white")
        con.print(line, style=style, highlight=False)


async def stream_events(events: AsyncIterator[Any], console: Console | None = None) -> str:
    """Render a stream of events live, returning the final answer.

    Tool calls and results are printed above a live answer line. Tokens appear
    as they arrive. When the output is not a terminal the text is written
    without cursor control, so redirected output stays readable.

    Args:
        events: An async iterator of stream events.
        console: Console to print to.

    Returns:
        The full answer text.

    """
    con = console or get_console()
    live_enabled = con.is_terminal and not con.no_color
    answer: list[str] = []
    first_token = True
    live = con.status("[bold green]thinking...", spinner="dots") if live_enabled else None
    if live is not None:
        live.start()
    try:
        async for event in events:
            if isinstance(event, ToolCallEvent):
                await _settle(live)
                con.print(
                    f"[bold magenta]\u25b8 tool[/] [bold]{event.name}[/]"
                    f"[cyan]({event.args})[/]"
                )
            elif isinstance(event, ToolResultEvent):
                await _settle(live)
                preview = event.content.replace(chr(10), " ")[:TOOL_PREVIEW_CHARS]
                con.print(f"[bold yellow]\u25b8 result[/] {preview}")
            elif isinstance(event, TokenEvent):
                if first_token:
                    await _settle(live)
                    first_token = False
                answer.append(event.text)
                if live_enabled:
                    con.print(f"[green]{event.text}[/]", end="")
                else:
                    sys.stdout.write(event.text)
                    sys.stdout.flush()
            elif isinstance(event, FinalEvent):
                if live_enabled:
                    con.print()
    finally:
        if live is not None:
            live.stop()
    return "".join(answer)


async def _settle(live: Any) -> None:
    """Stop the spinner so a following print is not overwritten."""
    if live is not None:
        live.stop()


def render_markdown(text: str, console: Console | None = None) -> str:
    """Render text as Markdown, with code blocks highlighted.

    Uses markdown_it.MarkdownIt to parse the source and Rich's
    Markdown to render it in the terminal. Code fences are displayed
    with syntax highlighting if a terminal is available.

    Args:
        text: The markdown text to render.
        console: Rich console to render to. When omitted the function
            returns the plain-text representation so it can be written
            to a file or captured by tests.

    Returns:
        The full rendered text. When a console is supplied the function
        prints to it and returns the exported text; otherwise it returns
        the raw markdown-processed string.

    """
    from rich.markdown import Markdown

    con = console or get_console(force_terminal=False)
    # When not a terminal we strip ANSI codes so the result is file-safe.
    # Use a Pygments style that exists; None falls back to Rich's default.
    rendered = Markdown(text, code_theme="solarized-dark" if con.is_terminal else None)
    con.print(rendered)
    return con.export_text()
