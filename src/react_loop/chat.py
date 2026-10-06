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

import asyncio
import re
import uuid
from collections.abc import AsyncIterator, Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any

from langchain_core.messages import AnyMessage

from react_loop.console import print_error, print_plain_trace
from react_loop.research import DEEP_RESEARCH_SYSTEM_PROMPT
from react_loop.runner import final_answer
from react_loop.streaming import StreamEvent

if TYPE_CHECKING:
    from react_loop.market_state import MarketStateManager
    from react_loop.runner import ReActRunner
    from react_loop.streaming import ReActStreamRunner

#: Shown before every question, keeping the user's own lines visually separate
#: from the panels the agent prints.
PROMPT = "you > "

#: Commands understood at the prompt. Kept in one place so that /help and the
#: dispatch in :func:`run_chat` cannot drift apart.
COMMANDS: dict[str, str] = {
    "/help": "Show this list of commands",
    "/clear": "Clear the current conversation history",
    "/trace": "Print the full conversation trace",
    "/sources": "List URLs and source references from this conversation",
    "/research": "Conduct exhaustive deep research on a topic",
    "/stock": "Research a stock quote, fundamentals, and technicals",
    "/forecast": "Research bear/base/bull scenarios for a stock and horizon",
    "/compare": "Compare comma-separated stock symbols",
    "/backtest": "Evaluate historical forward returns for a symbol and horizon",
    "/report": "Show the latest answer as a Markdown report",
    "/watch": "Add a symbol to the watchlist",
    "/unwatch": "Remove a symbol from the watchlist",
    "/watchlist": "Show the watchlist",
    "/portfolio": "Show holdings or set one: /portfolio AAPL 10 150",
    "/alert": "Add an alert: /alert AAPL below 150",
    "/alerts": "Show alerts and evaluate current prices",
    "/cache": "Show or clear persistent market-data cache",
    "/refresh": "Refresh market data: /refresh MSFT",
    "/calendar": "Show earnings and catalyst events",
    "/save": "Save the current session to database",
    "/load": "Load a session by thread ID: /load <id>",
    "/debug": "Toggle interactive debug mode (step-through)",
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
    debug_mode: bool = False

    def set_debug(self, enabled: bool):
        self.debug_mode = enabled


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
        if not self.remembers:
            return []
        state = self.runner.graph.get_state({"configurable": {"thread_id": self.thread_id}})
        return list((state.values or {}).get("messages", []))

    def inject_messages(self, messages: list[AnyMessage]):
        """Manually insert messages into the graph state."""
        self.runner.graph.update_state(
            {"configurable": {"thread_id": self.thread_id}},
            {"messages": messages}
        )

    def ask(self, question: str) -> str:
        """Run one turn to completion and return the final answer.

        A synchronous caller may invoke this method from an async test or
        application callback. In that case the runner's sync bridge runs in
        a worker thread because ``asyncio.run`` cannot nest in the active loop.
        """
        kwargs = {
            "recursion_limit": self.recursion_limit,
            "thread_id": self._thread_id,
        }
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            result = self.runner.run(question, **kwargs)
        else:
            with ThreadPoolExecutor(max_workers=1) as executor:
                result = executor.submit(self.runner.run, question, **kwargs).result()
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
    market_state: MarketStateManager | None = None,
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
    state = market_state
    render_trace = show_trace or partial(print_plain_trace)

    def get_market_state() -> MarketStateManager:
        nonlocal state
        if state is None:
            from react_loop.market_state import MarketStateManager

            state = MarketStateManager()
        return state
    market_service = None

    def get_market_service():
        nonlocal market_service
        if market_service is None:
            from react_loop.market_data import CachedMarketDataService

            market_service = CachedMarketDataService()
        return market_service
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
            if command.name == "/cache":
                from react_loop.market_data import MarketDataCache

                if command.arg.strip().lower() == "clear":
                    write(f"Cleared {MarketDataCache().clear()} market-cache entries.")
                else:
                    from react_loop.tools import market_cache_status

                    write(market_cache_status.invoke({}))
                continue
            if command.name == "/refresh":
                if not command.arg:
                    write("Usage: /refresh <symbol>")
                    continue
                try:
                    ticker = get_market_state().normalize_symbol(command.arg)
                    service = get_market_service()
                    result = service.quote(ticker, refresh=True)
                    freshness = "stale" if result.stale else "fresh"
                    write(
                        f"{ticker}: {result.value.price} as of {result.value.as_of} "
                        f"({freshness}; source {result.source.provider})"
                    )
                except Exception as exc:
                    write(f"Error: refresh failed: {exc}")
                continue
            if command.name == "/calendar":
                if not command.arg:
                    write("Usage: /calendar <symbol|watchlist|next 30d>")
                    continue
                from react_loop.tools import market_calendar

                calendar_parts = command.arg.split()
                if calendar_parts[0].lower() in {"watchlist", "next"}:
                    symbols = get_market_state().watchlist()
                    if not symbols:
                        write("Watchlist is empty.")
                        continue
                    days = 365
                    future_only = calendar_parts[0].lower() == "next"
                    if future_only and len(calendar_parts) > 1:
                        try:
                            days = int(calendar_parts[1].rstrip("d"))
                        except ValueError:
                            write("Usage: /calendar next <days>d")
                            continue
                    outputs = [
                        market_calendar.invoke(
                            {"symbol": symbol, "days": days, "future_only": future_only}
                        )
                        for symbol in symbols
                    ]
                    write("\n\n".join(outputs))
                else:
                    write(market_calendar.invoke({"symbol": calendar_parts[0]}))
                continue
            if command.name == "/save":
                from react_loop.persistence import SessionManager
                sm = SessionManager()
                sm.save_session(session.thread_id, session.messages())
                write(f"Session saved as {session.thread_id}.")
                continue
            if command.name == "/load":
                if not command.arg:
                    write("Please provide a thread ID: /load <id>")
                    continue
                from react_loop.persistence import SessionManager
                sm = SessionManager()
                msgs = sm.load_session(command.arg)
                if msgs:
                    session._thread_id = command.arg
                    session.inject_messages(msgs)
                    write(f"Session {command.arg} loaded.")
                else:
                    write(f"Session {command.arg} not found.")
                continue
            if command.name == "/debug":
                session.set_debug(not session.debug_mode)
                write(f"Debug mode {'enabled' if session.debug_mode else 'disabled'}.")
                continue
            if command.name == "/research":
                if not command.arg:
                    write("Please provide a topic: /research <topic>")
                    continue

                write(f"🔍 Deep researching: {command.arg}...")
                text = (
                    f"[RESEARCH MODE]\n{DEEP_RESEARCH_SYSTEM_PROMPT}\n\n"
                    f"Topic: {command.arg}"
                )
                # Do NOT continue; let it fall through to the turn() call.
            elif command.name in {"/stock", "/forecast", "/compare", "/backtest"}:
                if not command.arg:
                    write(f"Please provide a topic: {command.name} <symbol or question>")
                    continue
                text = (
                    f"[RESEARCH MODE]\n{DEEP_RESEARCH_SYSTEM_PROMPT}\n\n"
                    f"Topic: {command.arg}"
                )
                write(f"Researching: {command.arg}...")
            elif command.name == "/sources":
                urls = sorted(
                    {
                        match
                        for message in session.messages()
                        for match in re.findall(r"https?://[^ )]+", str(message.content))
                    }
                )
                write("\n".join(urls) if urls else "No URLs found in the conversation.")
                continue
            elif command.name == "/report":
                messages = session.messages()
                write(final_answer(messages) if messages else "No answer is available yet.")
                continue
            elif command.name == "/watch":
                if not command.arg:
                    write("Usage: /watch <symbol>")
                    continue
                try:
                    ticker = get_market_state().add_watch(command.arg)
                    write(f"Added {ticker} to the watchlist.")
                except ValueError as exc:
                    write(f"Error: {exc}")
                continue
            elif command.name == "/unwatch":
                if not command.arg:
                    write("Usage: /unwatch <symbol>")
                    continue
                try:
                    ticker = get_market_state().remove_watch(command.arg)
                    write(f"Removed {ticker} from the watchlist.")
                except ValueError as exc:
                    write(f"Error: {exc}")
                continue
            elif command.name == "/watchlist":
                parts = command.arg.split()
                if not parts:
                    symbols = get_market_state().watchlist()
                    write("Watchlist: " + (", ".join(symbols) if symbols else "(empty)"))
                    continue
                action = parts[0].lower()
                if action not in {"dashboard", "refresh", "export"}:
                    write("Usage: /watchlist [dashboard|refresh|export <path>]")
                    continue
                try:
                    service = get_market_service()
                    if action == "refresh":
                        for symbol in get_market_state().watchlist():
                            service.cache.clear(symbol)
                    from react_loop.market_dashboard import watchlist_output

                    if action == "export":
                        if len(parts) != 2:
                            write("Usage: /watchlist export <path>")
                            continue
                        path = parts[1]
                        output_format = "csv" if path.lower().endswith(".csv") else "json"
                        content = watchlist_output(get_market_state(), service, output_format)
                        from pathlib import Path

                        Path(path).write_text(content, encoding="utf-8")
                        write(f"Watchlist exported to {path}.")
                    else:
                        write(watchlist_output(get_market_state(), service))
                except Exception as exc:
                    write(f"Error: watchlist dashboard unavailable: {exc}")
                continue
            elif command.name == "/portfolio":
                parts = command.arg.split()
                action = parts[0].lower() if parts else "summary"
                if action in {"summary", "performance", "allocation", "export"}:
                    try:
                        service = get_market_service()
                        from react_loop.market_dashboard import build_portfolio_report
                        from react_loop.portfolio import render_portfolio

                        report = build_portfolio_report(get_market_state(), service)
                        if action == "allocation":
                            rows = report["allocation"]
                            write(
                                "Symbol | Sector | Market value | Allocation\n"
                                "--- | --- | ---: | ---:\n"
                                + "\n".join(
                                    f"{row['symbol']} | {row['sector'] or 'unknown'} | "
                                    f"{row['market_value'] if row['market_value'] is not None else 'unavailable'} | "
                                    f"{row['percent'] if row['percent'] is not None else 'unavailable'}"
                                    for row in rows
                                )
                            )
                        elif action == "export":
                            if len(parts) != 2:
                                write("Usage: /portfolio export <path>")
                                continue
                            path = parts[1]
                            output_format = "csv" if path.lower().endswith(".csv") else "json"
                            from pathlib import Path

                            Path(path).write_text(
                                render_portfolio(report, output_format), encoding="utf-8"
                            )
                            write(f"Portfolio exported to {path}.")
                        else:
                            write(render_portfolio(report))
                    except Exception as exc:
                        write(f"Error: portfolio analytics unavailable: {exc}")
                    continue
                if len(parts) != 3:
                    write("Usage: /portfolio <symbol> <quantity> <cost_basis>")
                    continue
                try:
                    ticker = get_market_state().set_position(
                        parts[0], float(parts[1]), float(parts[2])
                    )
                    write(f"Saved {ticker} position.")
                except ValueError as exc:
                    write(f"Error: {exc}")
                continue
            elif command.name == "/alert":
                parts = command.arg.split()
                manager = get_market_state()
                if parts and parts[0].lower() in {"disable", "enable", "delete"}:
                    if len(parts) != 2:
                        write("Usage: /alert <disable|enable|delete> <id>")
                        continue
                    try:
                        alert_id = int(parts[1])
                        if parts[0].lower() == "delete":
                            manager.delete_alert(alert_id)
                            write(f"Deleted alert {alert_id}.")
                        else:
                            manager.set_alert_enabled(alert_id, parts[0].lower() == "enable")
                            write(f"Alert {alert_id} {'enabled' if parts[0].lower() == 'enable' else 'disabled'}.")
                    except (ValueError, TypeError) as exc:
                        write(f"Error: {exc}")
                    continue
                condition_type = "price"
                if len(parts) == 3:
                    symbol, operator, threshold_text = parts
                elif len(parts) in {4, 5}:
                    symbol, condition_type, operator, threshold_text = parts[:4]
                else:
                    write(
                        "Usage: /alert <symbol> [price|change|volume|rsi|moving_average|drawdown] "
                        "<above|below> <threshold> [cooldown_seconds]"
                    )
                    continue
                try:
                    cooldown = int(parts[4]) if len(parts) == 5 else 3600
                    alert_id = manager.add_alert(
                        symbol,
                        operator.lower(),
                        float(threshold_text.rstrip("%")),
                        condition_type,
                        cooldown,
                    )
                    write(f"Created alert {alert_id}.")
                except ValueError as exc:
                    write(f"Error: {exc}")
                continue
            elif command.name == "/alerts":
                if command.arg.strip().lower() == "history":
                    history = get_market_state().alert_history()
                    write(
                        "\n".join(
                            f"{row['triggered_at']} | alert {row['alert_id']} | "
                            f"{row['status']} | {row['message']}"
                            for row in history
                        )
                        or "Alert history: (empty)"
                    )
                    continue
                configured = get_market_state().alerts()
                triggered = get_market_state().evaluate_alerts() if configured else []
                lines = [
                    f"{alert['id']}: {alert['symbol']} {alert['condition_type']} "
                    f"{alert['operator']} {alert['threshold']:.2f} "
                    f"({'enabled' if alert['enabled'] else 'disabled'})"
                    for alert in configured
                ]
                if triggered:
                    lines.append("Triggered: " + "; ".join(triggered))
                write("\n".join(lines) if lines else "Alerts: (none)")
                continue
            else:
                write(f"Unknown command {command.name}. Type /help for the list.")
                continue
        try:
            await turn(session, text)
        except Exception as exc:
            on_error(exc)
