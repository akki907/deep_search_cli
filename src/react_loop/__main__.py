"""Command line entry point.

``react-loop "your question"`` answers once and exits. ``react-loop`` with no
question starts an interactive session that remembers the conversation.
"""

import argparse
import asyncio
import sys

from langchain_core.messages import AnyMessage
from langgraph.checkpoint.memory import InMemorySaver
from rich.console import Console

from react_loop.chat import PROMPT, ChatSession, banner, run_chat
from react_loop.console import (
    get_console,
    print_error,
    print_plain_trace,
    print_trace,
    setup_logging,
    stream_events,
)
from react_loop.demo import DEMO_QUESTION, DEMO_SCRIPT
from react_loop.llm import ReActModel, ScriptedChatModel, build_llm, resolve_target
from react_loop.runner import ReActRunner
from react_loop.streaming import ReActStreamRunner


def _run_streaming(
    llm: ReActModel, question: str, args: argparse.Namespace, console: Console, use_rich: bool
) -> int:
    """Run one question with token streaming. Returns an exit code."""
    runner = ReActStreamRunner(llm, system_prompt=args.system_prompt)
    try:
        asyncio.run(
            stream_events(runner.astream(question, recursion_limit=args.max_steps), console)
        )
    except Exception as exc:
        if use_rich:
            print_error(exc, Console(stderr=True, force_terminal=True))
        else:
            print(f"error: the agent stopped early: {exc}", file=sys.stderr)
        return 1
    print()
    return 0


def _run_chat(llm: ReActModel, args: argparse.Namespace, console: Console, use_rich: bool) -> int:
    """Run an interactive session. Returns an exit code.

    The checkpointer is created here and lives only for this process, so the
    conversation is remembered while the session runs and forgotten on exit.
    """
    saver = InMemorySaver()
    # Streaming is the point of a session, so it is on whenever output goes
    # to a terminal. Redirected output stays plain, as in the one-shot path.
    streaming = args.stream or console.is_terminal

    runner: ReActRunner | ReActStreamRunner
    if streaming:
        runner = ReActStreamRunner(llm, system_prompt=args.system_prompt, checkpointer=saver)
    else:
        runner = ReActRunner(llm, system_prompt=args.system_prompt, checkpointer=saver)

    async def turn(session: ChatSession, question: str) -> None:
        """Run one question and print the answer."""
        if streaming:
            await stream_events(session.astream(question), console)
            print()
        else:
            console.print(f"[bold green]assistant>[/] {session.ask(question)}")

    error_console = Console(stderr=True, force_terminal=True) if use_rich else None

    def on_error(exc: BaseException) -> None:
        if error_console is not None:
            print_error(exc, error_console)
        else:
            print(f"error: {exc}", file=sys.stderr)

    def show_trace(messages: list[AnyMessage]) -> None:
        if use_rich:
            print_trace(messages, console)
        else:
            print_plain_trace(messages, console)

    def read_line(_prompt: str) -> str:
        # Rich's Prompt adds its own ': ' suffix, so the marker is drawn here
        # and input() only reads. input() raises EOFError at end of input and
        # KeyboardInterrupt on Ctrl-C, both of which end the session.
        console.print(f"[bold cyan]{PROMPT.strip()}[/] ", end="")
        return input()

    session = ChatSession(runner, recursion_limit=args.max_steps)
    provider, model = resolve_target(args.model, args.provider)
    console.print(banner(provider, model))
    return asyncio.run(
        run_chat(
            session,
            read_line=read_line,
            turn=turn,
            show_trace=show_trace,
            on_error=on_error,
        )
    )


def _build_parser() -> argparse.ArgumentParser:
    """Build the command line parser."""
    parser = argparse.ArgumentParser(prog="react_loop", description="Run a LangGraph ReAct loop.")
    parser.add_argument("question", nargs="*", help="The question to answer.")
    parser.add_argument(
        "--provider",
        default=None,
        help="openai | openrouter | together | groq | ollama | anthropic | scripted",
    )
    parser.add_argument("--model", default=None, help="Model name, e.g. gpt-4o-mini")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--system-prompt", default=None)
    parser.add_argument("--max-steps", type=int, default=25, help="LangGraph recursion limit")
    parser.add_argument("--no-trace", action="store_true", help="Print only the final answer")
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream the answer token by token as the model writes it",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run a canned offline ReAct trajectory (no API key needed)",
    )
    output = parser.add_mutually_exclusive_group()
    output.add_argument(
        "--rich",
        dest="rich",
        action="store_true",
        default=None,
        help="Force coloured Rich panels (the default on a terminal)",
    )
    output.add_argument(
        "--plain",
        dest="rich",
        action="store_false",
        help="Force plain text, no colour or panels",
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        help="Log level for the Rich logger: DEBUG, INFO, WARNING, ERROR",
    )
    return parser


def _run_once(
    llm: ReActModel, question: str, args: argparse.Namespace, console: Console, use_rich: bool
) -> int:
    """Answer one question and exit. Returns an exit code."""
    if args.stream:
        return _run_streaming(llm, question, args, console, use_rich)

    runner = ReActRunner(llm, system_prompt=args.system_prompt)
    try:
        result = runner.run(question, recursion_limit=args.max_steps)
    except Exception as exc:
        if use_rich:
            print_error(exc, Console(stderr=True, force_terminal=True))
        else:
            print(f"error: the agent stopped early: {exc}", file=sys.stderr)
        return 1

    messages = result["messages"]
    if not args.no_trace:
        if use_rich:
            print_trace(messages, console)
        else:
            print_plain_trace(messages, console)
        print()
    print(messages[-1].content)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Answer one question, or run an interactive session."""
    args = _build_parser().parse_args(argv)

    # Colour only when writing to a terminal, so redirected output stays plain.
    use_rich = sys.stdout.isatty() if args.rich is None else args.rich
    console = get_console(force_terminal=use_rich or None)
    setup_logging(args.log_level, force_terminal=use_rich or None)

    # No question and no demo means the user wants a session. Reading from a
    # pipe would look like a hang, so refuse instead of waiting for input that
    # is never coming.
    interactive = not args.demo and not args.question
    if interactive and not sys.stdin.isatty():
        print(
            "error: no question given and stdin is not a terminal.\n"
            "Pass a question, use --demo, or run react-loop in an interactive shell.",
            file=sys.stderr,
        )
        return 2

    llm: ReActModel | ScriptedChatModel | None = None
    if args.demo:
        question = DEMO_QUESTION
        llm = ScriptedChatModel(script=list(DEMO_SCRIPT))
    else:
        question = " ".join(args.question)
    try:
        llm = llm or build_llm(
            model=args.model, temperature=args.temperature, provider=args.provider
        )
    except (RuntimeError, ValueError) as exc:
        # Errors go to stderr so a shell can still capture the answer on stdout.
        if use_rich:
            print_error(exc, Console(stderr=True, force_terminal=True))
        else:
            print(f"error: {exc}", file=sys.stderr)
        return 2

    if interactive:
        return _run_chat(llm, args, console, use_rich)

    return _run_once(llm, question, args, console, use_rich)


if __name__ == "__main__":
    raise SystemExit(main())
