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


from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.formatted_text import HTML
from react_loop.chat import COMMANDS
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
        """Run one question and print the answer, with HITL approval for tools."""
        async def approve_tool(tool_calls: list[dict]) -> str | list[dict]:
            for tc in tool_calls:
                console.print(f"[bold yellow]⚠ Tool Call Request:[/] {tc['name']}({tc['args']})")
            
            choice = await prompt_session.prompt_async("[bold yellow]Approve? [A]pprove, [E]dit, [C]ancel: [/]")
            choice = choice.strip().lower()
            if choice.startswith('a'):
                return "approve"
            if choice.startswith('c'):
                return "cancel"
            if choice.startswith('e'):
                # Simple edit: just ask for a new JSON string for the first tool call
                # (In a real app, we'd loop through all tool calls)
                new_args = await prompt_session.prompt_async("[bold yellow]Enter new args (JSON): [/]")
                import json
                try:
                    args = json.loads(new_args)
                    return [{**tool_calls[0], "args": args}]
                except json.JSONDecodeError:
                    console.print("[red]Invalid JSON. Cancelling tool call.[/]")
                    return "cancel"
            return "cancel"

        if streaming:
            await stream_events(session.astream(question, approval_callback=approve_tool), console)
            print()
        else:
            answer = await session.aask(question, approval_callback=approve_tool)
            console.print(f"[bold green]assistant>[/] {answer}")
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

    # Use prompt_toolkit for autocomplete suggestions
    completer = WordCompleter(list(COMMANDS.keys()), ignore_case=True)
    prompt_session = PromptSession(completer=completer)

    async def read_line(_prompt: str) -> str:
        return await prompt_session.prompt_async(HTML('<cyan><b>you > </b></cyan>'))
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
    parser.add_argument(
        "--deep-research",
        action="store_true",
        help="Run an exhaustive research loop on the topic",
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

    final_text = messages[-1].content
    if use_rich:
        console.print(f"\n[bold green]Final Answer:[/]\n{final_text}")
    else:
        print(f"\nFinal Answer:\n{final_text}")

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
    interactive = not args.demo and not args.question and not args.deep_research
    deep_research_mode = args.deep_research
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

    if deep_research_mode:
        # Use a specialized system prompt to force exhaustive research
        research_prompt = (
            "You are a professional deep research agent. Your goal is to provide "
            "an exhaustive, detailed report on the topic. Do not stop at the first "
            "satisfactory answer. Search multiple sources, dig into details, "
            "and synthesize a comprehensive final answer."
        )
        from react_loop.runner import run_react
        result = run_react(
            question=question,
            llm=llm,
            system_prompt=research_prompt,
            verbose=use_rich
        )
        if use_rich:
            from react_loop.console import print_plain_trace
            print_plain_trace(result["messages"])
        else:
            from react_loop.runner import trace
            for line in trace(result["messages"]):
                print(line)
        return 0

    return _run_once(llm, question, args, console, use_rich)
if __name__ == "__main__":
    raise SystemExit(main())
