"""Command line entry point.

``react-loop "your question"`` answers once and exits. ``react-loop`` with no
question starts an interactive session that remembers the conversation.
"""

import argparse
import asyncio
import sys
from pathlib import Path

from langchain_core.messages import AnyMessage
from langgraph.checkpoint.memory import InMemorySaver
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.formatted_text import HTML
from rich.console import Console
from rich.text import Text

from react_loop.chat import COMMANDS, ChatSession, banner, run_chat
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
from react_loop.reporting import render_report
from react_loop.research import DEEP_RESEARCH_SYSTEM_PROMPT
from react_loop.runner import ReActRunner, final_answer
from react_loop.streaming import ReActStreamRunner


def _prompt(text: str, color: str, rich: bool) -> str | HTML:
    """Format a prompt for prompt_toolkit without exposing invalid markup."""
    if not rich:
        return text
    return HTML(f"<style fg='{color}' bold='true'>{text}</style>")

def _run_streaming(
    llm: ReActModel,
    question: str,
    args: argparse.Namespace,
    console: Console,
    use_rich: bool,
    *,
    system_prompt: str | None = None,
) -> int:
    """Run one question with token streaming. Returns an exit code."""
    runner = ReActStreamRunner(
        llm,
        system_prompt=args.system_prompt if system_prompt is None else system_prompt,
    )
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

    approved_tools: set[str] = set()

    async def approve_tool(tool_calls: list[dict]) -> str | list[dict]:
        """Approve each tool once per interactive process.

        Explicit approvals are remembered by tool name. Edited calls stay
        one-off because their arguments are specific to the current request;
        cancellations are also not remembered so the user can approve a later
        request intentionally.
        """
        if tool_calls and all(tc["name"] in approved_tools for tc in tool_calls):
            names = ", ".join(sorted({tc["name"] for tc in tool_calls}))
            console.print(f"[dim]↳ Using remembered approval: {names}[/]")
            return "approve"

        for tc in tool_calls:
            request = Text("⚠ Tool Call Request: ", style="bold yellow")
            request.append(f"{tc['name']}({tc['args']})")
            console.print(request)

        approval_prompt = _prompt(
            "Approve? [A]pprove, [E]dit, [C]ancel: ", "yellow", use_rich
        )
        choice = await prompt_async(approval_prompt)
        choice = choice.strip().lower()
        if choice.startswith("a") or choice == "y":
            approved_tools.update(tc["name"] for tc in tool_calls)
            names = ", ".join(sorted({tc["name"] for tc in tool_calls}))
            console.print(f"[green]✓ Approved for this session:[/] {names}")
            return "approve"
        if choice.startswith("c"):
            return "cancel"
        if choice.startswith("e"):
            # Simple edit: just ask for a new JSON string for the first tool call.
            # Edited calls are not remembered because their arguments are one-off.
            edit_prompt = _prompt("Enter new args (JSON): ", "yellow", use_rich)
            new_args = await prompt_async(edit_prompt)
            import json
            try:
                args = json.loads(new_args)
                return [{**tool_calls[0], "args": args}]
            except json.JSONDecodeError:
                console.print("[red]Invalid JSON. Cancelling tool call.[/]")
                return "cancel"
        console.print("[yellow]Tool call cancelled.[/]")
        return "cancel"

    async def turn(session: ChatSession, question: str) -> None:
        """Run one question and print the answer, with HITL approval for tools."""

        if session.debug_mode:
            # Debug mode: step-through execution
            console.print("[bold magenta]🐞 Debug Mode: Step-through enabled.[/]")
            async for event in session.astream(question, approval_callback=approve_tool):
                if hasattr(event, "name") and hasattr(event, "args"): # ToolCallEvent
                    console.print(f"[bold yellow]Step: Agent requested {event.name}[/]")
                    next_prompt = _prompt(
                        "Next? [Y]es / [S]teer: ", "magenta", use_rich
                    )
                    choice = await prompt_session.prompt_async(next_prompt)
                    if choice.strip().lower().startswith("s"):
                        steer_prompt = _prompt(
                            "Steer (Enter new response): ", "magenta", use_rich
                        )
                        steering = await prompt_session.prompt_async(steer_prompt)
                        from langchain_core.messages import AIMessage
                        session.inject_messages([AIMessage(content=steering)])
                        console.print("[bold green]Steered. Terminating turn.[/]")
                        break
                elif hasattr(event, "content") and not hasattr(event, "text"): # ToolResultEvent
                    console.print(f"[bold blue]Step: Tool {event.name} returned output.[/]")
                    continue_prompt = _prompt(
                        "Press Enter to continue... ", "magenta", use_rich
                    )
                    await prompt_session.prompt_async(continue_prompt)
                elif hasattr(event, "text"): # TokenEvent
                    console.print(event.text, end="")
                elif hasattr(event, "answer"): # FinalEvent
                    console.print(f"\n[bold green]Final Answer:[/] {event.answer}")
            print()
            return

        if streaming:
            await stream_events(session.astream(question, approval_callback=approve_tool), console)
            print()
        else:
            answer = await session.aask(question, approval_callback=approve_tool)
            from react_loop.console import render_markdown
            console.print(f"[bold green]assistant>[/] ")
            render_markdown(answer, console=console)
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

    async def prompt_async(prompt: str | HTML = "") -> str:
        try:
            return await prompt_session.prompt_async(prompt)
        except TypeError as exc:
            if "positional arguments" not in str(exc):
                raise
            return await prompt_session.prompt_async()

    async def read_line(_prompt: str) -> str:
        return await prompt_async(HTML('<style fg="cyan" bold="true">👤 you > </style>'))
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
    parser.add_argument(
        "--format",
        dest="report_format",
        choices=("text", "markdown", "json", "html"),
        default="text",
        help="Report output format",
    )
    parser.add_argument(
        "--output",
        help="Write the report to a file; use '-' for stdout",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Clear persistent market-data cache before running the request",
    )
    parser.add_argument(
        "--monitor-alerts",
        action="store_true",
        help="Run the persistent alert scheduler until interrupted",
    )
    parser.add_argument(
        "--alert-interval",
        type=float,
        default=60.0,
        help="Alert scheduler interval in seconds",
    )
    parser.add_argument(
        "--alert-webhook",
        help="Optional webhook URL for the alert scheduler",
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
    parser.add_argument(
        "--force-interactive",
        action="store_true",
        help="Force interactive mode even if stdin is not a terminal",
    )
    return parser


def _emit_report(
    final_text: str,
    question: str,
    args: argparse.Namespace,
    console: Console,
    use_rich: bool,
) -> None:
    """Print or export the final answer in the requested report format."""
    rendered = render_report(final_text, question, args.report_format)
    if args.output:
        if args.output == "-":
            print(rendered, end="" if rendered.endswith("\n") else "\n")
        else:
            Path(args.output).write_text(rendered, encoding="utf-8")
            print(f"Report written to {args.output}")
        return
    if args.report_format != "text":
        print(rendered, end="" if rendered.endswith("\n") else "\n")
        return
    if use_rich:
        console.print(f"\n[bold green]Final Answer:[/]\n{final_text}")
    else:
        print(f"\nFinal Answer:\n{final_text}")


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
    _emit_report(final_text, question, args, console, use_rich)

    return 0

def main(argv: list[str] | None = None) -> int:
    """Answer one question, or run an interactive session."""
    args = _build_parser().parse_args(argv)
    if args.stream and (args.output or args.report_format != "text"):
        print("error: --stream cannot be combined with report export options.", file=sys.stderr)
        return 2

    # Colour only when writing to a terminal, so redirected output stays plain.
    use_rich = sys.stdout.isatty() if args.rich is None else args.rich
    console = get_console(force_terminal=use_rich or None)
    setup_logging(args.log_level, force_terminal=use_rich or None)
    if args.refresh:
        from react_loop.market_data import MarketDataCache

        MarketDataCache().clear()
    if args.monitor_alerts:
        from react_loop.alerts import AlertScheduler, WebhookNotifier
        from react_loop.market_state import MarketStateManager

        notifiers = [WebhookNotifier(args.alert_webhook)] if args.alert_webhook else None
        try:
            AlertScheduler(
                MarketStateManager(),
                notifiers=notifiers,
                interval_seconds=args.alert_interval,
            ).run_forever()
        except KeyboardInterrupt:
            return 0
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

    # No question and no demo means the user wants a session. Reading from a
    # pipe would look like a hang, so refuse instead of waiting for input that
    # is never coming.
    interactive = not args.demo and not args.question and not args.deep_research
    deep_research_mode = args.deep_research
    if interactive and not sys.stdin.isatty() and not args.force_interactive:
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
    if deep_research_mode and not question.strip():
        print("error: --deep-research requires a research topic.", file=sys.stderr)
        return 2
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
        research_prompt = DEEP_RESEARCH_SYSTEM_PROMPT
        if args.system_prompt:
            research_prompt += f"\n\nAdditional instructions:\n{args.system_prompt}"

        if args.stream:
            return _run_streaming(
                llm,
                question,
                args,
                console,
                use_rich,
                system_prompt=research_prompt,
            )

        from react_loop.runner import run_react

        try:
            result = run_react(
                question=question,
                llm=llm,
                system_prompt=research_prompt,
                recursion_limit=args.max_steps,
                verbose=False,
            )
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

        final_text = final_answer(messages)
        _emit_report(final_text, question, args, console, use_rich)
        return 0

    return _run_once(llm, question, args, console, use_rich)
if __name__ == "__main__":
    raise SystemExit(main())
