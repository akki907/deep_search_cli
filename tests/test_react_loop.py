"""Tests for the ReAct loop. All of these run offline."""

import os

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from react_loop.graph import build_react_graph, should_continue
from react_loop.llm import ScriptedChatModel, build_llm
from react_loop.runner import ReActRunner, final_answer, trace
from react_loop.tools import calculator, search_knowledge_base


def tool_call(name: str, args: dict, call_id: str = "c1") -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


def test_calculator_evaluates_arithmetic():
    assert calculator.invoke({"expression": "(18 * 4) / 3"}) == "24"
    assert calculator.invoke({"expression": "2 ** 10"}) == "1024"


@pytest.mark.parametrize("bad", ["__import__('os')", "open('/etc/passwd')", "name", "1 +"])
def test_calculator_rejects_unsafe_input(bad):
    result = calculator.invoke({"expression": bad})
    assert result.startswith("Error:")


def test_knowledge_base_search_and_miss():
    hit = search_knowledge_base.invoke({"query": "What is the refund policy?"})
    assert "30 days" in hit
    assert search_knowledge_base.invoke({"query": "penguin migration"}).startswith("No matching")


def test_should_continue_routes_on_tool_calls():
    calling = tool_call("calculator", {"expression": "1+1"})
    assert should_continue({"messages": [calling]}) == "tools"
    assert should_continue({"messages": [AIMessage(content="done")]}) == "__end__"


def test_loop_calls_tool_then_answers():
    script = [
        tool_call("search_knowledge_base", {"query": "refund policy"}),
        AIMessage(content="Refunds are available within 30 days."),
    ]
    result = ReActRunner(ScriptedChatModel(script=script)).run("What is the refund policy?")

    kinds = [type(m).__name__ for m in result["messages"]]
    assert kinds == ["HumanMessage", "AIMessage", "ToolMessage", "AIMessage"]
    assert final_answer(result["messages"]) == "Refunds are available within 30 days."
    assert result["steps"] == 2


def test_loop_handles_multiple_tool_rounds():
    script = [
        tool_call("search_knowledge_base", {"query": "shipping times"}, "c1"),
        tool_call("calculator", {"expression": "15 * 3"}, "c2"),
        AIMessage(content="Express shipping for 3 items costs $45 and takes 1-2 days."),
    ]
    runner = ReActRunner(ScriptedChatModel(script=script))
    result = runner.run("Cost of express shipping for 3 items?")
    assert sum(isinstance(m, ToolMessage) for m in result["messages"]) == 2
    assert result["steps"] == 3


def test_model_result_is_appended_not_replaced():
    script = [AIMessage(content="Paris.")]
    result = ReActRunner(ScriptedChatModel(script=script)).run("Capital of France?")
    assert isinstance(result["messages"][0], HumanMessage)
    assert result["messages"][0].content == "Capital of France?"


def test_tool_errors_come_back_as_messages():
    script = [
        tool_call("calculator", {"expression": "import os"}),
        AIMessage(content="I could not run that expression."),
    ]
    result = ReActRunner(ScriptedChatModel(script=script)).run("run import os")
    tool_msgs = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert tool_msgs and "Error" in str(tool_msgs[0].content)


def test_system_prompt_is_not_duplicated_across_turns():
    llm = ScriptedChatModel(
        script=[tool_call("get_current_time", {}), AIMessage(content="It is now.")]
    )
    ReActRunner(llm, system_prompt="You are terse.").run("What time is it?")
    first_call, second_call = llm.calls
    assert len(first_call) == 2 and len(second_call) == 4
    assert sum(isinstance(m, ToolMessage) for m in second_call) == 1


def test_recursion_limit_stops_a_looping_agent():
    from langgraph.errors import GraphRecursionError

    script = [tool_call("get_current_time", {}, f"c{i}") for i in range(20)]
    with pytest.raises(GraphRecursionError):
        ReActRunner(ScriptedChatModel(script=script)).run("loop forever", recursion_limit=4)


def test_chat_only_loop_without_tools():
    graph = build_react_graph(ScriptedChatModel(script=[AIMessage(content="hi")]), tools=[])
    result = graph.invoke({"messages": [HumanMessage(content="hello")]})
    assert final_answer(result["messages"]) == "hi"


def test_trace_renders_every_step():
    script = [
        tool_call("calculator", {"expression": "21 * 2"}),
        AIMessage(content="The answer is 42."),
    ]
    result = ReActRunner(ScriptedChatModel(script=script)).run("21*2?")
    lines = trace(result["messages"])
    assert lines[0].startswith("user")
    assert any("call calculator" in line for line in lines)
    assert any(line.startswith("tool") for line in lines)
    assert lines[-1].endswith("The answer is 42.")


def test_state_reducer_totals_steps():
    from react_loop.state import _add_steps

    assert _add_steps(1, 2) == 3
    assert _add_steps(None, 2) == 2
def test_conversation_memory_with_checkpointer():
    from langgraph.checkpoint.memory import InMemorySaver

    from react_loop.llm import ScriptedChatModel

    save = InMemorySaver()
    llm = ScriptedChatModel(script=[AIMessage(content="First."), AIMessage(content="Second.")])
    runner = ReActRunner(llm, checkpointer=save)
    config = {"configurable": {"thread_id": "t1"}}
    runner.graph.invoke({"messages": [HumanMessage(content="one")]}, config)
    result = runner.graph.invoke({"messages": [HumanMessage(content="two")]}, config)
    history = [m.content for m in result["messages"] if isinstance(m, HumanMessage)]
    assert history == ["one", "two"]


def test_scripted_model_satisfies_the_model_protocol():
    from react_loop.llm import ReActModel

    assert isinstance(ScriptedChatModel(), ReActModel)
def test_cli_demo_runs_offline(capsys):
    from react_loop.__main__ import main

    assert main(["--demo"]) == 0
    out = capsys.readouterr().out
    assert "call search_knowledge_base" in out
    assert "call calculator" in out
    assert "$45" in out


def test_cli_rejects_unknown_provider(capsys):
    from react_loop.__main__ import main

    assert main(["hello", "--provider", "nope"]) == 2
    assert "error" in capsys.readouterr().err

@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="set OPENAI_API_KEY to run live model tests",
)
def test_live_openai_agent_answers_a_question():
    from react_loop.runner import ReActRunner
    from react_loop.tools import calculator

    result = ReActRunner(build_llm(provider="openai"), tools=[calculator]).run("What is 21 * 2?")
    assert "42" in str(result["messages"][-1].content)


def test_openrouter_provider_targets_openrouter(monkeypatch):
    """The openrouter provider must not fall back to the OpenAI base URL."""
    from react_loop.llm import build_llm

    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test-key")
    llm = build_llm(provider="openrouter", model="stealth/space-bunny-alpha")

    assert str(llm.openai_api_base) == "https://openrouter.ai/api/v1"
    assert llm.model_name == "stealth/space-bunny-alpha"
    assert llm.openai_api_key.get_secret_value() == "or-test-key"


def test_compatible_provider_uses_its_own_key_variable(tmp_path, monkeypatch):
    """A provider must read its own key variable, not OPENAI_API_KEY."""
    from react_loop.llm import build_llm

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    monkeypatch.setenv("GROQ_API_KEY", "gsk-groq")
    llm = build_llm(provider="groq")

    assert llm.openai_api_key.get_secret_value() == "gsk-groq"
    assert str(llm.openai_api_base) == "https://api.groq.com/openai/v1"


def test_plain_trace_keeps_the_original_format(capsys):
    """Redirected output must keep the user > / assistant> / tool < format."""
    from react_loop.console import print_plain_trace

    script = [
        tool_call("calculator", {"expression": "21 * 2"}),
        AIMessage(content="The answer is 42."),
    ]
    result = ReActRunner(ScriptedChatModel(script=script)).run("21*2?")
    print_plain_trace(result["messages"])

    out = capsys.readouterr().out
    assert "user      > 21*2?" in out
    assert "assistant> call calculator" in out
    assert "tool     < calculator: 42" in out
    assert out.strip().endswith("The answer is 42.")


def test_render_messages_produces_one_panel_per_step():
    """Every message in the loop should become a panel."""
    from rich.console import Console

    from react_loop.console import render_messages

    script = [
        tool_call("calculator", {"expression": "21 * 2"}),
        AIMessage(content="The answer is 42."),
    ]
    result = ReActRunner(ScriptedChatModel(script=script)).run("21*2?")
    console = Console(record=True, width=100, force_terminal=False)
    console.print(render_messages(result["messages"]))
    text = console.export_text()

    assert text.count("user") >= 1
    assert "calculator" in text
    assert "The answer is 42." in text


def test_setup_logging_accepts_a_level_name():
    """A level name from the CLI must configure the logger without raising."""
    import logging

    from react_loop.console import setup_logging

    setup_logging("DEBUG", force_terminal=False)
    assert logging.getLogger("react_loop").level == logging.DEBUG
    setup_logging("WARNING", force_terminal=False)
    assert logging.getLogger("react_loop").level == logging.WARNING


def test_cli_plain_mode_has_no_ansi_codes(capsys):
    """--plain must produce clean text for piping to a file."""
    from react_loop.__main__ import main

    assert main(["--demo", "--plain"]) == 0
    out = capsys.readouterr().out

    assert "\x1b[" not in out
    assert "call calculator" in out


def test_cli_rich_mode_renders_panels(capsys):
    """--rich must use bordered panels."""
    from react_loop.__main__ import main

    assert main(["--demo", "--rich"]) == 0
    out = capsys.readouterr().out

    assert "\u250c" in out or "\u256d" in out  # box-drawing corner
    # Rich shows the call as calculator({...}), without the plain "call " prefix.
    assert "calculator(" in out
    assert "tool \u00b7 calculator" in out


def test_stream_runner_yields_events_in_order():
    """The stream must report tool calls, results, then tokens, then a final event."""
    import asyncio

    from react_loop.demo import DEMO_QUESTION, DEMO_SCRIPT
    from react_loop.streaming import (
        FinalEvent,
        ReActStreamRunner,
    )
    from react_loop.tools import calculator, search_knowledge_base

    runner = ReActStreamRunner(
        ScriptedChatModel(script=list(DEMO_SCRIPT)),
        tools=[calculator, search_knowledge_base],
    )

    async def collect():
        return [event async for event in runner.astream(DEMO_QUESTION)]

    events = asyncio.run(collect())
    kinds = [type(e).__name__ for e in events]

    assert kinds[0] == "ToolCallEvent"
    assert "ToolResultEvent" in kinds
    assert "TokenEvent" in kinds
    assert kinds[-1] == "FinalEvent"
    # Exactly one final event, and it is last.
    assert kinds.count("FinalEvent") == 1
    assert isinstance(events[-1], FinalEvent)


def test_stream_tokens_join_to_the_final_answer():
    """Joined tokens must equal the final answer, with no tool text mixed in."""
    import asyncio

    from react_loop.demo import DEMO_QUESTION, DEMO_SCRIPT
    from react_loop.streaming import ReActStreamRunner, collect_text
    from react_loop.tools import calculator, search_knowledge_base

    runner = ReActStreamRunner(
        ScriptedChatModel(script=list(DEMO_SCRIPT)),
        tools=[calculator, search_knowledge_base],
    )

    async def collect():
        return [event async for event in runner.astream(DEMO_QUESTION)]

    events = asyncio.run(collect())
    final = events[-1]

    assert collect_text(events) == final.answer
    assert "$45" in final.answer
    # Tool output must not leak into the answer.
    assert "refund policy" not in collect_text(events).lower()


def test_stream_emits_several_tokens_not_one_chunk():
    """Word-level streaming means more than one token for a multi-word answer."""
    import asyncio

    from react_loop.streaming import ReActStreamRunner, TokenEvent

    runner = ReActStreamRunner(
        ScriptedChatModel(script=[AIMessage(content="one two three four")]), tools=[]
    )

    async def collect():
        return [event async for event in runner.astream("hello")]

    events = asyncio.run(collect())
    tokens = [e for e in events if isinstance(e, TokenEvent)]

    assert len(tokens) == 4
    assert "".join(t.text for t in tokens) == "one two three four"


def test_stream_and_invoke_give_the_same_answer():
    """Streaming must not change the result compared with the blocking run."""
    import asyncio

    from react_loop.demo import DEMO_SCRIPT
    from react_loop.runner import ReActRunner
    from react_loop.streaming import ReActStreamRunner
    from react_loop.tools import calculator, search_knowledge_base

    tools = [calculator, search_knowledge_base]
    script = list(DEMO_SCRIPT)
    blocked = ReActRunner(ScriptedChatModel(script=script), tools=tools).run("q")

    async def go():
        runner = ReActStreamRunner(ScriptedChatModel(script=script), tools=tools)
        return [event async for event in runner.astream("q")]

    events = asyncio.run(go())

    assert events[-1].answer == blocked["messages"][-1].content


def test_stream_tool_call_args_are_real_dicts():
    """Tool call args must survive the chunk round-trip as a dict."""
    import asyncio

    from react_loop.streaming import ReActStreamRunner, ToolCallEvent

    runner = ReActStreamRunner(
        ScriptedChatModel(script=[tool_call("calculator", {"expression": "21 * 2"})]),
        tools=[calculator],
    )

    async def collect():
        return [event async for event in runner.astream("q")]

    events = asyncio.run(collect())
    call = next(e for e in events if isinstance(e, ToolCallEvent))

    assert call.name == "calculator"
    assert call.args == {"expression": "21 * 2"}


def test_cli_stream_demo_writes_the_answer(capsys):
    """--stream must still put the answer on stdout."""
    from react_loop.__main__ import main

    assert main(["--demo", "--stream", "--plain"]) == 0
    out = capsys.readouterr().out

    assert "$45" in out




def test_compatible_provider_without_key_is_an_error(tmp_path, monkeypatch):
    """A missing key should fail early with a clear message, not at call time."""
    from react_loop.llm import build_llm

    # Run outside the project root so a local .env cannot supply the key.
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("REACT_MODEL", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)

    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        build_llm(provider="openrouter")


def test_provider_model_falls_back_to_env(tmp_path, monkeypatch):
    """REACT_MODEL and LLM_MODEL still override the provider default."""
    from react_loop.llm import build_llm

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test-key")
    monkeypatch.setenv("REACT_MODEL", "stealth/space-bunny-alpha")
    assert build_llm(provider="openrouter").model_name == "stealth/space-bunny-alpha"


def test_env_file_supplies_provider_settings(tmp_path, monkeypatch):
    """build_llm should read provider, model, and key from a .env file."""
    from react_loop.llm import build_llm

    env_file = tmp_path / ".env"
    env_file.write_text(
        "LLM_PROVIDER=openrouter\n"
        "REACT_MODEL=stealth/space-bunny-alpha\n"
        "OPENROUTER_API_KEY=or-from-dotenv\n"
    )
    monkeypatch.chdir(tmp_path)
    for name in ("LLM_PROVIDER", "REACT_MODEL", "LLM_MODEL", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(name, raising=False)

    llm = build_llm()

    assert str(llm.openai_api_base) == "https://openrouter.ai/api/v1"
    assert llm.model_name == "stealth/space-bunny-alpha"
    assert llm.openai_api_key.get_secret_value() == "or-from-dotenv"


def test_real_env_vars_take_precedence_over_env_file(tmp_path, monkeypatch):
    """An exported variable must win over the value in .env."""
    from react_loop.llm import build_llm

    (tmp_path / ".env").write_text("OPENROUTER_API_KEY=or-from-dotenv\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-from-shell")
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")

    assert build_llm(provider="openrouter").openai_api_key.get_secret_value() == "or-from-shell"


def test_load_env_is_safe_without_python_dotenv(monkeypatch):
    """A missing python-dotenv must not break anything."""
    import builtins

    from react_loop import llm as llm_module

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "dotenv":
            raise ImportError("no dotenv")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    llm_module.load_env()



def test_cli_deep_research_uses_research_prompt_and_requested_budget(monkeypatch, capsys):
    from react_loop.__main__ import main

    calls = {}

    monkeypatch.setattr("react_loop.__main__.build_llm", lambda **_kwargs: object())

    def fake_run_react(**kwargs):
        calls.update(kwargs)
        return {
            "messages": [
                HumanMessage(content="Compare databases."),
                AIMessage(content="A structured research report."),
            ]
        }

    monkeypatch.setattr("react_loop.runner.run_react", fake_run_react)

    assert (
        main(
            [
                "--deep-research",
                "--max-steps",
                "7",
                "--no-trace",
                "--plain",
                "Compare databases.",
            ]
        )
        == 0
    )

    assert calls["recursion_limit"] == 7
    assert calls["verbose"] is False
    assert "stock_market_data" in calls["system_prompt"]
    assert "bear/base/bull" in calls["system_prompt"]
    output = capsys.readouterr().out
    assert "A structured research report." in output
    assert "user      >" not in output


def test_cli_deep_research_requires_a_topic(monkeypatch, capsys):
    from react_loop.__main__ import main

    monkeypatch.setattr(
        "react_loop.__main__.build_llm",
        lambda **_kwargs: pytest.fail("must validate the topic before building the model"),
    )

    assert main(["--deep-research", "--plain"]) == 2
    assert "requires a research topic" in capsys.readouterr().err
