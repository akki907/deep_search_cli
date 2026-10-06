"""Interactive approval memory tests."""

from unittest.mock import patch

from langchain_core.messages import AIMessage

from react_loop.__main__ import main
from react_loop.llm import ScriptedChatModel


def tool_call(call_id: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "calculator",
                "args": {"expression": "2 + 2"},
                "id": call_id,
            }
        ],
    )


def test_approved_tool_is_not_prompted_again_in_same_chat(monkeypatch, capsys):
    """A second request for an approved tool executes without another prompt."""
    inputs = iter(["first question", "a", "second question", "/exit"])
    prompts: list[str] = []

    async def prompt_async(*args, **kwargs):
        prompts.append(str(args[0]) if args else "")
        return next(inputs)

    model = ScriptedChatModel(
        script=[
            tool_call("call-1"),
            AIMessage(content="first answer"),
            tool_call("call-2"),
            AIMessage(content="second answer"),
        ]
    )

    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    with patch("react_loop.__main__.build_llm", return_value=model), patch(
        "prompt_toolkit.PromptSession.prompt_async", side_effect=prompt_async
    ):
        assert main(["--plain"]) == 0

    output = capsys.readouterr().out
    approval_prompts = [prompt for prompt in prompts if "Approve?" in prompt]
    assert len(approval_prompts) == 1
    assert "first answer" in output
    assert "second answer" in output


def test_rich_prompts_are_valid_prompt_toolkit_markup():
    """Rich approval prompts must parse before a terminal input is shown."""
    from prompt_toolkit.formatted_text import HTML, to_formatted_text
    from react_loop.__main__ import _prompt

    rich_prompt = _prompt("Approve? [A]pprove: ", "yellow", True)
    assert isinstance(rich_prompt, HTML)
    assert "Approve? [A]pprove:" in "".join(
        text for _style, text in to_formatted_text(rich_prompt)
    )
    assert _prompt("Approve? [A]pprove: ", "yellow", False) == "Approve? [A]pprove: "
