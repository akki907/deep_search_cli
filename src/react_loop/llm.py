"""LLM plumbing for the ReAct agent.

The agent node only needs two things from a model: ``bind_tools`` and
``invoke``. Anything satisfying :class:`ReActModel` can drive the graph,
which keeps the tests free of network calls.
"""

import json
import os
import re
from collections.abc import Callable, Iterator, Sequence
from typing import Any, Protocol, runtime_checkable

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, AnyMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import Field

Script = Sequence[AIMessage] | Callable[[list[AnyMessage]], AIMessage]


def load_env() -> None:
    """Load a ``.env`` file into ``os.environ`` if python-dotenv is available.

    Real environment variables win: ``load_dotenv`` does not overwrite what is
    already set, so an exported variable always takes precedence over the file.
    A missing ``python-dotenv`` is not an error, since the variables may be set
    in the shell instead.
    """
    try:
        from dotenv import find_dotenv, load_dotenv
    except ImportError:
        return
    # Search from the working directory, not from this file's location, so the
    # project root is the only place a .env is picked up.
    path = find_dotenv(usecwd=True)
    if path:
        load_dotenv(path)

#: OpenAI-compatible endpoints. Each entry maps a provider name to the base URL,
#: the environment variable holding its key, and a fallback model. Plain
#: ``openai`` keeps LangChain's own default base URL and reads ``OPENAI_API_KEY``.
COMPATIBLE_PROVIDERS: dict[str, tuple[str | None, str, str]] = {
    "openai": (None, "OPENAI_API_KEY", "gpt-4o-mini"),
    "together": (
        "https://api.together.xyz/v1",
        "TOGETHER_API_KEY",
        "meta-llama/Llama-3-8b-chat-hf",
    ),
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY", "llama-3.3-70b-versatile"),
    "ollama": ("http://localhost:11434/v1", "OLLAMA_API_KEY", "llama3.1"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", "stealth/space-bunny-alpha"),
}


@runtime_checkable
class ReActModel(Protocol):
    """Minimal model surface the agent node relies on."""

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "ReActModel":
        """Return a copy of the model that can request the given tools."""

    def invoke(self, messages: Sequence[BaseMessage], **kwargs: Any) -> AnyMessage:
        """Return the assistant message for the given conversation."""




def _split_words(text: str) -> list[str]:
    """Split text into stream chunks, keeping the trailing space on each word.

    Joining the result reproduces the input exactly, which matters because the
    streamed and non-streamed paths must give the same final answer.
    """
    return re.findall(r"\S+\s*", text)


class ScriptedChatModel(BaseChatModel):
    """Offline chat model that replays a fixed script of assistant turns.

    Each call returns the next scripted :class:`AIMessage`, so a test can
    describe an exact ReAct trajectory (tool call, tool call, final answer)
    and assert on it. Once the script is exhausted the model says so, which
    keeps a runaway loop from hanging.

    Args:
        script: Turns to replay, or a callable that builds a turn from the
            messages it receives.

    """

    script: Script = ()
    calls: list[list[AnyMessage]] = Field(default_factory=list)
    tools: list[Any] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "ScriptedChatModel":
        """Record the tools. The script already decides which ones get called."""
        self.tools = list(tools)
        return self

    def _next_turn(self, messages: list[BaseMessage]) -> AIMessage:
        """Record the call and return the next scripted turn."""
        self.calls.append(list(messages))
        index = len(self.calls) - 1
        if callable(self.script):
            message = self.script(messages)
        elif index < len(self.script):
            message = self.script[index]
        else:
            message = AIMessage(content="[script exhausted]")
        return message

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Return the next scripted turn and record the messages received."""
        return ChatResult(generations=[ChatGeneration(message=self._next_turn(messages))])

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Yield the next scripted turn one word at a time.

        Lets the offline demo exercise the streaming path with no network, and
        keeps the joined text identical to the non-streaming result.
        """
        message = self._next_turn(messages)
        if message.tool_calls:
            # A tool call must arrive whole, since the graph needs name and args.
            # Args must be a JSON string here; LangChain parses them back to a
            # dict when the chunks are merged.
            yield ChatGenerationChunk(
                message=AIMessageChunk(
                    content="",
                    tool_call_chunks=[
                        {
                            "name": call["name"],
                            "args": json.dumps(call["args"]),
                            "id": call["id"],
                            "index": index,
                        }
                        for index, call in enumerate(message.tool_calls)
                    ],
                )
            )
            return
        for word in _split_words(str(message.content)):
            yield ChatGenerationChunk(message=AIMessageChunk(content=word))


def resolve_target(
    model: str | None = None, provider: str | None = None
) -> tuple[str, str | None]:
    """Resolve the provider name and model id from flags and the environment.

    Split out of :func:`build_llm` so the CLI can label a session with the
    provider and model it actually used. Call :func:`load_env` first.

    Args:
        model: Model name from the command line, or None to read the env.
        provider: Provider name from the command line, or None to read the env.

    Returns:
        The provider name, lowercased, and the model id if one was given.

    """
    resolved = (provider or os.getenv("LLM_PROVIDER") or "ollama").lower()
    return resolved, model or os.getenv("REACT_MODEL") or os.getenv("LLM_MODEL")


def _build_model(
    provider: str,
    model: str | None = None,
    temperature: float = 0.0,
) -> ReActModel:
    """Helper to build a single model from provider and model id."""
    if provider == "scripted":
        return ScriptedChatModel()
    if provider in COMPATIBLE_PROVIDERS:
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            msg = "Install the extra first: uv sync --extra openai"
            raise RuntimeError(msg) from exc
        base_url, key_var, default_model = COMPATIBLE_PROVIDERS[provider]
        api_key = os.getenv(key_var)
        if not api_key and provider != "ollama":
            msg = f"{provider} needs an API key: set {key_var}"
            raise ValueError(msg)
        kwargs: dict[str, Any] = {
            "model": model or default_model,
            "temperature": temperature,
        }
        if base_url:
            kwargs["base_url"] = base_url
        if api_key:
            kwargs["api_key"] = api_key
        elif provider == "ollama":
            kwargs["api_key"] = "ollama"
        return ChatOpenAI(**kwargs)

    if provider == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as exc:
            msg = "Install the extra first: uv sync --extra anthropic"
            raise RuntimeError(msg) from exc
        return ChatAnthropic(model=model or "claude-sonnet-4-5", temperature=temperature)
    msg = f"Unknown provider: {provider!r}"
    raise ValueError(msg)


def build_llm(
    model: str | None = None,
    temperature: float = 0.0,
    provider: str | None = None,
) -> ReActModel:
    """Create a chat model from the environment.

    ``provider`` is ``openai``, ``anthropic``, ``scripted``, or one of the
    OpenAI-compatible endpoints in :data:`COMPATIBLE_PROVIDERS`
    (``together``, ``groq``, ``ollama``). With no provider it
    is read from ``LLM_PROVIDER``, defaulting to ``ollama``. ``model`` falls
    back to ``REACT_MODEL`` or ``LLM_MODEL``, then to the provider default.

    For a compatible endpoint the key is read from that provider's own
    variable, such as ``TOGETHER_API_KEY`` for ``together``. A missing key
    is a :class:`ValueError` rather than a late failure at call time.

    Args:
        model: Model name, such as ``gpt-4o-mini`` or
            ``meta-llama/Llama-3-8b-chat-hf`` for Together.
        temperature: Sampling temperature.
        provider: Which provider to build.

    Returns:
        A model that satisfies :class:`ReActModel`.

    Raises:
        RuntimeError: If the provider package is not installed.
        ValueError: If the provider name is unknown or its key is missing.
    """
    load_env()
    provider, model = resolve_target(model, provider)
    return _build_model(provider, model, temperature)
