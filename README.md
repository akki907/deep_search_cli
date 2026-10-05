# ReAct loop in LangGraph

A ReAct agent (reason + act) built as an explicit LangGraph state graph. The
loop is small enough to read in one file, and it runs offline so you can see it
work without an API key.

## The loop

Three pieces make up the cycle:

| Piece | Job |
| --- | --- |
| `agent` node | Calls the model. The model either answers or asks for tools. |
| `tools` node | Runs the requested tools and appends the results as `ToolMessage`. |
| `should_continue` | Sends control to `tools` while the model is asking for tools, otherwise ends the run. |

That `agent -> tools -> agent` cycle is the ReAct loop: observe the question,
reason about it, act with a tool, observe the output, and repeat until the model
stops requesting tools.

```mermaid
graph TD;
    __start__([__start__]):::first
    agent(agent)
    tools(tools)
    __end__([__end__]):::last
    __start__ --> agent;
    agent -.-> __end__;
    agent -.-> tools;
    tools --> agent;
```

## Quickstart

```bash
make install   # sync dependencies, including the provider packages
make demo      # run the canned trajectory
```

Without `make`, the same thing is `uv sync --extra openai` then
`uv run python -m react_loop --demo`.

The demo replays a fixed trajectory, so it needs no key:

```
user      > Express shipping costs $15 per order. How much is it for 3 items, and what is the refund window?
assistant> call search_knowledge_base({'query': 'shipping times and refund policy'})
tool     < search_knowledge_base: [refund policy] Refunds are available within 30 days of purchase...
assistant> call calculator({'expression': '15 * 3'})
tool     < calculator: 45
assistant> Express shipping for 3 items costs $45 and arrives in 1-2 business days. Refunds are available within 30 days of purchase.
```

### With a real model

```bash
uv sync --extra openai
export OPENAI_API_KEY=sk-...
uv run python -m react_loop "How much is express shipping for 3 items?"
```

Useful flags: `--provider` (`openai`, `openrouter`, `together`, `groq`,
`ollama`, `anthropic`, `scripted`), `--model`, `--temperature`, `--max-steps`,
`--no-trace`.

### With OpenRouter

Put your settings in a `.env` file in the project root. It is gitignored, and
`.env.example` is the committed template:

```bash
cp .env.example .env
```

```
LLM_PROVIDER=openrouter
REACT_MODEL=stealth/space-bunny-alpha
OPENROUTER_API_KEY=or-...
```

`build_llm` loads that file, then run the loop with no flags:

```bash
uv run python -m react_loop "How much is express shipping for 3 items?"
```

The lookup searches the working directory only, and real environment variables
take precedence over the file, so `export` still overrides `.env` when needed.
To switch models without editing the file:

```bash
uv run python -m react_loop --model stealth/space-bunny-alpha "What is the refund window?"
```

A missing key fails immediately with a message naming the variable, rather than
at the first model call.

| Provider | Key variable | Base URL |
| --- | --- | --- |
| `openai` | `OPENAI_API_KEY` | LangChain default |
| `openrouter` | `OPENROUTER_API_KEY` | `https://openrouter.ai/api/v1` |
| `together` | `TOGETHER_API_KEY` | `https://api.together.xyz/v1` |
| `groq` | `GROQ_API_KEY` | `https://api.groq.com/openai/v1` |
| `ollama` | not required | `http://localhost:11434/v1` |

## Use it in code

```python
from react_loop import ReActRunner, build_llm

runner = ReActRunner(build_llm(provider="openai"))
result = runner.run("What is the refund window?")
print(result["messages"][-1].content)
print(result["steps"])  # how many reasoning steps the loop took
```

Swap in your own tools by passing any LangChain tool:

```python
from langchain_core.tools import tool

@tool
def get_order_status(order_id: str) -> str:
    """Look up the status of an order."""
    return f"Order {order_id} shipped."

runner = ReActRunner(build_llm(), tools=[get_order_status])
```

## Layout

| File | Contents |
| --- | --- |
| `src/react_loop/graph.py` | `build_react_graph` and the `should_continue` router. |
| `src/react_loop/state.py` | `AgentState`: the `messages` list and the `steps` counter. |
| `src/react_loop/llm.py` | The `ReActModel` protocol, the offline `ScriptedChatModel`, and `build_llm`. |
| `src/react_loop/console.py` | Rich panels, the Rich logger, and live stream rendering. |
| `src/react_loop/streaming.py` | `ReActStreamRunner` and the typed stream events. | |
| `src/react_loop/tools.py` | Demo tools: a safe calculator, a help-centre search, current time. |
| `src/react_loop/runner.py` | `ReActRunner`, plus `trace` and `final_answer` helpers. |
| `src/react_loop/demo.py` | The canned offline trajectory used by `--demo`. |

## Notes

- **Step budget.** LangGraph's `recursion_limit` (default 25) stops an agent that
  keeps asking for tools. It raises `GraphRecursionError`. Pass
  `runner.run(q, recursion_limit=8)` to change it.
- **Memory.** Pass a checkpointer to keep a conversation:
  `ReActRunner(llm, checkpointer=InMemorySaver())`, then send a fixed
  `{"configurable": {"thread_id": "..."}}` config on each call.
- **Tool errors.** `ToolNode` returns the error text to the model instead of
  crashing, so the agent can correct itself on the next turn.
- **System prompt.** It is prepended fresh on each agent call, so it never
  accumulates in the message history.
- **Prebuilt option.** LangGraph also ships `create_react_agent`, which wraps this
  same loop. This project builds the nodes and edges by hand so the control flow
  is visible and editable.

## Streaming

Add `--stream` to see the answer arrive token by token instead of all at once:

```bash
uv run python -m react_loop --stream "Why is the sky blue?"
make stream-run Q="Why is the sky blue?"   # same thing
```

Tool calls and results print first, then the answer streams in live. On a
terminal a spinner shows while the model thinks. Redirected output gets plain
text with no cursor control, so it still reads correctly in a file.

Streaming uses `graph.astream` with `stream_mode=["messages", "updates"]`. The
`messages` stream carries tool output as well as answer text, so the runner
filters on the `agent` node; otherwise tool results leak into the answer.

Use it from your own code:

```python
from react_loop import build_llm
from react_loop.streaming import ReActStreamRunner

runner = ReActStreamRunner(build_llm())
async for event in runner.astream("Why is the sky blue?"):
    print(event)
```

Events are plain dataclasses, so you can match on the type:

| Event | Meaning |
| --- | --- |
| `ToolCallEvent` | The model asked for a tool. Has `name`, `args`, `id`. |
| `ToolResultEvent` | A tool finished. Has `name`, `content`, `id`. |
| `TokenEvent` | A fragment of the answer. Has `text`. |
| `FinalEvent` | The run is over. Has `messages`, `answer`, `steps`. |

`collect_text(events)` joins the tokens, which always equals `FinalEvent.answer`.

## Terminal output

The loop prints coloured Rich panels on a terminal: one panel per step, with the
user turn in blue, tool calls in magenta, tool results in yellow, and the final
answer in green as rendered Markdown. Long tool output is clipped.

```bash
uv run python -m react_loop --demo          # panels on a terminal
uv run python -m react_loop --demo --plain  # plain text, no colour
uv run python -m react_loop --demo --rich   # force panels even when piped
```

Rich is used only when stdout is a terminal, so piping to a file or capturing
output in a test still gives the original `user >` / `assistant>` / `tool <`
lines. `NO_COLOR` also turns colour off. Errors go to stderr, so the answer on
stdout stays easy to capture.

Use it from your own code:

```python
from react_loop import ReActRunner, build_llm, print_trace, setup_logging

setup_logging("INFO")
result = ReActRunner(build_llm()).run("What is the refund window?")
print_trace(result["messages"])
```

## Make targets

Run `make` for the full list.

| Target | Job |
| --- | --- |
| `make install` | Sync everything, including the provider extras |
| `make demo` | Run the offline demo, no API key needed |
| `make stream` | Run the offline demo with token streaming |
| `make stream-run Q="..."` | Run a question with streaming |
| `make run Q="..."` | Run a question through the configured provider |
| `make env` | Create `.env` from `.env.example` if missing |
| `make test` | Run the test suite |
| `make lint` / `make fmt` | Check or fix lint rules |
| `make check` | Lint and test |
| `make clean` | Remove caches and build artefacts |

## Development

```bash
make check        # uv run ruff check . && uv run pytest
```

38 offline tests, no network. The skipped one is the live OpenAI test, which
needs `OPENAI_API_KEY`.
