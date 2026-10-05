import asyncio
import uuid
from typing import Any
from collections.abc import Sequence, Callable, AsyncIterator

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from react_loop.graph import build_react_graph
from react_loop.llm import ReActModel, build_llm
from react_loop.tools import ALL_TOOLS
from react_loop.logger import setup_session_logger

class ReActRunner:
    """Runs a compiled ReAct graph and reports the final answer.

    The runner handles the graph execution and supports human-in-the-loop
    interrupts for tool approval.
    """

    def __init__(
        self,
        llm: ReActModel,
        tools: Sequence[Any] | None = None,
        system_prompt: str | None = None,
        checkpointer: Any | None = None,
        name: str = "react_agent",
    ) -> None:
        self.llm = llm
        self.system_prompt = system_prompt
        self.name = name
        
        self.session_id = uuid.uuid4().hex[:8]
        self.logger = setup_session_logger(self.session_id)
        tool_names = [t.name for t in (tools or ALL_TOOLS)]
        self.logger.info(f"Starting session {self.session_id} with tools: {tool_names}")
        
        cp = checkpointer or InMemorySaver()
        self.graph = build_react_graph(
            llm=llm,
            tools=ALL_TOOLS if tools is None else tools,
            system_prompt=system_prompt,
            checkpointer=cp,
            name=name,
        )
        self.checkpointer = cp

    async def run_async(
        self,
        question: str,
        config: dict[str, Any] | None = None,
        approval_callback: Callable[[list[Any]], str | list[Any]] | None = None,
        recursion_limit: int = 25,
    ) -> dict[str, Any]:
        """Run one question with optional human-in-the-loop approval for tool calls.

        Args:
            question: The user message to start from.
            config: LangGraph config (e.g. thread_id).
            approval_callback: A callback called before tool execution.
                Returns 'approve', 'cancel', or a list of modified tool calls.
            recursion_limit: Maximum agent/tool rounds.

        Returns:
            The final graph state.
        """
        if config is None:
            config = {"configurable": {"thread_id": self.session_id}, "recursion_limit": recursion_limit}
        elif "recursion_limit" not in config:
            config["recursion_limit"] = recursion_limit

        start = {"messages": [HumanMessage(content=question)]}
        self.logger.info(f"Session {self.session_id}: User question: {question}")
        state = self.graph.invoke(start, config)

        while True:
            snapshot = self.graph.get_state(config)
            if not snapshot.next or "tools" not in snapshot.next:
                break

            if approval_callback:
                last_msg = snapshot.values["messages"][-1]
                tool_calls = getattr(last_msg, "tool_calls", [])
                
                decision = await approval_callback(tool_calls)
                self.logger.info(f"Session {self.session_id}: Tool approval decision: {decision}")
                
                if decision == "approve":
                    pass
                elif decision == "cancel":
                    cancel_msgs = [
                        ToolMessage(
                            tool_call_id=tc["id"],
                            content="Tool call cancelled by user.",
                        )
                        for tc in tool_calls
                    ]
                    self.graph.update_state(config, {"messages": cancel_msgs})
                elif isinstance(decision, list):
                    new_msg = AIMessage(content=last_msg.content, tool_calls=decision)
                    self.graph.update_state(config, {"messages": [new_msg]})
                else:
                    if tool_calls:
                        self.graph.update_state(config, {"messages": [ToolMessage(tool_call_id=tool_calls[0]["id"], content="Cancelled")]})

            state = self.graph.invoke(None, config)

        return state

    def run(
        self,
        question: str,
        *,
        recursion_limit: int = 25,
        thread_id: str | None = None,
    ) -> dict[str, Any]:
        """Sync run (no HITL)."""
        async def _sync_wrap():
            config = {"configurable": {"thread_id": thread_id or self.session_id}}
            # For sync runs, we auto-approve all tool calls to avoid hanging on interrupts.
            async def auto_approve(tool_calls): return "approve"
            return await self.run_async(question, config=config, approval_callback=auto_approve, recursion_limit=recursion_limit)
        
        return asyncio.run(_sync_wrap())
    async def astream(
        self,
        question: str,
        recursion_limit: int = 25,
        thread_id: str | None = None,
        approval_callback: Callable[[list[Any]], str | list[Any]] | None = None,
    ) -> AsyncIterator[Any]:
        """Delegate streaming to a ReActStreamRunner."""
        from react_loop.streaming import ReActStreamRunner
        streamer = ReActStreamRunner(
            self.llm,
            system_prompt=self.system_prompt,
            checkpointer=self.checkpointer,
            name=self.name,
        )
        async for event in streamer.astream(
            question,
            recursion_limit=recursion_limit,
            thread_id=thread_id or self.session_id,
            approval_callback=approval_callback,
        ):
            yield event



def final_answer(messages: list[AnyMessage]) -> str:
    """Return the text of the last assistant message, or an empty string."""
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            return str(message.content)
    return ""


def trace(messages: list[AnyMessage]) -> list[str]:
    """Render the loop as readable lines, one per message."""
    lines: list[str] = []
    for message in messages:
        if isinstance(message, HumanMessage):
            lines.append(f"user      > {message.content}")
        elif isinstance(message, AIMessage):
            if message.tool_calls:
                for tc in message.tool_calls:
                    lines.append(f"assistant> call {tc['name']}({tc['args']})")
            else:
                lines.append(f"assistant> {message.content}")
        elif isinstance(message, ToolMessage):
            text = message.content
            lines.append(f"tool     < {message.name}: {text[:200]}")
    return lines


def run_react(
    question: str,
    llm: ReActModel | None = None,
    tools: Sequence[Any] | None = None,
    system_prompt: str | None = None,
    *,
    recursion_limit: int = 25,
    verbose: bool = True,
) -> dict[str, Any]:
    """Build a runner, answer one question, and print the trace."""
    runner = ReActRunner(
        llm or build_llm(), tools=tools, system_prompt=system_prompt
    )
    result = runner.run(question, recursion_limit=recursion_limit)
    if verbose:
        for line in trace(result["messages"]):
            print(line)
    return result
