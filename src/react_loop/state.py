"""Graph state for the ReAct agent."""

from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages


def _add_steps(left: int | None, right: int | None) -> int:
    """Reducer that totals the ``steps`` counter across updates."""
    return (left or 0) + (right or 0)


class AgentState(TypedDict):
    """State threaded through the ReAct loop.

    ``messages`` is the conversation so far. ``add_messages`` appends new
    messages and merges updates by message id, so a node can return only
    what it produced.
    """

    messages: Annotated[list[AnyMessage], add_messages]
    # Bumped by the agent node on every pass, so callers can see how many
    # reasoning steps the loop took.
    steps: Annotated[int, _add_steps]


def last_message(state: AgentState) -> Any:
    """Return the most recent message in the state."""
    return state["messages"][-1]
