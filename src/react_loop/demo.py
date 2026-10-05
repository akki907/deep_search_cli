"""A canned ReAct trajectory so the loop can be seen without an API key.

``python -m react_loop --demo`` runs this offline.
"""

from langchain_core.messages import AIMessage

DEMO_QUESTION = (
    "Express shipping costs $15 per order. How much is it for 3 items, "
    "and what is the refund window?"
)

DEMO_SCRIPT = [
    AIMessage(
        content="",
        tool_calls=[
            {
                "name": "search_knowledge_base",
                "args": {"query": "shipping times and refund policy"},
                "id": "call_1",
            }
        ],
    ),
    AIMessage(
        content="",
        tool_calls=[
            {"name": "calculator", "args": {"expression": "15 * 3"}, "id": "call_2"}
        ],
    ),
    AIMessage(
        content=(
            "Express shipping for 3 items costs $45 and arrives in 1-2 business "
            "days. Refunds are available within 30 days of purchase."
        )
    ),
]
