"""Tools available to the ReAct agent.

The calculator, clock, and help-centre search are offline and side-effect
free. :func:`web_search` and :func:`wikipedia_search` reach the network, so
they are imported lazily and report a failure as text the model can read
rather than raising. Replace any of them with your own ``@tool`` functions.
"""

import ast
import operator
from datetime import UTC, datetime
from typing import Any

from langchain_core.tools import tool

_OPERATORS: dict[type[ast.AST], Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
}


@tool
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression such as ``(18 * 4) / 3``.

    Args:
        expression: The arithmetic expression to evaluate.

    """
    return _safe_eval(expression)


def _safe_eval(expression: str) -> str:
    """Evaluate arithmetic on numbers only. No names, calls, or attributes."""

    def _eval(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
            return _OPERATORS[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPERATORS:
            return _OPERATORS[type(node.op)](_eval(node.operand))
        msg = f"Unsupported expression element: {type(node).__name__}"
        raise ValueError(msg)

    try:
        tree = ast.parse(expression, mode="eval")
        result = _eval(tree)
    except Exception as exc:
        return f"Error: {exc}"
    return str(int(result)) if isinstance(result, float) and result.is_integer() else str(result)


KNOWLEDGE_BASE: dict[str, str] = {
    "refund policy": (
        "Refunds are available within 30 days of purchase. Refunds over "
        "$100 require manager approval and are issued to the original "
        "payment method within 5 business days."
    ),
    "shipping times": (
        "Standard shipping takes 3-5 business days. Express shipping arrives "
        "in 1-2 business days and costs $15."
    ),
    "support hours": (
        "Support is available 24/7 by chat. Email support replies within one "
        "business day."
    ),
}


@tool
def search_knowledge_base(query: str) -> str:
    """Search the product help knowledge base.

    Args:
        query: Free-text question or keywords to look up.

    """
    terms = {word.lower().strip(".,?") for word in query.split()}
    hits = [
        f"[{title}] {body}"
        for title, body in KNOWLEDGE_BASE.items()
        if terms & set(title.lower().replace("-", " ").split())
    ]
    if not hits:
        return "No matching articles found."
    return "\n".join(hits)


@tool
def get_current_time() -> str:
    """Return the current date and time in UTC as an ISO-8601 string."""
    return datetime.now(UTC).isoformat(timespec="seconds")


#: One tool call must not flood the context window, so result counts are capped
#: no matter what the model asks for.
MAX_SEARCH_RESULTS = 5

#: Seconds to wait on a search before giving up.
SEARCH_TIMEOUT = 10

#: Wikipedia asks every client to identify itself. A generic requests User-Agent
#: is answered with HTTP 429 rather than results, so this must be set.
WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "react-loop/0.1 (ReAct CLI demo; https://pypi.org/project/react-loop/)"

#: Characters kept from a result snippet, to bound one tool result.
SNIPPET_CHARS = 300


def _bounded(max_results: int) -> int:
    """Clamp a model-supplied result count into a usable range."""
    return max(1, min(max_results, MAX_SEARCH_RESULTS))


def _oneline(text: Any) -> str:
    """Collapse whitespace so a snippet stays on one line."""
    return " ".join(str(text or "").split())


def _clip(text: str) -> str:
    """Truncate a snippet, marking that it was cut."""
    return text if len(text) <= SNIPPET_CHARS else text[:SNIPPET_CHARS] + "..."


@tool
def web_search(query: str, max_results: int = 3) -> str:
    """Search the web with DuckDuckGo and return titles, links, and snippets.

    Use this for current events, documentation, and anything outside the help
    knowledge base. Prefer wikipedia_search for encyclopedic facts.

    Args:
        query: What to search for.
        max_results: How many results to return, at most 5.

    """
    from ddgs import DDGS

    try:
        # The default "auto" backend mixes in engines that reject generic
        # clients, so the engine is pinned rather than left to chance.
        results = DDGS(timeout=SEARCH_TIMEOUT).text(
            query, max_results=_bounded(max_results), backend="duckduckgo"
        )
    except Exception as exc:
        return f"Error: web search failed: {exc}. Retry once, or answer without it."
    if not results:
        return f"No web results for {query!r}."
    return "\n\n".join(
        "\n".join([
            _oneline(hit.get("title")) or "untitled",
            _oneline(hit.get("href")),
            _clip(_oneline(hit.get("body"))),
        ])
        for hit in results
    )


def _wiki(params: dict[str, str]) -> dict[str, Any]:
    """Call the MediaWiki API and return the parsed body. Raises on failure."""
    import requests

    response = requests.get(
        WIKIPEDIA_API,
        params={"format": "json", "formatversion": "2", **params},
        headers={"User-Agent": USER_AGENT},
        timeout=SEARCH_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()


def _wiki_summary(title: str) -> str:
    """Return the opening extract for one article, or "" when unavailable."""
    try:
        data = _wiki({
            "action": "query",
            "prop": "extracts",
            "exintro": "1",
            "explaintext": "1",
            "redirects": "1",
            "titles": title,
        })
    except Exception:
        return ""
    for page in data.get("query", {}).get("pages", []):
        if not page.get("missing"):
            return _clip(_oneline(page.get("extract")))
    return ""


@tool
def wikipedia_search(query: str, max_results: int = 3) -> str:
    """Look up a topic on Wikipedia and return article titles with summaries.

    Use this for encyclopedic facts: people, places, history, science. Use
    web_search instead for anything time-sensitive.

    Args:
        query: Topic to look up.
        max_results: How many articles to return, at most 5.

    """
    try:
        hits = _wiki({
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": str(_bounded(max_results)),
        }).get("query", {}).get("search", [])
    except Exception as exc:
        return f"Error: Wikipedia search failed: {exc}"
    if not hits:
        return f"No Wikipedia articles for {query!r}."

    blocks = []
    for hit in hits:
        title = _oneline(hit.get("title"))
        summary = _wiki_summary(title)
        blocks.append(f"{title}\n{summary}" if summary else title)
    return "\n\n".join(blocks)


@tool
def wikipedia_summary(title: str) -> str:
    """Return the opening summary of one Wikipedia article.

    Args:
        title: Exact or approximate article title, such as "Ada Lovelace".

    """
    summary = _wiki_summary(title)
    return summary or f"No Wikipedia summary for {title!r}."

@tool
def delegate(agent_name: str, query: str) -> str:
    """Delegate a task to a specialized agent. Use this when a specific persona 
    (e.g. 'researcher', 'coder', 'writer') is better suited for the query.
    
    Args:
        agent_name: The name of the agent to delegate to.
        query: The specific question or task for the agent.
    """
    from react_loop.runner import ReActRunner, run_react
    from react_loop.llm import build_llm

    PERSONAS = {
        "researcher": "You are a world-class research scientist. Provide exhaustive, evidence-backed answers with citations.",
        "coder": "You are a senior software engineer. Provide concise, efficient, and bug-free code implementation.",
        "writer": "You are a professional editor and writer. Focus on clarity, tone, and narrative flow.",
    }

    prompt = PERSONAS.get(agent_name.lower(), "You are a helpful specialized assistant.")
    
    # We run the sub-agent as a one-shot ReAct loop
    result = run_react(
        question=query,
        system_prompt=prompt,
        verbose=False
    )
    from react_loop.runner import final_answer
    return f"[{agent_name}] {final_answer(result['messages'])}"




ALL_TOOLS = [
    calculator,
    search_knowledge_base,
    get_current_time,
    web_search,
    wikipedia_search,
    wikipedia_summary,
    delegate,
]
