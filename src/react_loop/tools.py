"""Tools available to the ReAct agent.

The calculator, clock, and help-centre search are offline and side-effect
free. :func:`web_search`, :func:`wikipedia_search`, and
:func:`stock_market_data` reach the network, so they are imported lazily and
report a failure as text the model can read rather than raising. Replace any
of them with your own ``@tool`` functions.
"""

import ast
import operator
import re
from datetime import UTC, datetime
from statistics import fmean
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


@tool
def python_executor(code: str) -> str:
    """Execute Python code locally and return the output. Use with caution.

    Args:
        code: The Python code to execute.
    """
    import subprocess
    import sys

    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=10,
            input=None,
        )
        output = (result.stdout + result.stderr).strip()
        return output or "Code executed successfully with no output."
    except subprocess.TimeoutExpired:
        return "Error: Code execution timed out after 10 seconds."
    except Exception as e:
        return f"Error: {str(e)}"


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

STOCK_API_URL = "https://query1.finance.yahoo.com/v8/finance/chart"
STOCK_PERIODS = frozenset({"1mo", "3mo", "6mo", "1y", "2y", "5y", "max"})
STOCK_SYMBOL_PATTERN = re.compile(r"[A-Z0-9][A-Z0-9.-]{0,11}")


@tool
def stock_market_data(symbol: str, period: str = "1y") -> str:
    """Fetch a bounded quote and daily-price summary for a stock or ETF.

    This is market data for research and scenario analysis, not a guaranteed
    forecast or investment recommendation. Use web_search separately for
    earnings, filings, news, and business fundamentals.

    Args:
        symbol: Ticker symbol such as ``AAPL`` or ``MSFT``.
        period: History window: ``1mo``, ``3mo``, ``6mo``, ``1y``, ``2y``,
            ``5y``, or ``max``.

    """
    import requests

    ticker = symbol.strip().upper()
    if not STOCK_SYMBOL_PATTERN.fullmatch(ticker):
        return f"Error: invalid stock symbol {symbol!r}."
    if period not in STOCK_PERIODS:
        allowed = ", ".join(sorted(STOCK_PERIODS))
        return f"Error: invalid period {period!r}; choose one of: {allowed}."

    try:
        response = requests.get(
            f"{STOCK_API_URL}/{ticker}",
            params={"range": period, "interval": "1d", "events": "history"},
            headers={"User-Agent": USER_AGENT},
            timeout=SEARCH_TIMEOUT,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        return f"Error: stock data unavailable for {ticker}: {exc}"

    result = (payload.get("chart", {}).get("result") or [None])[0]
    if not result:
        return f"Error: no market data returned for {ticker}."

    meta = result.get("meta") or {}
    timestamps = result.get("timestamp") or []
    quote = ((result.get("indicators") or {}).get("quote") or [{}])[0]
    closes = quote.get("close") or []
    volumes = quote.get("volume") or []
    observations: list[tuple[str, float, int]] = []
    for index, (timestamp, close) in enumerate(zip(timestamps, closes, strict=False)):
        if close is None:
            continue
        try:
            price = float(close)
            date = datetime.fromtimestamp(int(timestamp), UTC).date().isoformat()
        except (TypeError, ValueError, OverflowError):
            continue
        volume = volumes[index] if index < len(volumes) and volumes[index] else 0
        observations.append((date, price, int(volume)))

    if not observations:
        return f"Error: no usable daily prices returned for {ticker}."

    prices = [price for _, price, _ in observations]
    first_price = prices[0]
    last_price = prices[-1]
    period_change = ((last_price / first_price) - 1) * 100 if first_price else 0.0
    previous_close = meta.get("previousClose")
    market_price = meta.get("regularMarketPrice")
    as_of = meta.get("regularMarketTime")
    if as_of:
        try:
            as_of_text = datetime.fromtimestamp(int(as_of), UTC).isoformat()
        except (TypeError, ValueError, OverflowError):
            as_of_text = observations[-1][0]
    else:
        as_of_text = observations[-1][0]
    volume_values = [volume for _, _, volume in observations if volume]
    average_volume = fmean(volume_values) if volume_values else 0.0
    quote_text = f"{float(market_price):.2f}" if market_price is not None else f"{last_price:.2f}"
    previous_text = (
        f"{float(previous_close):.2f}" if previous_close is not None else "unavailable"
    )

    lines = [
        f"Symbol: {ticker}",
        "Data source: Yahoo Finance chart endpoint",
        f"Exchange: {meta.get('exchangeName') or 'unavailable'}",
        f"Currency: {meta.get('currency') or 'unavailable'}",
        f"As of: {as_of_text} UTC",
        f"Current/last price: {quote_text}",
        f"Previous close: {previous_text}",
        f"History window: {period}",
        f"Observations: {len(observations)} daily closes",
        f"Period change: {period_change:.2f}%",
        f"Period high: {max(prices):.2f}",
        f"Period low: {min(prices):.2f}",
        f"Average daily volume: {average_volume:.0f}",
        "Recent daily closes:",
    ]
    lines.extend(f"- {date}: {price:.2f}" for date, price, _ in observations[-10:])
    return "\n".join(lines)


@tool
def web_search(query: str, max_results: int = 3) -> str:
    """Search the web and return bounded titles, links, and snippets.

    DuckDuckGo is tried first. A Google-backed DDGS search is used once when
    the primary backend fails, which handles transient rate limits without
    making the model repeat the same failed call.

    Use this for current events, documentation, and anything outside the help
    knowledge base. Prefer wikipedia_search for encyclopedic facts.

    Args:
        query: What to search for.
        max_results: How many results to return, at most 5.

    """
    from ddgs import DDGS

    searcher = DDGS(timeout=SEARCH_TIMEOUT)
    bounded = _bounded(max_results)
    try:
        results = searcher.text(query, max_results=bounded, backend="duckduckgo")
    except Exception as primary_exc:
        try:
            results = searcher.text(query, max_results=bounded, backend="google")
        except Exception as fallback_exc:
            return (
                "Error: web search failed after trying DuckDuckGo and Google: "
                f"{primary_exc}; fallback failed: {fallback_exc}. "
                "Do not retry the same query; use wikipedia_search or answer with caveats."
            )
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
    python_executor,
    stock_market_data,
    web_search,
    wikipedia_search,
    wikipedia_summary,
    delegate,
]





