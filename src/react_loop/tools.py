"""Tools available to the ReAct agent.

The calculator, clock, and help-centre search are offline and side-effect
free. :func:`web_search`, :func:`wikipedia_search`, and
:func:`stock_market_data` reach the network, so they are imported lazily and
report a failure as text the model can read rather than raising. Replace any
of them with your own ``@tool`` functions.
"""

import ast
import json
import math
import operator
import re
import time
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
STOCK_CACHE_TTL_SECONDS = 60.0
STOCK_MIN_REQUEST_INTERVAL_SECONDS = 0.1
_stock_cache: dict[
    tuple[str, str], tuple[float, dict[str, Any], list[tuple[str, float, int]]]
] = {}
_stock_last_request_at = 0.0


def _pace_stock_request() -> None:
    """Keep chart requests from being bursty within one process."""
    global _stock_last_request_at
    now = time.monotonic()
    delay = STOCK_MIN_REQUEST_INTERVAL_SECONDS - (now - _stock_last_request_at)
    if delay > 0:
        time.sleep(delay)
    _stock_last_request_at = time.monotonic()

def _fetch_stock_data(
    ticker: str, period: str
) -> tuple[dict[str, Any], list[tuple[str, float, int]], str | None]:
    """Fetch and normalize Yahoo daily observations for one ticker."""
    cache_key = (ticker, period)
    cached = _stock_cache.get(cache_key)
    if cached and time.monotonic() - cached[0] < STOCK_CACHE_TTL_SECONDS:
        return cached[1], cached[2], None

    requested_ticker = ticker
    try:
        payload = _request_stock_payload(ticker, period)
    except Exception as exc:
        resolved = _resolve_yahoo_ticker(ticker)
        if not resolved:
            return {}, [], f"Error: stock data unavailable for {requested_ticker}: {exc}"
        try:
            payload = _request_stock_payload(resolved, period)
        except Exception as resolved_exc:
            return (
                {},
                [],
                f"Error: stock data unavailable for {requested_ticker} "
                f"(tried {resolved}): {resolved_exc}",
            )
        ticker = resolved

    result = (payload.get("chart", {}).get("result") or [None])[0]
    if not result:
        return {}, [], f"Error: no market data returned for {ticker}."

    meta = result.get("meta") or {}
    meta.setdefault("symbol", ticker)
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
        return {}, [], f"Error: no usable daily prices returned for {ticker}."
    cached_value = (time.monotonic(), meta, observations)
    _stock_cache[cache_key] = cached_value
    if ticker != requested_ticker:
        _stock_cache[(ticker, period)] = cached_value
    return meta, observations, None


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
    ticker = _normalize_market_ticker(symbol)
    if not STOCK_SYMBOL_PATTERN.fullmatch(ticker):
        return f"Error: invalid stock symbol {symbol!r}."
    if period not in STOCK_PERIODS:
        allowed = ", ".join(sorted(STOCK_PERIODS))
        return f"Error: invalid period {period!r}; choose one of: {allowed}."

    meta, observations, error = _fetch_stock_data(ticker, period)
    if error:
        return error
    display_ticker = str(meta.get("symbol") or ticker)
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
        f"Symbol: {display_ticker}",
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

def _average(values: list[float]) -> float:
    return fmean(values) if values else 0.0


@tool
def stock_technicals(symbol: str, period: str = "1y") -> str:
    """Calculate transparent technical indicators from daily stock prices."""
    ticker = _normalize_market_ticker(symbol)
    if not STOCK_SYMBOL_PATTERN.fullmatch(ticker):
        return f"Error: invalid stock symbol {symbol!r}."
    if period not in STOCK_PERIODS:
        allowed = ", ".join(sorted(STOCK_PERIODS))
        return f"Error: invalid period {period!r}; choose one of: {allowed}."
    meta, observations, error = _fetch_stock_data(ticker, period)
    if error:
        return error

    prices = [price for _, price, _ in observations]
    returns = [
        (prices[index] / prices[index - 1]) - 1
        for index in range(1, len(prices))
        if prices[index - 1]
    ]
    recent_returns = returns[-20:]
    gains = [value for value in returns[-14:] if value > 0]
    losses = [-value for value in returns[-14:] if value < 0]
    average_gain = _average(gains)
    average_loss = _average(losses)
    rsi = 100.0 if average_loss == 0 and average_gain else (
        100 - (100 / (1 + (average_gain / average_loss)))
        if average_loss
        else 50.0
    )
    peak = prices[0]
    max_drawdown = 0.0
    for price in prices:
        peak = max(peak, price)
        max_drawdown = min(max_drawdown, (price / peak) - 1)
    sma20 = _average(prices[-20:])
    sma50 = _average(prices[-50:])
    volatility = math.sqrt(_average([(value - _average(recent_returns)) ** 2 for value in recent_returns]))
    trend = "bullish" if prices[-1] > sma20 > sma50 else "bearish" if prices[-1] < sma20 < sma50 else "mixed"
    as_of = meta.get("regularMarketTime") or observations[-1][0]
    return "\n".join(
        [
            f"Symbol: {ticker}",
            "Data source: Yahoo Finance chart endpoint",
            f"As of: {as_of}",
            f"History window: {period}",
            f"Last close: {prices[-1]:.2f}",
            f"SMA20: {sma20:.2f}",
            f"SMA50: {sma50:.2f}",
            f"RSI14: {rsi:.2f}",
            f"Annualized 20-day volatility estimate: {volatility * math.sqrt(252) * 100:.2f}%",
            f"Maximum drawdown in window: {max_drawdown * 100:.2f}%",
            f"Technical trend classification: {trend}",
        ]
    )


STOCK_SEARCH_URL = "https://query2.finance.yahoo.com/v1/finance/search"
STOCK_SUMMARY_URL = "https://query1.finance.yahoo.com/v10/finance/quoteSummary"
def _normalize_market_ticker(symbol: str) -> str:
    """Normalize Yahoo exchange aliases such as ``NSE:CEAT`` and ``CEAT.BSE``."""
    ticker = symbol.strip().upper()
    if ":" in ticker:
        exchange, ticker = ticker.split(":", 1)
        suffix = {"NSE": ".NS", "BSE": ".BO"}.get(exchange)
        if suffix:
            return f"{ticker}{suffix}"
    if ticker.endswith(".NSE"):
        return f"{ticker[:-4]}.NS"
    if ticker.endswith(".BSE"):
        return f"{ticker[:-4]}.BO"
    return ticker




def _request_stock_payload(ticker: str, period: str) -> dict[str, Any]:
    """Request one Yahoo chart payload with the shared timeout and pacing."""
    import requests

    _pace_stock_request()
    response = requests.get(
        f"{STOCK_API_URL}/{ticker}",
        params={"range": period, "interval": "1d", "events": "history"},
        headers={"User-Agent": USER_AGENT},
        timeout=SEARCH_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()


def _resolve_yahoo_ticker(ticker: str) -> str | None:
    """Resolve a Yahoo ticker after its direct chart request fails."""
    search_term = ticker
    requested_suffix: str | None = None
    if "." in ticker:
        base, suffix = ticker.rsplit(".", 1)
        if suffix not in {"NS", "BO"}:
            return None
        search_term = base
        requested_suffix = f".{suffix}"
    import requests

    try:
        _pace_stock_request()
        response = requests.get(
            STOCK_SEARCH_URL,
            params={"q": search_term, "quotesCount": 10, "newsCount": 0},
            headers={"User-Agent": USER_AGENT},
            timeout=SEARCH_TIMEOUT,
        )
        response.raise_for_status()
        quotes = response.json().get("quotes") or []
    except Exception:
        return None

    exact = next(
        (
            str(quote["symbol"]).upper()
            for quote in quotes
            if str(quote.get("symbol", "")).upper() == ticker
        ),
        None,
    )
    if exact:
        return exact

    indian = [
        str(quote["symbol"]).upper()
        for quote in quotes
        if str(quote.get("exchange", "")).upper() in {"NSI", "NSE", "BSE", "BOM"}
        and str(quote.get("symbol", "")).upper().endswith((".NS", ".BO"))
    ]
    if requested_suffix:
        indian = [symbol for symbol in indian if symbol.endswith(requested_suffix)]
    matching_indian = next(
        (symbol for symbol in indian if symbol.split(".", 1)[0].startswith(search_term)),
        None,
    )
    if matching_indian:
        return matching_indian
    return next((symbol for symbol in indian if symbol.startswith(f"{search_term}.")), None)


def _yahoo_value(value: Any) -> Any:
    if isinstance(value, dict):
        return value.get("raw", value.get("fmt", value.get("longFmt")))
    return value


def _metric(section: dict[str, Any], key: str, percent: bool = False) -> str:
    value = _yahoo_value(section.get(key))
    if value is None:
        return "unavailable"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{number * 100:.2f}%" if percent else f"{number:.2f}"


@tool
def resolve_stock_symbol(query: str) -> str:
    """Resolve a company name or ticker into candidate symbols and exchanges."""
    import requests

    if not query.strip():
        return "Error: a company name or ticker is required."
    try:
        response = requests.get(
            STOCK_SEARCH_URL,
            params={"q": query.strip(), "quotesCount": 5, "newsCount": 0},
            headers={"User-Agent": USER_AGENT},
            timeout=SEARCH_TIMEOUT,
        )
        response.raise_for_status()
        quotes = response.json().get("quotes") or []
    except Exception as exc:
        return f"Error: stock symbol lookup failed: {exc}"
    if not quotes:
        return f"No stock symbols found for {query!r}."
    lines = [
        "Data source: Yahoo Finance symbol-search endpoint",
        *(
            f"- {quote.get('symbol', 'unknown')}: "
            f"{quote.get('shortname') or quote.get('longname') or 'unknown'} "
            f"({quote.get('exchange') or 'unknown'}, {quote.get('quoteType') or 'unknown'})"
            for quote in quotes
        ),
    ]
    return "\n".join(lines)


@tool
def stock_fundamentals(symbol: str) -> str:
    """Fetch valuation, profitability, balance-sheet, and company metadata."""
    import requests

    ticker = symbol.strip().upper()
    if not STOCK_SYMBOL_PATTERN.fullmatch(ticker):
        return f"Error: invalid stock symbol {symbol!r}."
    try:
        response = requests.get(
            f"{STOCK_SUMMARY_URL}/{ticker}",
            params={
                "modules": ",".join(
                    [
                        "price",
                        "summaryDetail",
                        "defaultKeyStatistics",
                        "financialData",
                        "assetProfile",
                        "calendarEvents",
                    ]
                )
            },
            headers={"User-Agent": USER_AGENT},
            timeout=SEARCH_TIMEOUT,
        )
        response.raise_for_status()
        result = ((response.json().get("quoteSummary") or {}).get("result") or [None])[0]
    except Exception as exc:
        return f"Error: fundamentals unavailable for {ticker}: {exc}"
    if not result:
        return f"Error: no fundamentals returned for {ticker}."

    price = result.get("price") or {}
    detail = result.get("summaryDetail") or {}
    stats = result.get("defaultKeyStatistics") or {}
    financial = result.get("financialData") or {}
    profile = result.get("assetProfile") or {}
    return "\n".join(
        [
            f"Symbol: {ticker}",
            "Data source: Yahoo Finance quote summary",
            f"Company: {_yahoo_value(price.get('longName')) or _yahoo_value(price.get('shortName')) or 'unavailable'}",
            f"Sector: {profile.get('sector') or 'unavailable'}",
            f"Industry: {profile.get('industry') or 'unavailable'}",
            f"Market cap: {_metric(price, 'marketCap')}",
            f"Trailing P/E: {_metric(detail, 'trailingPE')}",
            f"Forward P/E: {_metric(detail, 'forwardPE')}",
            f"PEG ratio: {_metric(stats, 'pegRatio')}",
            f"Revenue growth: {_metric(financial, 'revenueGrowth', percent=True)}",
            f"Profit margin: {_metric(financial, 'profitMargins', percent=True)}",
            f"Operating margin: {_metric(financial, 'operatingMargins', percent=True)}",
            f"Return on equity: {_metric(financial, 'returnOnEquity', percent=True)}",
            f"Debt/equity: {_metric(financial, 'debtToEquity')}",
            f"Free cash flow: {_metric(financial, 'freeCashflow')}",
            f"Dividend yield: {_metric(detail, 'dividendYield', percent=True)}",
        ]
    )


@tool
def compare_stocks(symbols: str, period: str = "1y") -> str:
    """Compare up to six stock or ETF symbols using the same market window."""
    requested = [
        _normalize_market_ticker(part)
        for part in re.split(r"[,\s]+", symbols)
        if part.strip()
    ]
    if not requested:
        return "Error: provide at least one stock symbol."
    if len(requested) > 6:
        return "Error: compare_stocks accepts at most six symbols."
    if period not in STOCK_PERIODS:
        allowed = ", ".join(sorted(STOCK_PERIODS))
        return f"Error: invalid period {period!r}; choose one of: {allowed}."
    rows = [
        "Data source: Yahoo Finance chart endpoint",
        "",
        "Symbol | Last price | Period change | High | Low",
        "--- | ---: | ---: | ---: | ---:",
    ]
    for ticker in requested:
        if not STOCK_SYMBOL_PATTERN.fullmatch(ticker):
            rows.append(f"{ticker} | invalid symbol | - | - | -")
            continue
        _meta, observations, error = _fetch_stock_data(ticker, period)
        if error:
            rows.append(f"{ticker} | unavailable | - | - | -")
            continue
        prices = [price for _, price, _ in observations]
        change = ((prices[-1] / prices[0]) - 1) * 100 if prices[0] else 0.0
        rows.append(f"{ticker} | {prices[-1]:.2f} | {change:.2f}% | {max(prices):.2f} | {min(prices):.2f}")
    return "\n".join(rows)


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * weight


@tool
def stock_backtest(
    symbol: str,
    period: str = "5y",
    horizon_days: int = 60,
    benchmark_symbol: str = "",
    transaction_cost_bps: float = 0.0,
    slippage_bps: float = 0.0,
) -> str:
    """Run a time-ordered historical forward-return evaluation."""
    ticker = _normalize_market_ticker(symbol)
    if not STOCK_SYMBOL_PATTERN.fullmatch(ticker):
        return f"Error: invalid stock symbol {symbol!r}."
    if period not in STOCK_PERIODS:
        allowed = ", ".join(sorted(STOCK_PERIODS))
        return f"Error: invalid period {period!r}; choose one of: {allowed}."
    if not 1 <= horizon_days <= 756:
        return "Error: horizon_days must be between 1 and 756."
    if transaction_cost_bps < 0 or slippage_bps < 0:
        return "Error: transaction_cost_bps and slippage_bps must be non-negative."

    _, observations, error = _fetch_stock_data(ticker, period)
    if error:
        return error
    prices = [price for _, price, _ in observations]
    outcomes: list[float] = []
    adverse: list[float] = []
    favorable: list[float] = []
    for index in range(len(prices) - horizon_days):
        start = prices[index]
        if not start:
            continue
        path = [price / start - 1 for price in prices[index + 1 : index + horizon_days + 1]]
        outcomes.append(path[-1] * 100)
        adverse.append(min(path) * 100)
        favorable.append(max(path) * 100)
    if not outcomes:
        return (
            f"Error: {ticker} has fewer than {horizon_days + 1} usable observations "
            f"for a {horizon_days}-trading-day backtest."
        )

    round_trip_cost = 2 * (transaction_cost_bps + slippage_bps) / 100
    net_outcomes = [value - round_trip_cost for value in outcomes]
    positive = sum(outcome > 0 for outcome in net_outcomes) / len(net_outcomes) * 100
    lines = [
        f"Historical backtest: {ticker}",
        "Data source: Yahoo Finance chart endpoint",
        "Evaluation: time-ordered walk-forward forward returns",
        f"Window: {period}; horizon: {horizon_days} trading days",
        f"Observations evaluated: {len(net_outcomes)}",
        f"Transaction cost: {transaction_cost_bps:.2f} bps per side",
        f"Slippage: {slippage_bps:.2f} bps per side",
        f"10th percentile net return: {_percentile(net_outcomes, 0.10):.2f}%",
        f"Median net return: {_percentile(net_outcomes, 0.50):.2f}%",
        f"90th percentile net return: {_percentile(net_outcomes, 0.90):.2f}%",
        f"Positive-return frequency: {positive:.2f}%",
        f"Positive-return frequency after costs: {positive:.2f}%",
        f"Maximum adverse excursion: {_percentile(adverse, 0.50):.2f}%",
        f"Maximum favorable excursion: {_percentile(favorable, 0.50):.2f}%",
    ]
    if len(net_outcomes) < 30:
        lines.append("Warning: low sample size; results are unstable and descriptive only.")

    benchmark = _normalize_market_ticker(benchmark_symbol)
    if benchmark:
        if not STOCK_SYMBOL_PATTERN.fullmatch(benchmark):
            lines.append(f"Benchmark warning: invalid symbol {benchmark_symbol!r}.")
        else:
            _, benchmark_observations, benchmark_error = _fetch_stock_data(benchmark, period)
            benchmark_prices = [price for _, price, _ in benchmark_observations]
            if benchmark_error or len(benchmark_prices) <= horizon_days:
                lines.append(f"Benchmark warning: unavailable for {benchmark}.")
            else:
                benchmark_returns = [
                    benchmark_prices[index + horizon_days] / benchmark_prices[index] - 1
                    for index in range(len(benchmark_prices) - horizon_days)
                    if benchmark_prices[index]
                ]
                lines.append(
                    f"Benchmark {benchmark} median return: "
                    f"{_percentile([value * 100 for value in benchmark_returns], 0.50):.2f}%"
                )
    lines.append("Interpretation: historical distribution only; not a forecast or guarantee.")
    return "\n".join(lines)
 


@tool
def evaluate_forecast_probabilities(forecasts_json: str, bin_count: int = 5) -> str:
    """Calculate Brier score and calibration bins for binary forecasts."""
    try:
        payload = json.loads(forecasts_json)
        rows = payload.get("predictions", payload) if isinstance(payload, dict) else payload
        from react_loop.forecast_eval import ForecastObservation, brier_score, calibration_bins

        observations = [
            ForecastObservation(float(row["probability"]), int(row["outcome"]))
            for row in rows
        ]
        if any(item.outcome not in {0, 1} or not 0 <= item.probability <= 1 for item in observations):
            raise ValueError("probability must be 0..1 and outcome must be 0 or 1")
        score = brier_score(observations)
        bins = calibration_bins(observations, bin_count)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return f"Error: invalid forecast observations: {exc}"
    return json.dumps(
        {
            "brier_score": score,
            "sample_size": len(observations),
            "calibration_bins": bins,
        },
        indent=2,
    )


@tool
def sec_filings(symbol: str, filing_type: str = "10-K", limit: int = 5) -> str:
    """Return recent SEC filings with accession numbers and source URLs."""
    ticker = symbol.strip().upper()
    if not STOCK_SYMBOL_PATTERN.fullmatch(ticker):
        return f"Error: invalid stock symbol {symbol!r}."
    try:
        from react_loop.sec_data import sec_filings_data

        filings = sec_filings_data(ticker, filing_type, limit)
    except Exception as exc:
        return f"Error: SEC filings unavailable for {ticker}: {exc}"
    if not filings:
        return f"No {filing_type.upper()} filings found for {ticker}."
    lines = [f"SEC filings for {ticker}", "Source: SEC EDGAR"]
    for filing in filings:
        lines.extend(
            [
                f"- {filing['form']} filed {filing['filing_date']} "
                f"(report date {filing['report_date'] or 'unavailable'})",
                f"  Accession: {filing['accession_number']}",
                f"  Company: {filing['company']}",
                f"  URL: {filing['url']}",
                f"  Retrieved: {filing['retrieved_at']}",
            ]
        )
    return "\n".join(lines)


@tool
def earnings_history(symbol: str) -> str:
    """Return reported versus estimated EPS history for a symbol."""
    ticker = symbol.strip().upper()
    if not STOCK_SYMBOL_PATTERN.fullmatch(ticker):
        return f"Error: invalid stock symbol {symbol!r}."
    try:
        from react_loop.sec_data import earnings_history_data

        rows = earnings_history_data(ticker)
    except Exception as exc:
        return f"Error: earnings history unavailable for {ticker}: {exc}"
    if not rows:
        return f"No earnings history returned for {ticker}."
    lines = [
        f"Earnings history: {ticker}",
        "Source: Yahoo Finance earnings history endpoint",
        "Quarter | Actual EPS | Estimate EPS | Surprise | Surprise %",
        "--- | ---: | ---: | ---: | ---:",
    ]
    for row in rows:
        lines.append(
            f"{row['quarter'] or 'unavailable'} | {row['actual_eps']!s} | "
            f"{row['estimate_eps']!s} | {row['surprise']!s} | "
            f"{row['surprise_percent']!s}"
        )
    lines.append(f"Source URL: {rows[0]['source_url']}")
    return "\n".join(lines)


@tool
def earnings_calendar(symbol: str) -> str:
    """Return upcoming provider-supplied earnings dates."""
    ticker = symbol.strip().upper()
    if not STOCK_SYMBOL_PATTERN.fullmatch(ticker):
        return f"Error: invalid stock symbol {symbol!r}."
    try:
        from react_loop.sec_data import earnings_calendar_data

        events = earnings_calendar_data(ticker)
    except Exception as exc:
        return f"Error: earnings calendar unavailable for {ticker}: {exc}"
    if not events:
        return f"No upcoming earnings dates returned for {ticker}."
    lines = [f"Earnings calendar: {ticker}", "Date | Status | Confidence", "--- | --- | ---"]
    lines.extend(
        f"{event['expected_date'] or 'unknown'} | {event['status']} | {event['confidence']}"
        for event in events
    )
    lines.append(f"Source URL: {events[0]['source_url']}")
    return "\n".join(lines)


@tool
def market_calendar(
    symbol: str,
    filing_limit: int = 5,
    days: int = 365,
    future_only: bool = False,
) -> str:
    """Combine upcoming earnings and recent SEC catalyst events."""
    ticker = symbol.strip().upper()
    if days < 1:
        return "Error: days must be positive."
    if not STOCK_SYMBOL_PATTERN.fullmatch(ticker):
        return f"Error: invalid stock symbol {symbol!r}."
    from react_loop.sec_data import earnings_calendar_data, sec_filings_data

    events: list[dict[str, Any]] = []
    errors: list[str] = []
    try:
        events.extend(earnings_calendar_data(ticker))
    except Exception as exc:
        errors.append(f"earnings: {exc}")
    for form in ("8-K", "10-Q"):
        try:
            filings = sec_filings_data(ticker, form, filing_limit)
        except Exception as exc:
            errors.append(f"{form}: {exc}")
            continue
        events.extend(
            {
                "expected_date": filing["filing_date"],
                "event_type": filing["form"],
                "status": "confirmed",
                "confidence": "SEC filing",
                "title": f"{filing['company']} {filing['form']}",
                "source_url": filing["url"],
            }
            for filing in filings
        )
    events.sort(key=lambda event: event["expected_date"] or "9999-99-99")
    if future_only:
        today = datetime.now(UTC).date()
        cutoff = today.fromordinal(today.toordinal() + days)
        filtered: list[dict[str, Any]] = []
        for event in events:
            try:
                event_date = datetime.fromisoformat(event["expected_date"]).date()
            except (TypeError, ValueError):
                continue
            if today <= event_date <= cutoff:
                filtered.append(event)
        events = filtered
    if not events and errors:
        return f"Error: catalyst calendar unavailable for {ticker}: {'; '.join(errors)}"
    lines = [
        f"Catalyst calendar: {ticker}",
        "Date | Event | Status | Confidence | Source",
        "--- | --- | --- | --- | ---",
    ]
    lines.extend(
        f"{event['expected_date'] or 'unknown'} | {event['title']} | "
        f"{event['status']} | {event['confidence']} | {event.get('source_url', '')}"
        for event in events
    )
    if errors:
        lines.append(f"Warnings: {'; '.join(errors)}")
    return "\n".join(lines)


@tool
def market_cache_status(symbol: str = "") -> str:
    """Show persistent market-cache entries and freshness metadata."""
    from react_loop.market_data import MarketDataCache

    entries = MarketDataCache().entries(symbol.strip().upper() or None)
    if not entries:
        return "Market cache is empty."
    lines = ["Cache key | Symbol | Type | Period | Provider | Fetched | Fresh | Data date"]
    lines.append("--- | --- | --- | --- | --- | --- | --- | ---")
    lines.extend(
        f"{entry['cache_key']} | {entry['symbol']} | {entry['data_type']} | "
        f"{entry['period']} | {entry['provider']} | {entry['fetched_at']} | "
        f"{'yes' if entry['fresh'] else 'stale'} | {entry['data_at'] or 'unknown'}"
        for entry in entries
    )
    return "\n".join(lines)


@tool
def clear_market_cache(symbol: str = "") -> str:
    """Clear all persistent market-cache entries or one symbol's entries."""
    from react_loop.market_data import MarketDataCache

    ticker = symbol.strip().upper() or None
    removed = MarketDataCache().clear(ticker)
    return f"Cleared {removed} market-cache entr{'y' if removed == 1 else 'ies'}."


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
    stock_technicals,
    resolve_stock_symbol,
    stock_fundamentals,
    compare_stocks,
    stock_backtest,
    evaluate_forecast_probabilities,
    sec_filings,
    earnings_history,
    earnings_calendar,
    market_calendar,
    market_cache_status,
    clear_market_cache,
    wikipedia_search,
    wikipedia_summary,
    delegate,
]





