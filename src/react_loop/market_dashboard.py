"""Market-state dashboard orchestration over normalized provider data."""

from __future__ import annotations

from typing import Any

from react_loop.market_data import CachedMarketDataService, MarketDataError
from react_loop.market_state import MarketStateManager
from react_loop.portfolio import calculate_portfolio, render_portfolio, render_watchlist


def _technical_snapshot(closes: list[float]) -> tuple[float | None, str]:
    if not closes:
        return None, "unavailable"
    returns = [closes[index] / closes[index - 1] - 1 for index in range(1, len(closes)) if closes[index - 1]]
    gains = [value for value in returns[-14:] if value > 0]
    losses = [-value for value in returns[-14:] if value < 0]
    average_gain = sum(gains) / len(gains) if gains else 0.0
    average_loss = sum(losses) / len(losses) if losses else 0.0
    rsi = 100.0 if average_loss == 0 and average_gain else 50.0
    if average_loss:
        rsi = 100 - 100 / (1 + average_gain / average_loss)
    sma20 = sum(closes[-20:]) / min(20, len(closes))
    sma50 = sum(closes[-50:]) / min(50, len(closes))
    trend = "bullish" if closes[-1] > sma20 > sma50 else "bearish" if closes[-1] < sma20 < sma50 else "mixed"
    return rsi, trend


def build_portfolio_report(
    manager: MarketStateManager,
    service: CachedMarketDataService,
    *,
    period: str = "1y",
    benchmark_symbol: str = "",
) -> dict[str, Any]:
    """Fetch portfolio inputs and calculate a timestamped analytics report."""
    positions = manager.portfolio()
    quotes: dict[str, dict[str, Any]] = {}
    histories: dict[str, list[dict[str, Any]]] = {}
    sectors: dict[str, str | None] = {}
    for position in positions:
        symbol = position["symbol"]
        try:
            quote = service.quote(symbol)
            quotes[symbol] = {
                "price": quote.value.price,
                "as_of": quote.value.as_of,
                "stale": quote.stale,
                "source": quote.source.url or quote.source.provider,
            }
        except MarketDataError:
            quotes[symbol] = {"price": None, "stale": False}
        try:
            fundamentals = service.fundamentals(symbol)
            sectors[symbol] = fundamentals.value.sector
        except MarketDataError:
            sectors[symbol] = None
        try:
            history = service.history(symbol, period)
            histories[symbol] = [
                {"date": bar.date, "close": bar.close, "volume": bar.volume}
                for bar in history.value
            ]
        except MarketDataError:
            histories[symbol] = []
    benchmark = None
    if benchmark_symbol:
        try:
            benchmark = [
                {"date": bar.date, "close": bar.close}
                for bar in service.history(benchmark_symbol, period).value
            ]
        except MarketDataError:
            benchmark = None
    return calculate_portfolio(
        positions,
        quotes,
        sectors=sectors,
        histories=histories,
        benchmark=benchmark,
    )


def build_watchlist_rows(
    manager: MarketStateManager,
    service: CachedMarketDataService,
    *,
    period: str = "1y",
) -> list[dict[str, Any]]:
    """Build resilient dashboard rows; one provider failure does not abort others."""
    rows: list[dict[str, Any]] = []
    alerts = manager.alerts(enabled_only=True)
    symbols = manager.watchlist()
    for symbol in symbols:
        row: dict[str, Any] = {
            "symbol": symbol,
            "freshness": "unavailable",
            "alert_status": "configured" if any(item["symbol"] == symbol for item in alerts) else "none",
        }
        try:
            quote = service.quote(symbol)
            current = quote.value.price
            row.update(
                {
                    "price": f"{current:.2f}" if current is not None else "unavailable",
                    "freshness": "stale" if quote.stale else quote.value.as_of,
                }
            )
            if current is not None and quote.value.previous_close:
                row["daily_change_percent"] = f"{(current / quote.value.previous_close - 1) * 100:.2f}%"
        except MarketDataError as exc:
            row["error"] = str(exc)
        try:
            history = service.history(symbol, period)
            closes = [bar.close for bar in history.value]
            if closes:
                row["period_change_percent"] = f"{(closes[-1] / closes[0] - 1) * 100:.2f}%"
                rsi, trend = _technical_snapshot(closes)
                row["rsi"] = f"{rsi:.2f}" if rsi is not None else "unavailable"
                row["trend"] = trend
        except MarketDataError as exc:
            row["history_error"] = str(exc)
        rows.append(row)
    return rows


def portfolio_output(manager: MarketStateManager, service: CachedMarketDataService, output_format: str = "text") -> str:
    """Build and render the standard portfolio report."""
    return render_portfolio(build_portfolio_report(manager, service), output_format)


def watchlist_output(manager: MarketStateManager, service: CachedMarketDataService, output_format: str = "text") -> str:
    """Build and render the standard watchlist dashboard."""
    return render_watchlist(build_watchlist_rows(manager, service), output_format)
