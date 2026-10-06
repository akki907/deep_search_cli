"""Pure portfolio and watchlist calculations and renderers."""

from __future__ import annotations

import csv
import io
import json
import math
from datetime import UTC, datetime
from statistics import fmean
from typing import Any


def _drawdown(values: list[float]) -> float:
    peak = values[0] if values else 0.0
    maximum = 0.0
    for value in values:
        peak = max(peak, value)
        if peak:
            maximum = min(maximum, value / peak - 1)
    return maximum


def _volatility(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    returns = [values[index] / values[index - 1] - 1 for index in range(1, len(values)) if values[index - 1]]
    if len(returns) < 2:
        return None
    mean = fmean(returns)
    return math.sqrt(fmean([(value - mean) ** 2 for value in returns])) * math.sqrt(252)


def calculate_portfolio(
    positions: list[dict[str, Any]],
    quotes: dict[str, dict[str, Any]],
    *,
    sectors: dict[str, str | None] | None = None,
    histories: dict[str, list[dict[str, Any]]] | None = None,
    benchmark: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Calculate valuations without treating unavailable quotes as zero."""
    sectors = sectors or {}
    histories = histories or {}
    valuation_time = datetime.now(UTC).isoformat(timespec="seconds")
    valued: list[dict[str, Any]] = []
    total_cost = 0.0
    total_value = 0.0
    available_value = 0.0
    for position in positions:
        symbol = position["symbol"]
        quantity = float(position["quantity"])
        cost_basis = float(position["cost_basis"])
        cost = quantity * cost_basis
        quote = quotes.get(symbol) or {}
        price = quote.get("price")
        market_value = quantity * float(price) if price is not None else None
        gain = market_value - cost if market_value is not None else None
        gain_percent = gain / cost * 100 if gain is not None and cost else None
        total_cost += cost
        if market_value is not None:
            total_value += market_value
            available_value += market_value
        valued.append(
            {
                "symbol": symbol,
                "quantity": quantity,
                "cost_basis": cost_basis,
                "cost": cost,
                "price": price,
                "market_value": market_value,
                "gain_loss": gain,
                "gain_loss_percent": gain_percent,
                "sector": sectors.get(symbol),
                "as_of": quote.get("as_of"),
                "stale": bool(quote.get("stale", False)),
                "source": quote.get("source"),
            }
        )
    allocation = []
    for item in valued:
        allocation.append(
            {
                "symbol": item["symbol"],
                "sector": item["sector"],
                "market_value": item["market_value"],
                "percent": item["market_value"] / available_value * 100
                if item["market_value"] is not None and available_value
                else None,
            }
        )

    history_values: list[float] = []
    dates: list[str] = []
    if histories:
        by_date: dict[str, float] = {}
        for item in valued:
            bars = {bar["date"]: bar["close"] for bar in histories.get(item["symbol"], [])}
            for date, close in bars.items():
                by_date[date] = by_date.get(date, 0.0) + item["quantity"] * close
        dates = sorted(by_date)
        history_values = [by_date[date] for date in dates]

    benchmark_return = None
    if benchmark and len(benchmark) >= 2:
        first = benchmark[0].get("close")
        last = benchmark[-1].get("close")
        if first:
            benchmark_return = (last / first - 1) * 100
    all_prices_available = all(item["market_value"] is not None for item in valued)
    portfolio_return = (
        (total_value / total_cost - 1) * 100
        if total_cost and all_prices_available
        else None
    )
    return {
        "valued_at": valuation_time,
        "positions": valued,
        "totals": {
            "cost": total_cost,
            "market_value": total_value if all_prices_available else None,
            "available_market_value": available_value,
            "gain_loss": total_value - total_cost if all_prices_available else None,
            "return_percent": portfolio_return,
        },
        "allocation": allocation,
        "historical": {
            "dates": dates,
            "volatility_annualized": _volatility(history_values),
            "maximum_drawdown": _drawdown(history_values) if history_values else None,
        },
        "benchmark_return_percent": benchmark_return,
    }


def render_portfolio(report: dict[str, Any], output_format: str = "text") -> str:
    """Render portfolio analytics as text, Markdown, JSON, or CSV."""
    if output_format == "json":
        return json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if output_format == "csv":
        stream = io.StringIO()
        writer = csv.DictWriter(
            stream,
            fieldnames=["symbol", "quantity", "cost_basis", "price", "market_value", "gain_loss", "gain_loss_percent"],
        )
        writer.writeheader()
        for position in report["positions"]:
            writer.writerow({field: position.get(field) for field in writer.fieldnames})
        return stream.getvalue()
    lines = [
        "Portfolio summary",
        f"Valued at: {report['valued_at']}",
        f"Total cost: {report['totals']['cost']:.2f}",
        f"Market value: {report['totals']['market_value'] if report['totals']['market_value'] is not None else 'unavailable'}",
        f"Gain/loss: {report['totals']['gain_loss'] if report['totals']['gain_loss'] is not None else 'unavailable'}",
        "",
        "Symbol | Quantity | Price | Market value | Gain/loss | Return",
        "--- | ---: | ---: | ---: | ---: | ---:",
    ]
    lines.extend(
        f"{position['symbol']} | {position['quantity']:g} | "
        f"{position['price'] if position['price'] is not None else 'unavailable'} | "
        f"{position['market_value'] if position['market_value'] is not None else 'unavailable'} | "
        f"{position['gain_loss'] if position['gain_loss'] is not None else 'unavailable'} | "
        f"{position['gain_loss_percent'] if position['gain_loss_percent'] is not None else 'unavailable'}"
        for position in report["positions"]
    )
    return "\n".join(lines)


def render_watchlist(rows: list[dict[str, Any]], output_format: str = "text") -> str:
    """Render watchlist rows in terminal, Markdown, JSON, or CSV form."""
    fields = ["symbol", "price", "daily_change_percent", "period_change_percent", "rsi", "trend", "freshness", "alert_status"]
    if output_format == "json":
        return json.dumps(rows, indent=2, ensure_ascii=False) + "\n"
    if output_format == "csv":
        stream = io.StringIO()
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in rows)
        return stream.getvalue()
    lines = [
        "Symbol | Price | Daily % | Period % | RSI | Trend | Freshness | Alerts",
        "--- | ---: | ---: | ---: | ---: | --- | --- | ---",
    ]
    lines.extend(
        "{symbol} | {price} | {daily_change_percent} | {period_change_percent} | "
        "{rsi} | {trend} | {freshness} | {alert_status}".format(
            symbol=row.get("symbol", ""),
            price=row.get("price", "unavailable"),
            daily_change_percent=row.get("daily_change_percent", "unavailable"),
            period_change_percent=row.get("period_change_percent", "unavailable"),
            rsi=row.get("rsi", "unavailable"),
            trend=row.get("trend", "unavailable"),
            freshness=row.get("freshness", "unavailable"),
            alert_status=row.get("alert_status", "none"),
        )
        for row in rows
    )
    return "\n".join(lines)
