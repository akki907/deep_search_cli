"""Offline tests for stock market-data tool behavior."""

import pytest

from react_loop.tools import (
    compare_stocks,
    resolve_stock_symbol,
    stock_fundamentals,
    stock_market_data,
    stock_technicals,
    stock_backtest,
)


class FakeResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self.payload


def chart_payload() -> dict:
    return {
        "chart": {
            "result": [
                {
                    "meta": {
                        "currency": "USD",
                        "exchangeName": "NMS",
                        "regularMarketPrice": 120.0,
                        "previousClose": 110.0,
                        "regularMarketTime": 1704240000,
                    },
                    "timestamp": [1704067200, 1704153600, 1704240000],
                    "indicators": {
                        "quote": [{"close": [100.0, 110.0, 120.0], "volume": [1000, 1100, 1300]}]
                    },
                }
            ]
        }
    }


def test_stock_market_data_returns_bounded_price_summary(monkeypatch):
    seen: dict = {}

    def respond(url, params=None, headers=None, timeout=None):
        seen.update(url=url, params=params, headers=headers, timeout=timeout)
        return FakeResponse(chart_payload())

    monkeypatch.setattr("requests.get", respond)

    out = stock_market_data.invoke({"symbol": "msft", "period": "1y"})

    assert "Symbol: MSFT" in out
    assert "Current/last price: 120.00" in out
    assert "Period change: 20.00%" in out
    assert "Period high: 120.00" in out
    assert "- 2024-01-03: 120.00" in out
    assert "react-loop" in seen["headers"]["User-Agent"]


def test_stock_tools_are_registered_for_agents():
    from react_loop.tools import ALL_TOOLS

    names = {tool.name for tool in ALL_TOOLS}
    assert {
        "stock_market_data",
        "stock_technicals",
        "resolve_stock_symbol",
        "stock_fundamentals",
        "compare_stocks",
        "stock_backtest",
    } <= names


def test_stock_technicals_reports_indicators(monkeypatch):
    monkeypatch.setattr("requests.get", lambda *_args, **_kwargs: FakeResponse(chart_payload()))

    out = stock_technicals.invoke({"symbol": "MSFT", "period": "1y"})

    assert "SMA20: 110.00" in out
    assert "SMA50: 110.00" in out
    assert "RSI14: 100.00" in out
    assert "Maximum drawdown in window: 0.00%" in out

def test_stock_market_data_rejects_invalid_symbol_before_network(monkeypatch):
    monkeypatch.setattr(
        "requests.get",
        lambda *_args, **_kwargs: pytest.fail("invalid symbols must not reach the network"),
    )

    out = stock_market_data.invoke({"symbol": "AAPL/../../secret"})

    assert out == "Error: invalid stock symbol 'AAPL/../../secret'."


def test_stock_market_data_reports_invalid_period():
    out = stock_market_data.invoke({"symbol": "AAPL", "period": "10y"})

    assert out.startswith("Error: invalid period '10y'")
    assert "1y" in out


def test_stock_fundamentals_returns_key_metrics(monkeypatch):
    payload = {
        "quoteSummary": {
            "result": [
                {
                    "price": {
                        "longName": "Example Corp",
                        "marketCap": {"raw": 1_000_000},
                    },
                    "summaryDetail": {
                        "trailingPE": {"raw": 20},
                        "forwardPE": {"raw": 18},
                        "dividendYield": {"raw": 0.01},
                    },
                    "defaultKeyStatistics": {"pegRatio": {"raw": 1.5}},
                    "financialData": {
                        "revenueGrowth": {"raw": 0.12},
                        "profitMargins": {"raw": 0.2},
                        "operatingMargins": {"raw": 0.25},
                        "returnOnEquity": {"raw": 0.3},
                        "debtToEquity": {"raw": 40},
                        "freeCashflow": {"raw": 500_000},
                    },
                    "assetProfile": {"sector": "Technology", "industry": "Software"},
                }
            ]
        }
    }
    monkeypatch.setattr("requests.get", lambda *_args, **_kwargs: FakeResponse(payload))

    out = stock_fundamentals.invoke({"symbol": "EXM"})

    assert "Company: Example Corp" in out
    assert "Trailing P/E: 20.00" in out
    assert "Revenue growth: 12.00%" in out
    assert "Sector: Technology" in out


def test_resolve_stock_symbol_returns_candidates(monkeypatch):
    payload = {
        "quotes": [
            {
                "symbol": "MSFT",
                "shortname": "Microsoft Corporation",
                "exchange": "NMS",
                "quoteType": "EQUITY",
            }
        ]
    }
    monkeypatch.setattr("requests.get", lambda *_args, **_kwargs: FakeResponse(payload))

    out = resolve_stock_symbol.invoke({"query": "Microsoft"})

    assert "MSFT" in out
    assert "Microsoft Corporation" in out


def test_compare_stocks_returns_comparable_table(monkeypatch):
    monkeypatch.setattr("requests.get", lambda *_args, **_kwargs: FakeResponse(chart_payload()))

    out = compare_stocks.invoke({"symbols": "MSFT, AAPL", "period": "1y"})

    assert "Symbol | Last price | Period change" in out
    assert "MSFT | 120.00 | 20.00%" in out
    assert "AAPL | 120.00 | 20.00%" in out


def test_stock_market_data_caches_same_window(monkeypatch):
    calls = 0

    def fake_get(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return FakeResponse(chart_payload())

    monkeypatch.setattr("requests.get", fake_get)

    stock_market_data.invoke({"symbol": "CACHE", "period": "1mo"})
    stock_market_data.invoke({"symbol": "CACHE", "period": "1mo"})

    assert calls == 1


def test_stock_backtest_reports_historical_distribution(monkeypatch):
    monkeypatch.setattr("requests.get", lambda *_args, **_kwargs: FakeResponse(chart_payload()))

    out = stock_backtest.invoke({"symbol": "BACK", "period": "1y", "horizon_days": 1})

    assert "Historical backtest: BACK" in out
    assert "Observations evaluated: 2" in out
    assert "Positive-return frequency: 100.00%" in out
    assert "not a forecast" in out
