"""Offline tests for stock market-data tool behavior."""

import pytest

from react_loop.tools import stock_market_data


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


def test_stock_market_data_is_registered_for_agents():
    from react_loop.tools import ALL_TOOLS

    assert any(tool.name == "stock_market_data" for tool in ALL_TOOLS)


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
