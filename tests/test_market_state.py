"""Tests for persistent portfolio, watchlist, and alert state."""

from react_loop.market_state import MarketStateManager


def test_market_state_persists_watchlist_portfolio_and_alerts(tmp_path):
    manager = MarketStateManager(str(tmp_path / "market.db"))

    assert manager.add_watch("aapl") == "AAPL"
    assert manager.watchlist() == ["AAPL"]
    assert manager.set_position("MSFT", 2, 300) == "MSFT"
    assert manager.portfolio() == [{"symbol": "MSFT", "quantity": 2.0, "cost_basis": 300.0}]
    assert manager.add_alert("AAPL", "below", 150) == 1
    assert manager.alerts()[0]["operator"] == "below"

    reopened = MarketStateManager(str(tmp_path / "market.db"))
    assert reopened.watchlist() == ["AAPL"]
    assert reopened.portfolio()[0]["symbol"] == "MSFT"
    assert reopened.alerts()[0]["threshold"] == 150.0


def test_market_state_rejects_invalid_values(tmp_path):
    manager = MarketStateManager(str(tmp_path / "market.db"))

    for operation in (
        lambda: manager.add_watch("AAPL/secret"),
        lambda: manager.set_position("AAPL", -1, 100),
        lambda: manager.add_alert("AAPL", "equals", 100),
    ):
        try:
            operation()
        except ValueError:
            pass
        else:
            raise AssertionError("invalid market state must raise ValueError")


def test_market_state_evaluates_triggered_alerts(tmp_path, monkeypatch):
    payload = {
        "chart": {
            "result": [
                {
                    "meta": {},
                    "timestamp": [1704067200],
                    "indicators": {"quote": [{"close": [100], "volume": [1]}]},
                }
            ]
        }
    }

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return payload

    monkeypatch.setattr("requests.get", lambda *_args, **_kwargs: Response())
    manager = MarketStateManager(str(tmp_path / "market.db"))
    manager.add_alert("ALRT", "below", 150)

    assert manager.evaluate_alerts() == ["Alert 1: ALRT is 100.00 (below 150.00)"]
