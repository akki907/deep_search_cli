"""Deterministic coverage for the next feature specification."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from react_loop.alerts import AlertScheduler, MemoryNotifier
from react_loop.forecast_eval import ForecastObservation, brier_score
from react_loop.market_data import (
    CachedMarketDataService,
    FallbackMarketDataProvider,
    Fundamentals,
    MarketDataCache,
    MarketDataError,
    MarketEvent,
    PriceBar,
    Quote,
    SourceInfo,
)
from react_loop.market_state import MarketStateManager
from react_loop.portfolio import calculate_portfolio, render_watchlist
from react_loop.reporting import render_report
from react_loop.sec_data import earnings_calendar_data, earnings_history_data, sec_filings_data


class FakeProvider:
    name = "fixture"

    def __init__(self):
        self.calls = {"quote": 0, "history": 0, "fundamentals": 0, "events": 0}
        self.fail = False

    def quote(self, symbol):
        self.calls["quote"] += 1
        if self.fail:
            raise MarketDataError("fixture unavailable")
        return Quote(
            symbol=symbol,
            price=120,
            previous_close=110,
            currency="USD",
            exchange="TEST",
            as_of="2024-01-03",
            source=SourceInfo("fixture", "provider", "fixture://quote", "2024-01-03"),
        )

    def history(self, symbol, period):
        self.calls["history"] += 1
        if self.fail:
            raise MarketDataError("fixture unavailable")
        return [
            PriceBar("2024-01-01", 100, 1000),
            PriceBar("2024-01-02", 110, 1100),
            PriceBar("2024-01-03", 120, 1200),
        ]

    def fundamentals(self, symbol):
        self.calls["fundamentals"] += 1
        return Fundamentals(
            symbol,
            "Fixture Co",
            "Technology",
            "Software",
            {"trailing_pe": 20},
            SourceInfo("fixture", "provider", "fixture://fundamentals", "2024-01-03"),
        )

    def events(self, symbol):
        self.calls["events"] += 1
        return [
            MarketEvent(
                symbol,
                "earnings",
                "2024-02-01",
                "estimated",
                "fixture",
                "Fixture earnings",
                SourceInfo("fixture", "calendar", "fixture://events", "2024-01-03"),
            )
        ]


def test_persistent_market_cache_reuses_and_falls_back_stale(tmp_path):
    provider = FakeProvider()
    cache = MarketDataCache(tmp_path / "cache.db")
    service = CachedMarketDataService(provider, cache, ttl_seconds=60)

    first = service.history("FIX", "1y")
    second = service.history("FIX", "1y")

    assert not first.stale
    assert not second.stale
    assert provider.calls["history"] == 1

    provider.fail = True
    stale_service = CachedMarketDataService(provider, cache, ttl_seconds=0)
    stale = stale_service.history("FIX", "1y")
    assert stale.stale
    assert stale.value[-1].close == 120
    assert cache.clear("FIX") == 1


def test_sec_and_earnings_adapters_normalize_sources(monkeypatch):
    class Response:
        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self.payload

    def fake_get(url, **_kwargs):
        if "company_tickers" in url:
            return Response({"0": {"ticker": "FIX", "cik_str": 123, "title": "Fixture Co"}})
        if "submissions" in url:
            return Response(
                {
                    "filings": {
                        "recent": {
                            "form": ["10-K"],
                            "accessionNumber": ["0000123-24-000001"],
                            "primaryDocument": ["fixture.htm"],
                            "filingDate": ["2024-02-01"],
                            "reportDate": ["2023-12-31"],
                        }
                    }
                }
            )
        return Response(
            {
                "quoteSummary": {
                    "result": [
                        {
                            "earningsHistory": {
                                "history": [
                                    {
                                        "quarter": {"raw": 1704067200},
                                        "epsActual": {"raw": 2.0},
                                        "epsEstimate": {"raw": 1.8},
                                        "epsDifference": {"raw": 0.2},
                                        "surprisePercent": {"raw": 0.1},
                                    }
                                ]
                            },
                            "calendarEvents": {
                                "earnings": {"earningsDate": [{"raw": 1706745600}]}
                            },
                        }
                    ]
                }
            }
        )

    monkeypatch.setattr("requests.get", fake_get)
    filings = sec_filings_data("FIX", "10-K", 1)
    history = earnings_history_data("FIX")
    calendar = earnings_calendar_data("FIX")

    assert filings[0]["accession_number"] == "0000123-24-000001"
    assert filings[0]["source_type"] == "sec_filing"
    assert history[0]["actual_eps"] == 2.0
    assert calendar[0]["status"] == "estimated"


def test_portfolio_analytics_marks_missing_prices_unavailable():
    report = calculate_portfolio(
        [
            {"symbol": "FIX", "quantity": 2, "cost_basis": 100},
            {"symbol": "MISS", "quantity": 1, "cost_basis": 50},
        ],
        {"FIX": {"price": 120, "as_of": "2024-01-03"}, "MISS": {"price": None}},
    )

    assert report["totals"]["available_market_value"] == 240
    assert report["totals"]["market_value"] is None
    assert report["positions"][1]["market_value"] is None


def test_alert_scheduler_honors_cooldown_and_notifies(tmp_path):
    manager = MarketStateManager(str(tmp_path / "state.db"))
    alert_id = manager.add_alert("FIX", "above", 100, cooldown_seconds=3600)
    notifier = MemoryNotifier()
    scheduler = AlertScheduler(manager, [notifier])
    now = datetime(2024, 1, 1, tzinfo=UTC)
    def fetcher(_symbol):
        return [("2024-01-01", 120, 1000)]

    first = scheduler.run_once(now=now, fetcher=fetcher)
    second = scheduler.run_once(now=now + timedelta(minutes=30), fetcher=fetcher)
    third = scheduler.run_once(now=now + timedelta(hours=1), fetcher=fetcher)

    assert len(first) == 1
    assert second == []
    assert len(third) == 1
    assert len(notifier.messages) == 2
    assert manager.alert_history(alert_id)[0]["status"] == "triggered"


def test_alert_notification_failure_is_recorded_without_disabling_alert(tmp_path):
    class FailingNotifier:
        name = "failing"

        def notify(self, _message, _alert):
            raise RuntimeError("delivery unavailable")

    manager = MarketStateManager(str(tmp_path / "state.db"))
    alert_id = manager.add_alert("FIX", "above", 100)
    scheduler = AlertScheduler(manager, [FailingNotifier()])

    assert scheduler.run_once(
        now=datetime(2024, 1, 1, tzinfo=UTC),
        fetcher=lambda _symbol: [("2024-01-01", 120, 1000)],
    )

    assert manager.alerts()[0]["enabled"] is True
    history = manager.alert_history(alert_id)
    assert history[-1]["status"] == "notification_error"
    assert "delivery unavailable" in history[-1]["message"]


def test_probability_evaluation_and_structured_report():
    observations = [ForecastObservation(0.8, 1), ForecastObservation(0.2, 0)]
    assert brier_score(observations) == pytest.approx(0.04)

    answer = "# Executive summary\nA finding.\n\n# Sources\n1. Filing https://example.test/filing"
    payload = json.loads(render_report(answer, "Research FIX", "json"))
    assert payload["symbols"] == ["FIX"]
    assert payload["source_entries"][0]["url"] == "https://example.test/filing"


def test_watchlist_json_render_is_stable():
    output = json.loads(render_watchlist([{"symbol": "FIX", "price": "120.00"}], "json"))
    assert output[0]["symbol"] == "FIX"


def test_fallback_provider_uses_second_provider_after_failure():
    primary = FakeProvider()
    primary.fail = True
    secondary = FakeProvider()

    result = FallbackMarketDataProvider([primary, secondary]).quote("FIX")

    assert result.price == 120
    assert primary.calls["quote"] == 1
    assert secondary.calls["quote"] == 1


def test_transactions_update_average_cost_and_reject_oversell(tmp_path):
    manager = MarketStateManager(str(tmp_path / "state.db"))

    manager.add_transaction("FIX", "buy", 2, 100)
    manager.add_transaction("FIX", "buy", 2, 120)

    assert manager.portfolio() == [{"symbol": "FIX", "quantity": 4.0, "cost_basis": 110.0}]
    manager.add_transaction("FIX", "sell", 1, 130)
    assert manager.portfolio()[0]["quantity"] == 3.0
    with pytest.raises(ValueError, match="cannot sell"):
        manager.add_transaction("FIX", "sell", 4, 130)


def test_report_html_preserves_source_links():
    html = render_report(
        "# Sources\n1. Filing https://example.test/filing",
        "Research FIX",
        "html",
    )

    assert '<a href="https://example.test/filing">https://example.test/filing</a>' in html
