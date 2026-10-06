"""Normalized market-data providers with persistent SQLite caching."""

from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Generic, Protocol, TypeVar

import requests

from react_loop.tools import SEARCH_TIMEOUT, STOCK_API_URL, STOCK_SUMMARY_URL, USER_AGENT

T = TypeVar("T")


@dataclass(frozen=True)
class SourceInfo:
    """Provenance metadata attached to normalized market data."""

    provider: str
    source_type: str
    url: str
    retrieved_at: str
    published_at: str | None = None


@dataclass(frozen=True)
class PriceBar:
    """One normalized daily closing-price observation."""

    date: str
    close: float
    volume: int


@dataclass(frozen=True)
class Quote:
    """A normalized latest quote."""

    symbol: str
    price: float | None
    previous_close: float | None
    currency: str | None
    exchange: str | None
    as_of: str
    source: SourceInfo


@dataclass(frozen=True)
class Fundamentals:
    """A normalized fundamentals snapshot with unavailable values preserved."""

    symbol: str
    company: str | None
    sector: str | None
    industry: str | None
    metrics: dict[str, float | str | None]
    source: SourceInfo


@dataclass(frozen=True)
class MarketEvent:
    """A normalized future or historical market event."""

    symbol: str | None
    event_type: str
    expected_date: str | None
    status: str
    confidence: str
    title: str
    source: SourceInfo


@dataclass(frozen=True)
class CachedResult(Generic[T]):
    """Data plus cache freshness metadata."""

    value: T
    stale: bool
    fetched_at: str
    data_at: str | None
    source: SourceInfo


class MarketDataProvider(Protocol):
    """Provider contract consumed by the market-data service."""

    name: str

    def quote(self, symbol: str) -> Quote: ...

    def history(self, symbol: str, period: str) -> list[PriceBar]: ...

    def fundamentals(self, symbol: str) -> Fundamentals: ...

    def events(self, symbol: str) -> list[MarketEvent]: ...


class MarketDataError(RuntimeError):
    """Raised when a provider cannot return usable market data."""


class MarketDataCache:
    """Persistent SQLite cache for normalized provider payloads."""

    def __init__(self, db_path: str | Path = "market_data_cache.db") -> None:
        self.db_path = str(db_path)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS market_cache (
                    cache_key TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    data_type TEXT NOT NULL,
                    interval TEXT NOT NULL,
                    period TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    fetched_at TEXT NOT NULL,
                    data_at TEXT,
                    payload TEXT NOT NULL
                )
                """
            )

    def get(self, cache_key: str) -> dict[str, Any] | None:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT symbol, data_type, interval, period, provider, fetched_at, data_at, payload "
                "FROM market_cache WHERE cache_key = ?",
                (cache_key,),
            ).fetchone()
        if row is None:
            return None
        symbol, data_type, interval, period, provider, fetched_at, data_at, payload = row
        return {
            "symbol": symbol,
            "data_type": data_type,
            "interval": interval,
            "period": period,
            "provider": provider,
            "fetched_at": fetched_at,
            "data_at": data_at,
            "payload": json.loads(payload),
        }

    def put(
        self,
        cache_key: str,
        *,
        symbol: str,
        data_type: str,
        interval: str,
        period: str,
        provider: str,
        fetched_at: str,
        data_at: str | None,
        payload: Any,
    ) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO market_cache(
                    cache_key, symbol, data_type, interval, period, provider,
                    fetched_at, data_at, payload
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    provider=excluded.provider,
                    fetched_at=excluded.fetched_at,
                    data_at=excluded.data_at,
                    payload=excluded.payload
                """,
                (
                    cache_key,
                    symbol,
                    data_type,
                    interval,
                    period,
                    provider,
                    fetched_at,
                    data_at,
                    json.dumps(payload, ensure_ascii=False),
                ),
            )

    def clear(self, symbol: str | None = None) -> int:
        with sqlite3.connect(self.db_path) as conn:
            if symbol:
                cursor = conn.execute("DELETE FROM market_cache WHERE symbol = ?", (symbol.upper(),))
            else:
                cursor = conn.execute("DELETE FROM market_cache")
            return cursor.rowcount

    def entries(self, symbol: str | None = None) -> list[dict[str, Any]]:
        query = (
            "SELECT cache_key, symbol, data_type, interval, period, provider, fetched_at, data_at "
            "FROM market_cache"
        )
        args: tuple[str, ...] = ()
        if symbol:
            query += " WHERE symbol = ?"
            args = (symbol.upper(),)
        query += " ORDER BY symbol, data_type, period"
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(query, args).fetchall()
        now = datetime.now(UTC)
        entries: list[dict[str, Any]] = []
        for row in rows:
            try:
                age_seconds = max(0.0, (now - datetime.fromisoformat(row[6])).total_seconds())
            except ValueError:
                age_seconds = None
            entries.append(
                {
                    "cache_key": row[0],
                    "symbol": row[1],
                    "data_type": row[2],
                    "interval": row[3],
                    "period": row[4],
                    "provider": row[5],
                    "fetched_at": row[6],
                    "data_at": row[7],
                    "age_seconds": age_seconds,
                    "fresh": age_seconds is not None and age_seconds < 60,
                }
            )
        return entries


class YahooFinanceProvider:
    """Yahoo Finance implementation of the normalized provider contract."""

    name = "yahoo_finance"

    def _source(self, url: str, source_type: str = "provider") -> SourceInfo:
        return SourceInfo(
            provider=self.name,
            source_type=source_type,
            url=url,
            retrieved_at=datetime.now(UTC).isoformat(timespec="seconds"),
        )

    def _get(self, url: str, params: dict[str, Any]) -> dict[str, Any]:
        try:
            response = requests.get(
                url,
                params=params,
                headers={"User-Agent": USER_AGENT},
                timeout=SEARCH_TIMEOUT,
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            raise MarketDataError(f"Yahoo Finance request failed: {exc}") from exc
        return payload

    def _chart(self, symbol: str, period: str) -> tuple[dict[str, Any], list[PriceBar]]:
        url = f"{STOCK_API_URL}/{symbol}"
        payload = self._get(
            url,
            {"range": period, "interval": "1d", "events": "history"},
        )
        result = (payload.get("chart", {}).get("result") or [None])[0]
        if not result:
            raise MarketDataError(f"Yahoo Finance returned no chart data for {symbol}.")
        quote = ((result.get("indicators") or {}).get("quote") or [{}])[0]
        timestamps = result.get("timestamp") or []
        closes = quote.get("close") or []
        volumes = quote.get("volume") or []
        bars: list[PriceBar] = []
        for index, (timestamp, close) in enumerate(zip(timestamps, closes, strict=False)):
            if close is None:
                continue
            try:
                date = datetime.fromtimestamp(int(timestamp), UTC).date().isoformat()
                price = float(close)
                volume = int(volumes[index] or 0) if index < len(volumes) else 0
            except (TypeError, ValueError, OverflowError):
                continue
            bars.append(PriceBar(date=date, close=price, volume=volume))
        if not bars:
            raise MarketDataError(f"Yahoo Finance returned no usable prices for {symbol}.")
        return result.get("meta") or {}, bars

    def quote(self, symbol: str) -> Quote:
        meta, bars = self._chart(symbol, "5d")
        as_of = meta.get("regularMarketTime")
        if as_of:
            try:
                as_of_text = datetime.fromtimestamp(int(as_of), UTC).isoformat()
            except (TypeError, ValueError, OverflowError):
                as_of_text = bars[-1].date
        else:
            as_of_text = bars[-1].date
        return Quote(
            symbol=symbol,
            price=float(meta["regularMarketPrice"]) if meta.get("regularMarketPrice") is not None else bars[-1].close,
            previous_close=float(meta["previousClose"]) if meta.get("previousClose") is not None else None,
            currency=meta.get("currency"),
            exchange=meta.get("exchangeName"),
            as_of=as_of_text,
            source=self._source(f"{STOCK_API_URL}/{symbol}"),
        )

    def history(self, symbol: str, period: str) -> list[PriceBar]:
        _meta, bars = self._chart(symbol, period)
        return bars

    def fundamentals(self, symbol: str) -> Fundamentals:
        url = f"{STOCK_SUMMARY_URL}/{symbol}"
        payload = self._get(
            url,
            {
                "modules": ",".join(
                    [
                        "price",
                        "summaryDetail",
                        "defaultKeyStatistics",
                        "financialData",
                        "assetProfile",
                    ]
                )
            },
        )
        result = ((payload.get("quoteSummary") or {}).get("result") or [None])[0]
        if not result:
            raise MarketDataError(f"Yahoo Finance returned no fundamentals for {symbol}.")

        def raw(section: dict[str, Any], key: str) -> Any:
            value = section.get(key)
            return value.get("raw", value.get("fmt")) if isinstance(value, dict) else value

        price = result.get("price") or {}
        detail = result.get("summaryDetail") or {}
        stats = result.get("defaultKeyStatistics") or {}
        financial = result.get("financialData") or {}
        profile = result.get("assetProfile") or {}
        return Fundamentals(
            symbol=symbol,
            company=raw(price, "longName") or raw(price, "shortName"),
            sector=profile.get("sector"),
            industry=profile.get("industry"),
            metrics={
                "market_cap": raw(price, "marketCap"),
                "trailing_pe": raw(detail, "trailingPE"),
                "forward_pe": raw(detail, "forwardPE"),
                "peg_ratio": raw(stats, "pegRatio"),
                "revenue_growth": raw(financial, "revenueGrowth"),
                "profit_margin": raw(financial, "profitMargins"),
                "operating_margin": raw(financial, "operatingMargins"),
                "return_on_equity": raw(financial, "returnOnEquity"),
                "debt_to_equity": raw(financial, "debtToEquity"),
                "free_cash_flow": raw(financial, "freeCashflow"),
                "dividend_yield": raw(detail, "dividendYield"),
            },
            source=self._source(url),
        )

    def events(self, symbol: str) -> list[MarketEvent]:
        url = f"{STOCK_SUMMARY_URL}/{symbol}"
        payload = self._get(url, {"modules": "calendarEvents"})
        result = ((payload.get("quoteSummary") or {}).get("result") or [None])[0]
        if not result:
            return []
        calendar = result.get("calendarEvents") or {}
        source = self._source(url, "calendar")
        events: list[MarketEvent] = []
        earnings = calendar.get("earnings") or {}
        dates = earnings.get("earningsDate") or []
        for value in dates:
            raw = value.get("raw") if isinstance(value, dict) else value
            expected = None
            if raw:
                try:
                    expected = datetime.fromtimestamp(int(raw), UTC).date().isoformat()
                except (TypeError, ValueError, OverflowError):
                    expected = str(raw)
            events.append(
                MarketEvent(
                    symbol=symbol,
                    event_type="earnings",
                    expected_date=expected,
                    status="estimated",
                    confidence="provider",
                    title=f"{symbol} earnings date",
                    source=source,
                )
            )
        return events


class FallbackMarketDataProvider:
    """Try providers in order and reject materially conflicting quote values."""

    name = "fallback"

    def __init__(self, providers: list[MarketDataProvider]) -> None:
        if not providers:
            raise ValueError("at least one market-data provider is required")
        self.providers = providers

    def _try(self, method: str, *args: Any) -> Any:
        errors: list[str] = []
        successful: list[Any] = []
        for provider in self.providers:
            try:
                value = getattr(provider, method)(*args)
                successful.append(value)
                if method != "quote":
                    return value
            except Exception as exc:
                errors.append(f"{provider.name}: {exc}")
        if successful:
            if method == "quote" and len(successful) > 1:
                prices = [item.price for item in successful if item.price is not None]
                if prices and max(prices) - min(prices) > max(prices) * 0.05:
                    raise MarketDataError("market-data providers returned conflicting quote values")
            return successful[0]
        raise MarketDataError("; ".join(errors) or "all market-data providers failed")

    def quote(self, symbol: str) -> Quote:
        return self._try("quote", symbol)

    def history(self, symbol: str, period: str) -> list[PriceBar]:
        return self._try("history", symbol, period)

    def fundamentals(self, symbol: str) -> Fundamentals:
        return self._try("fundamentals", symbol)

    def events(self, symbol: str) -> list[MarketEvent]:
        return self._try("events", symbol)


class CachedMarketDataService:
    """Provider facade with persistent cache and stale-data fallback."""

    def __init__(
        self,
        provider: MarketDataProvider | None = None,
        cache: MarketDataCache | None = None,
        *,
        ttl_seconds: float = 60.0,
        providers: list[MarketDataProvider] | None = None,
    ) -> None:
        if provider is not None and providers is not None:
            raise ValueError("provide provider or providers, not both")
        self.provider = (
            provider
            if provider is not None
            else FallbackMarketDataProvider(providers)
            if providers
            else YahooFinanceProvider()
        )
        default_path = os.environ.get("REACT_LOOP_MARKET_CACHE", "market_data_cache.db")
        self.cache = cache or MarketDataCache(default_path)
        self.ttl_seconds = ttl_seconds

    @staticmethod
    def _now() -> datetime:
        return datetime.now(UTC)

    def _fresh(self, fetched_at: str) -> bool:
        try:
            fetched = datetime.fromisoformat(fetched_at)
        except ValueError:
            return False
        return (self._now() - fetched).total_seconds() < self.ttl_seconds

    def _result_source(self, entry: dict[str, Any]) -> SourceInfo:
        return SourceInfo(
            provider=entry["provider"],
            source_type="cached",
            url="",
            retrieved_at=entry["fetched_at"],
            published_at=entry["data_at"],
        )

    def _load_cached(self, key: str) -> dict[str, Any] | None:
        return self.cache.get(key)

    def history(self, symbol: str, period: str, *, refresh: bool = False) -> CachedResult[list[PriceBar]]:
        ticker = symbol.upper()
        key = f"{self.provider.name}:history:{ticker}:1d:{period}"
        entry = None if refresh else self._load_cached(key)
        if entry and self._fresh(entry["fetched_at"]):
            bars = [PriceBar(**item) for item in entry["payload"]]
            return CachedResult(bars, False, entry["fetched_at"], entry["data_at"], self._result_source(entry))
        try:
            bars = self.provider.history(ticker, period)
            fetched = self._now().isoformat(timespec="seconds")
            data_at = bars[-1].date if bars else None
            source = SourceInfo(
                provider=self.provider.name,
                source_type="provider",
                url=f"{STOCK_API_URL}/{ticker}",
                retrieved_at=fetched,
                published_at=data_at,
            )
            self.cache.put(
                key,
                symbol=ticker,
                data_type="history",
                interval="1d",
                period=period,
                provider=self.provider.name,
                fetched_at=fetched,
                data_at=data_at,
                payload=[asdict(bar) for bar in bars],
            )
            return CachedResult(bars, False, fetched, data_at, source)
        except Exception as exc:
            if entry:
                bars = [PriceBar(**item) for item in entry["payload"]]
                source = self._result_source(entry)
                return CachedResult(bars, True, entry["fetched_at"], entry["data_at"], source)
            raise MarketDataError(str(exc)) from exc

    def quote(self, symbol: str, *, refresh: bool = False) -> CachedResult[Quote]:
        ticker = symbol.upper()
        key = f"{self.provider.name}:quote:1d:{ticker}"
        entry = None if refresh else self._load_cached(key)
        if entry and self._fresh(entry["fetched_at"]):
            quote = Quote(**entry["payload"], source=self._result_source(entry))
            return CachedResult(quote, False, entry["fetched_at"], entry["data_at"], quote.source)
        try:
            quote = self.provider.quote(ticker)
            fetched = self._now().isoformat(timespec="seconds")
            self.cache.put(
                key,
                symbol=ticker,
                data_type="quote",
                interval="1d",
                period="5d",
                provider=self.provider.name,
                fetched_at=fetched,
                data_at=quote.as_of,
                payload={
                    "symbol": quote.symbol,
                    "price": quote.price,
                    "previous_close": quote.previous_close,
                    "currency": quote.currency,
                    "exchange": quote.exchange,
                    "as_of": quote.as_of,
                },
            )
            return CachedResult(quote, False, fetched, quote.as_of, quote.source)
        except Exception as exc:
            if entry:
                cached = entry["payload"]
                source = self._result_source(entry)
                quote = Quote(**cached, source=source)
                return CachedResult(quote, True, entry["fetched_at"], entry["data_at"], source)
            raise MarketDataError(str(exc)) from exc

    def fundamentals(self, symbol: str, *, refresh: bool = False) -> CachedResult[Fundamentals]:
        ticker = symbol.upper()
        key = f"{self.provider.name}:fundamentals:1d:{ticker}"
        entry = None if refresh else self._load_cached(key)
        if entry and self._fresh(entry["fetched_at"]):
            payload = entry["payload"]
            fundamentals = Fundamentals(**payload, source=self._result_source(entry))
            return CachedResult(fundamentals, False, entry["fetched_at"], entry["data_at"], fundamentals.source)
        try:
            fundamentals = self.provider.fundamentals(ticker)
            fetched = self._now().isoformat(timespec="seconds")
            self.cache.put(
                key,
                symbol=ticker,
                data_type="fundamentals",
                interval="snapshot",
                period="current",
                provider=self.provider.name,
                fetched_at=fetched,
                data_at=fetched,
                payload={
                    "symbol": fundamentals.symbol,
                    "company": fundamentals.company,
                    "sector": fundamentals.sector,
                    "industry": fundamentals.industry,
                    "metrics": fundamentals.metrics,
                },
            )
            return CachedResult(fundamentals, False, fetched, fetched, fundamentals.source)
        except Exception as exc:
            if entry:
                payload = entry["payload"]
                source = self._result_source(entry)
                fundamentals = Fundamentals(**payload, source=source)
                return CachedResult(fundamentals, True, entry["fetched_at"], entry["data_at"], source)
            raise MarketDataError(str(exc)) from exc

    def events(self, symbol: str, *, refresh: bool = False) -> CachedResult[list[MarketEvent]]:
        ticker = symbol.upper()
        key = f"{self.provider.name}:events:1d:{ticker}"
        entry = None if refresh else self._load_cached(key)
        if entry and self._fresh(entry["fetched_at"]):
            events = [
                MarketEvent(**{**item, "source": self._result_source(entry)})
                for item in entry["payload"]
            ]
            return CachedResult(events, False, entry["fetched_at"], entry["data_at"], self._result_source(entry))
        try:
            events = self.provider.events(ticker)
            fetched = self._now().isoformat(timespec="seconds")
            self.cache.put(
                key,
                symbol=ticker,
                data_type="events",
                interval="event",
                period="current",
                provider=self.provider.name,
                fetched_at=fetched,
                data_at=events[0].expected_date if events else None,
                payload=[
                    {
                        "symbol": event.symbol,
                        "event_type": event.event_type,
                        "expected_date": event.expected_date,
                        "status": event.status,
                        "confidence": event.confidence,
                        "title": event.title,
                    }
                    for event in events
                ],
            )
            source = events[0].source if events else SourceInfo(self.provider.name, "calendar", f"{STOCK_SUMMARY_URL}/{ticker}", fetched)
            return CachedResult(events, False, fetched, events[0].expected_date if events else None, source)
        except Exception as exc:
            if entry:
                source = self._result_source(entry)
                events = [MarketEvent(**{**item, "source": source}) for item in entry["payload"]]
                return CachedResult(events, True, entry["fetched_at"], entry["data_at"], source)
            raise MarketDataError(str(exc)) from exc


def default_market_data_service() -> CachedMarketDataService:
    """Build the configured default service."""
    return CachedMarketDataService()
