"""SEC filing, earnings, and catalyst data adapters."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from typing import Any

import requests

SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
YAHOO_SUMMARY_URL = "https://query1.finance.yahoo.com/v10/finance/quoteSummary"
SEC_TIMEOUT = 15


def _headers() -> dict[str, str]:
    return {
        "User-Agent": os.environ.get(
            "REACT_LOOP_SEC_USER_AGENT",
            "react-loop/0.1 (research client; contact configured by operator)",
        ),
        "Accept-Encoding": "gzip, deflate",
    }


def _get(url: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    try:
        response = requests.get(url, params=params, headers=_headers(), timeout=SEC_TIMEOUT)
        response.raise_for_status()
        return response.json()
    except Exception as exc:
        raise RuntimeError(f"data request failed: {exc}") from exc


def resolve_cik(symbol: str) -> tuple[str, str]:
    """Resolve a ticker to a zero-padded SEC CIK and company title."""
    ticker = symbol.strip().upper()
    payload = _get(SEC_TICKERS_URL)
    for value in payload.values():
        if str(value.get("ticker", "")).upper() == ticker:
            cik = str(value.get("cik_str", "")).zfill(10)
            return cik, str(value.get("title") or ticker)
    raise ValueError(f"SEC does not list ticker {ticker!r}.")


def sec_filings_data(symbol: str, filing_type: str = "10-K", limit: int = 5) -> list[dict[str, Any]]:
    """Return recent SEC filings with stable accession URLs."""
    cik, company = resolve_cik(symbol)
    bounded_limit = max(1, min(int(limit), 20))
    form = filing_type.strip().upper()
    submissions_url = SEC_SUBMISSIONS_URL.format(cik=cik)
    payload = _get(submissions_url)
    recent = payload.get("filings", {}).get("recent", {})
    filings: list[dict[str, Any]] = []
    for index, value in enumerate(recent.get("form", [])):
        if form and value.upper() != form:
            continue
        accession = recent.get("accessionNumber", [])[index]
        accession_path = accession.replace("-", "")
        primary_document = recent.get("primaryDocument", [""])[index]
        filing_date = recent.get("filingDate", [None])[index]
        report_date = recent.get("reportDate", [None])[index]
        url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession_path}/{primary_document}"
        filings.append(
            {
                "symbol": symbol.upper(),
                "company": company,
                "cik": cik,
                "form": value,
                "filing_date": filing_date,
                "report_date": report_date,
                "accession_number": accession,
                "primary_document": primary_document,
                "url": url,
                "retrieved_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "source_type": "sec_filing",
            }
        )
        if len(filings) >= bounded_limit:
            break
    return filings


def _yahoo_value(value: Any) -> Any:
    if isinstance(value, dict):
        return value.get("raw", value.get("fmt"))
    return value


def earnings_history_data(symbol: str) -> list[dict[str, Any]]:
    """Return reported and estimated Yahoo earnings-history rows."""
    ticker = symbol.strip().upper()
    url = f"{YAHOO_SUMMARY_URL}/{ticker}"
    payload = _get(
        url,
        {"modules": "earningsHistory,earningsTrend,calendarEvents"},
    )
    result = ((payload.get("quoteSummary") or {}).get("result") or [None])[0]
    if not result:
        return []
    history = result.get("earningsHistory") or {}
    rows: list[dict[str, Any]] = []
    for item in history.get("history", []) or []:
        rows.append(
            {
                "symbol": ticker,
                "quarter": _yahoo_value(item.get("quarter")),
                "actual_eps": _yahoo_value(item.get("epsActual")),
                "estimate_eps": _yahoo_value(item.get("epsEstimate")),
                "surprise": _yahoo_value(item.get("epsDifference")),
                "surprise_percent": _yahoo_value(item.get("surprisePercent")),
                "source_url": url,
                "retrieved_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "source_type": "provider_earnings_history",
            }
        )
    return rows


def earnings_calendar_data(symbol: str) -> list[dict[str, Any]]:
    """Return upcoming earnings dates from the provider calendar."""
    ticker = symbol.strip().upper()
    url = f"{YAHOO_SUMMARY_URL}/{ticker}"
    payload = _get(url, {"modules": "calendarEvents"})
    result = ((payload.get("quoteSummary") or {}).get("result") or [None])[0]
    if not result:
        return []
    calendar = result.get("calendarEvents") or {}
    dates: list[dict[str, Any]] = []
    for item in calendar.get("earnings", {}).get("earningsDate", []) or []:
        raw = _yahoo_value(item)
        expected_date = None
        if raw:
            try:
                expected_date = datetime.fromtimestamp(int(raw), UTC).date().isoformat()
            except (TypeError, ValueError, OverflowError):
                expected_date = str(raw)
        dates.append(
            {
                "symbol": ticker,
                "event_type": "earnings",
                "title": f"{ticker} earnings",
                "expected_date": expected_date,
                "status": "estimated",
                "confidence": "provider",
                "source_url": url,
                "retrieved_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "source_type": "provider_calendar",
            }
        )
    return dates
