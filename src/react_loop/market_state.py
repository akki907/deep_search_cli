"""Persistent watchlists, portfolio holdings, and manually evaluated alerts."""

from __future__ import annotations

import re
import sqlite3
from datetime import UTC, datetime
from typing import Any

_SYMBOL = re.compile(r"[A-Z0-9][A-Z0-9.-]{0,11}")


class MarketStateManager:
    """Store user-owned market state in a small SQLite database."""

    def __init__(self, db_path: str = "market_state.db") -> None:
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS watchlist (
                    symbol TEXT PRIMARY KEY,
                    added_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS portfolio (
                    symbol TEXT PRIMARY KEY,
                    quantity REAL NOT NULL,
                    cost_basis REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    operator TEXT NOT NULL CHECK(operator IN ('above', 'below')),
                    threshold REAL NOT NULL,
                    enabled INTEGER NOT NULL DEFAULT 1
                );
                """
            )

    @staticmethod
    def normalize_symbol(symbol: str) -> str:
        ticker = symbol.strip().upper()
        if not _SYMBOL.fullmatch(ticker):
            raise ValueError(f"invalid stock symbol {symbol!r}")
        return ticker

    def add_watch(self, symbol: str) -> str:
        ticker = self.normalize_symbol(symbol)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR IGNORE INTO watchlist(symbol, added_at) VALUES (?, ?)",
                (ticker, datetime.now(UTC).isoformat(timespec="seconds")),
            )
        return ticker

    def remove_watch(self, symbol: str) -> str:
        ticker = self.normalize_symbol(symbol)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM watchlist WHERE symbol = ?", (ticker,))
        return ticker

    def watchlist(self) -> list[str]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("SELECT symbol FROM watchlist ORDER BY symbol").fetchall()
        return [row[0] for row in rows]

    def set_position(self, symbol: str, quantity: float, cost_basis: float) -> str:
        ticker = self.normalize_symbol(symbol)
        if quantity < 0 or cost_basis < 0:
            raise ValueError("quantity and cost_basis must be non-negative")
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO portfolio(symbol, quantity, cost_basis) VALUES (?, ?, ?) "
                "ON CONFLICT(symbol) DO UPDATE SET quantity=excluded.quantity, cost_basis=excluded.cost_basis",
                (ticker, quantity, cost_basis),
            )
        return ticker

    def portfolio(self) -> list[dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT symbol, quantity, cost_basis FROM portfolio ORDER BY symbol"
            ).fetchall()
        return [
            {"symbol": symbol, "quantity": quantity, "cost_basis": cost_basis}
            for symbol, quantity, cost_basis in rows
        ]

    def add_alert(self, symbol: str, operator: str, threshold: float) -> int:
        ticker = self.normalize_symbol(symbol)
        if operator not in {"above", "below"}:
            raise ValueError("operator must be 'above' or 'below'")
        if threshold < 0:
            raise ValueError("threshold must be non-negative")
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "INSERT INTO alerts(symbol, operator, threshold) VALUES (?, ?, ?)",
                (ticker, operator, threshold),
            )
            return int(cursor.lastrowid)

    def alerts(self) -> list[dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT id, symbol, operator, threshold, enabled FROM alerts ORDER BY id"
            ).fetchall()
        return [
            {
                "id": alert_id,
                "symbol": symbol,
                "operator": operator,
                "threshold": threshold,
                "enabled": bool(enabled),
            }
            for alert_id, symbol, operator, threshold, enabled in rows
        ]

    def evaluate_alerts(self) -> list[str]:
        """Fetch current prices and return only alerts whose condition is met."""
        from react_loop.tools import _fetch_stock_data

        triggered: list[str] = []
        for alert in self.alerts():
            if not alert["enabled"]:
                continue
            _, observations, error = _fetch_stock_data(alert["symbol"], "1mo")
            if error or not observations:
                continue
            current = observations[-1][1]
            matched = (
                current > alert["threshold"]
                if alert["operator"] == "above"
                else current < alert["threshold"]
            )
            if matched:
                triggered.append(
                    f"Alert {alert['id']}: {alert['symbol']} is {current:.2f} "
                    f"({alert['operator']} {alert['threshold']:.2f})"
                )
        return triggered
