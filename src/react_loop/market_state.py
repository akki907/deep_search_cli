"""Persistent watchlists, portfolios, alerts, and alert history."""

from __future__ import annotations

import re
import sqlite3
from datetime import UTC, datetime, timedelta
from typing import Any, Callable

_SYMBOL = re.compile(r"[A-Z0-9][A-Z0-9.-]{0,11}")
_ALERT_OPERATORS = frozenset({"above", "below"})
_ALERT_CONDITIONS = frozenset({"price", "change", "volume", "rsi", "moving_average", "drawdown"})


class MarketStateManager:
    """Store user-owned market state in SQLite."""

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
                CREATE TABLE IF NOT EXISTS portfolio_transactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    action TEXT NOT NULL CHECK(action IN ('buy', 'sell')),
                    quantity REAL NOT NULL,
                    price REAL NOT NULL,
                    occurred_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    condition_type TEXT NOT NULL DEFAULT 'price',
                    operator TEXT NOT NULL CHECK(operator IN ('above', 'below')),
                    threshold REAL NOT NULL,
                    enabled INTEGER NOT NULL DEFAULT 1,
                    cooldown_seconds INTEGER NOT NULL DEFAULT 3600,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    last_triggered_at TEXT,
                    last_error TEXT
                );
                CREATE TABLE IF NOT EXISTS alert_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    alert_id INTEGER NOT NULL,
                    triggered_at TEXT NOT NULL,
                    observed_value REAL,
                    status TEXT NOT NULL,
                    message TEXT NOT NULL,
                    FOREIGN KEY(alert_id) REFERENCES alerts(id)
                );
                """
            )
            self._ensure_alert_columns(conn)

    @staticmethod
    def _ensure_alert_columns(conn: sqlite3.Connection) -> None:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(alerts)")}
        additions = {
            "condition_type": "TEXT NOT NULL DEFAULT 'price'",
            "cooldown_seconds": "INTEGER NOT NULL DEFAULT 3600",
            "created_at": "TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP",
            "last_triggered_at": "TEXT",
            "last_error": "TEXT",
        }
        for name, definition in additions.items():
            if name not in columns:
                conn.execute(f"ALTER TABLE alerts ADD COLUMN {name} {definition}")

    @staticmethod
    def normalize_symbol(symbol: str) -> str:
        """Normalize and validate a ticker symbol."""
        ticker = symbol.strip().upper()
        if not _SYMBOL.fullmatch(ticker):
            raise ValueError(f"invalid stock symbol {symbol!r}")
        return ticker

    def add_watch(self, symbol: str) -> str:
        """Add a ticker to the persistent watchlist."""
        ticker = self.normalize_symbol(symbol)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR IGNORE INTO watchlist(symbol, added_at) VALUES (?, ?)",
                (ticker, datetime.now(UTC).isoformat(timespec="seconds")),
            )
        return ticker

    def remove_watch(self, symbol: str) -> str:
        """Remove a ticker from the persistent watchlist."""
        ticker = self.normalize_symbol(symbol)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM watchlist WHERE symbol = ?", (ticker,))
        return ticker

    def watchlist(self) -> list[str]:
        """Return watchlist symbols in stable order."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("SELECT symbol FROM watchlist ORDER BY symbol").fetchall()
        return [row[0] for row in rows]

    def set_position(self, symbol: str, quantity: float, cost_basis: float) -> str:
        """Replace a position with quantity and average cost basis."""
        ticker = self.normalize_symbol(symbol)
        if quantity < 0 or cost_basis < 0:
            raise ValueError("quantity and cost_basis must be non-negative")
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO portfolio(symbol, quantity, cost_basis) VALUES (?, ?, ?)
                ON CONFLICT(symbol) DO UPDATE SET
                    quantity=excluded.quantity,
                    cost_basis=excluded.cost_basis
                """,
                (ticker, quantity, cost_basis),
            )
        return ticker

    def add_transaction(
        self,
        symbol: str,
        action: str,
        quantity: float,
        price: float,
        occurred_at: str | None = None,
    ) -> int:
        """Record a buy/sell and update the position using average-cost accounting."""
        ticker = self.normalize_symbol(symbol)
        action = action.lower()
        if action not in {"buy", "sell"}:
            raise ValueError("action must be 'buy' or 'sell'")
        if quantity <= 0 or price < 0:
            raise ValueError("quantity must be positive and price must be non-negative")
        with sqlite3.connect(self.db_path) as conn:
            current = conn.execute(
                "SELECT quantity, cost_basis FROM portfolio WHERE symbol = ?", (ticker,)
            ).fetchone()
            current_quantity = float(current[0]) if current else 0.0
            current_cost = float(current[1]) if current else 0.0
            if action == "sell" and quantity > current_quantity:
                raise ValueError("cannot sell more shares than the current position")
            if action == "buy":
                new_quantity = current_quantity + quantity
                new_cost = (
                    current_quantity * current_cost + quantity * price
                ) / new_quantity
            else:
                new_quantity = current_quantity - quantity
                new_cost = current_cost if new_quantity else 0.0
            cursor = conn.execute(
                "INSERT INTO portfolio_transactions(symbol, action, quantity, price, occurred_at) VALUES (?, ?, ?, ?, ?)",
                (ticker, action, quantity, price, occurred_at or datetime.now(UTC).isoformat(timespec="seconds")),
            )
            conn.execute(
                """
                INSERT INTO portfolio(symbol, quantity, cost_basis) VALUES (?, ?, ?)
                ON CONFLICT(symbol) DO UPDATE SET quantity=excluded.quantity, cost_basis=excluded.cost_basis
                """,
                (ticker, new_quantity, new_cost),
            )
            return int(cursor.lastrowid)

    def portfolio(self) -> list[dict[str, Any]]:
        """Return current positions in stable symbol order."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT symbol, quantity, cost_basis FROM portfolio ORDER BY symbol"
            ).fetchall()
        return [
            {"symbol": symbol, "quantity": quantity, "cost_basis": cost_basis}
            for symbol, quantity, cost_basis in rows
        ]

    def transactions(self, symbol: str | None = None) -> list[dict[str, Any]]:
        """Return transaction history, optionally filtered by symbol."""
        query = "SELECT id, symbol, action, quantity, price, occurred_at FROM portfolio_transactions"
        args: tuple[str, ...] = ()
        if symbol:
            query += " WHERE symbol = ?"
            args = (self.normalize_symbol(symbol),)
        query += " ORDER BY occurred_at, id"
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(query, args).fetchall()
        return [
            {
                "id": row[0],
                "symbol": row[1],
                "action": row[2],
                "quantity": row[3],
                "price": row[4],
                "occurred_at": row[5],
            }
            for row in rows
        ]

    def add_alert(
        self,
        symbol: str,
        operator: str,
        threshold: float,
        condition_type: str = "price",
        cooldown_seconds: int = 3600,
    ) -> int:
        """Create a persistent threshold alert."""
        ticker = self.normalize_symbol(symbol)
        operator = operator.lower()
        condition_type = condition_type.lower()
        if operator not in _ALERT_OPERATORS:
            raise ValueError("operator must be 'above' or 'below'")
        if condition_type not in _ALERT_CONDITIONS:
            raise ValueError(f"condition_type must be one of: {', '.join(sorted(_ALERT_CONDITIONS))}")
        if threshold < 0 and condition_type not in {"change", "drawdown"}:
            raise ValueError("threshold must be non-negative for this condition")
        if cooldown_seconds < 0:
            raise ValueError("cooldown_seconds must be non-negative")
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                INSERT INTO alerts(symbol, condition_type, operator, threshold, cooldown_seconds, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    ticker,
                    condition_type,
                    operator,
                    threshold,
                    cooldown_seconds,
                    datetime.now(UTC).isoformat(timespec="seconds"),
                ),
            )
            return int(cursor.lastrowid)

    def alerts(self, enabled_only: bool = False) -> list[dict[str, Any]]:
        """Return configured alerts and lifecycle metadata."""
        query = (
            "SELECT id, symbol, condition_type, operator, threshold, enabled, cooldown_seconds, "
            "created_at, last_triggered_at, last_error FROM alerts"
        )
        if enabled_only:
            query += " WHERE enabled = 1"
        query += " ORDER BY id"
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(query).fetchall()
        return [
            {
                "id": row[0],
                "symbol": row[1],
                "condition_type": row[2],
                "operator": row[3],
                "threshold": row[4],
                "enabled": bool(row[5]),
                "cooldown_seconds": row[6],
                "created_at": row[7],
                "last_triggered_at": row[8],
                "last_error": row[9],
            }
            for row in rows
        ]

    def set_alert_enabled(self, alert_id: int, enabled: bool) -> None:
        """Enable or disable an alert."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "UPDATE alerts SET enabled = ? WHERE id = ?", (int(enabled), alert_id)
            )
            if cursor.rowcount == 0:
                raise ValueError(f"alert {alert_id} was not found")

    def delete_alert(self, alert_id: int) -> None:
        """Delete an alert and its history."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM alert_history WHERE alert_id = ?", (alert_id,))
            cursor = conn.execute("DELETE FROM alerts WHERE id = ?", (alert_id,))
            if cursor.rowcount == 0:
                raise ValueError(f"alert {alert_id} was not found")

    def alert_history(self, alert_id: int | None = None) -> list[dict[str, Any]]:
        """Return alert trigger and error history."""
        query = "SELECT id, alert_id, triggered_at, observed_value, status, message FROM alert_history"
        args: tuple[int, ...] = ()
        if alert_id is not None:
            query += " WHERE alert_id = ?"
            args = (alert_id,)
        query += " ORDER BY triggered_at, id"
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(query, args).fetchall()
        return [
            {
                "id": row[0],
                "alert_id": row[1],
                "triggered_at": row[2],
                "observed_value": row[3],
                "status": row[4],
                "message": row[5],
            }
            for row in rows
        ]

    @staticmethod
    def _condition_value(alert: dict[str, Any], observations: list[tuple[str, float, int]]) -> float:
        prices = [price for _, price, _ in observations]
        condition = alert["condition_type"]
        if condition == "price":
            return prices[-1]
        if condition == "change":
            return (prices[-1] / prices[0] - 1) * 100 if prices[0] else 0.0
        if condition == "volume":
            return float(observations[-1][2])
        if condition == "moving_average":
            window = min(20, len(prices))
            return sum(prices[-window:]) / window
        if condition == "drawdown":
            peak = max(prices)
            return (prices[-1] / peak - 1) * 100 if peak else 0.0
        returns = [prices[index] / prices[index - 1] - 1 for index in range(1, len(prices)) if prices[index - 1]]
        gains = [value for value in returns[-14:] if value > 0]
        losses = [-value for value in returns[-14:] if value < 0]
        average_gain = sum(gains) / len(gains) if gains else 0.0
        average_loss = sum(losses) / len(losses) if losses else 0.0
        if average_loss == 0:
            return 100.0 if average_gain else 50.0
        return 100 - 100 / (1 + average_gain / average_loss)

    def evaluate_alerts(
        self,
        now: datetime | None = None,
        fetcher: Callable[[str], list[tuple[str, float, int]]] | None = None,
    ) -> list[str]:
        """Evaluate enabled alerts, honoring cooldowns and recording history."""
        from react_loop.tools import _fetch_stock_data

        current_time = now or datetime.now(UTC)
        get_data = fetcher or (lambda symbol: _fetch_stock_data(symbol, "1mo")[1])
        triggered: list[str] = []
        for alert in self.alerts(enabled_only=True):
            try:
                observations = get_data(alert["symbol"])
                if not observations:
                    raise ValueError("no usable market observations")
                value = self._condition_value(alert, observations)
                last_triggered = alert["last_triggered_at"]
                if last_triggered:
                    previous = datetime.fromisoformat(last_triggered)
                    if current_time < previous + timedelta(seconds=alert["cooldown_seconds"]):
                        continue
                matched = value > alert["threshold"] if alert["operator"] == "above" else value < alert["threshold"]
                if not matched:
                    continue
                label = (
                    f"{alert['symbol']} is"
                    if alert["condition_type"] == "price"
                    else f"{alert['symbol']} {alert['condition_type']} is"
                )
                message = (
                    f"Alert {alert['id']}: {label} {value:.2f} "
                    f"({alert['operator']} {alert['threshold']:.2f})"
                )
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute(
                        "UPDATE alerts SET last_triggered_at = ?, last_error = NULL WHERE id = ?",
                        (current_time.isoformat(timespec="seconds"), alert["id"]),
                    )
                    conn.execute(
                        "INSERT INTO alert_history(alert_id, triggered_at, observed_value, status, message) VALUES (?, ?, ?, ?, ?)",
                        (alert["id"], current_time.isoformat(timespec="seconds"), value, "triggered", message),
                    )
                triggered.append(message)
            except Exception as exc:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("UPDATE alerts SET last_error = ? WHERE id = ?", (str(exc), alert["id"]))
                    conn.execute(
                        "INSERT INTO alert_history(alert_id, triggered_at, observed_value, status, message) VALUES (?, ?, ?, ?, ?)",
                        (alert["id"], current_time.isoformat(timespec="seconds"), None, "error", str(exc)),
                    )
        return triggered

    def record_alert_error(self, alert_id: int | None, message: str) -> None:
        """Record a notification failure without disabling the alert."""
        if alert_id is None:
            return
        now = datetime.now(UTC).isoformat(timespec="seconds")
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("UPDATE alerts SET last_error = ? WHERE id = ?", (message, alert_id))
            conn.execute(
                "INSERT INTO alert_history(alert_id, triggered_at, observed_value, status, message) "
                "VALUES (?, ?, ?, ?, ?)",
                (alert_id, now, None, "notification_error", message),
            )
