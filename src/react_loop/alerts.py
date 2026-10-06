"""Alert scheduling and notification adapters."""

from __future__ import annotations

import json
import smtplib
import threading
from dataclasses import dataclass
from email.message import EmailMessage
from typing import Any, Protocol

import requests

from react_loop.market_state import MarketStateManager


class AlertNotifier(Protocol):
    """Notification adapter contract."""

    name: str

    def notify(self, message: str, alert: dict[str, Any]) -> None: ...


@dataclass
class MemoryNotifier:
    """Deterministic notifier useful for tests and local inspection."""

    name: str = "memory"

    def __post_init__(self) -> None:
        self.messages: list[dict[str, Any]] = []

    def notify(self, message: str, alert: dict[str, Any]) -> None:
        self.messages.append({"message": message, "alert": alert})


@dataclass(frozen=True)
class WebhookNotifier:
    """POST alert payloads to a configured webhook endpoint."""

    url: str
    timeout: float = 10.0
    name: str = "webhook"

    def notify(self, message: str, alert: dict[str, Any]) -> None:
        response = requests.post(
            self.url,
            json={"text": message, "alert": alert},
            timeout=self.timeout,
        )
        response.raise_for_status()


@dataclass(frozen=True)
class EmailNotifier:
    """Send alert messages through an SMTP server."""

    host: str
    port: int
    sender: str
    recipient: str
    username: str | None = None
    password: str | None = None
    use_tls: bool = True
    name: str = "email"

    def notify(self, message: str, alert: dict[str, Any]) -> None:
        email = EmailMessage()
        email["Subject"] = f"react-loop alert: {alert['symbol']}"
        email["From"] = self.sender
        email["To"] = self.recipient
        email.set_content(message)
        with smtplib.SMTP(self.host, self.port, timeout=10) as server:
            if self.use_tls:
                server.starttls()
            if self.username:
                server.login(self.username, self.password or "")
            server.send_message(email)


class AlertScheduler:
    """Evaluate persisted alerts periodically and notify configured adapters."""

    def __init__(
        self,
        manager: MarketStateManager,
        notifiers: list[AlertNotifier] | None = None,
        interval_seconds: float = 60.0,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        self.manager = manager
        self.notifiers = notifiers or [MemoryNotifier()]
        self.interval_seconds = interval_seconds

    def run_once(self, *, now=None, fetcher=None) -> list[str]:
        """Evaluate alerts once and send each triggered message."""
        messages = self.manager.evaluate_alerts(now=now, fetcher=fetcher)
        if not messages:
            return []
        alerts = self.manager.alerts()
        for message in messages:
            alert = next(
                (item for item in alerts if f"Alert {item['id']}:" in message),
                {"id": None, "symbol": "unknown"},
            )
            for notifier in self.notifiers:
                try:
                    notifier.notify(message, alert)
                except Exception as exc:
                    self.manager.record_alert_error(alert.get("id"), f"{notifier.name}: {exc}")
        return messages

    def run_forever(self, stop_event: threading.Event | None = None) -> None:
        """Run until a stop event is set; intended for an explicitly started worker."""
        stopper = stop_event or threading.Event()
        while not stopper.is_set():
            self.run_once()
            stopper.wait(self.interval_seconds)


def notifier_from_json(config: str) -> AlertNotifier:
    """Build a notifier from a JSON object for CLI configuration."""
    payload = json.loads(config)
    kind = payload.pop("type", "memory")
    if kind == "memory":
        return MemoryNotifier()
    if kind == "webhook":
        return WebhookNotifier(**payload)
    if kind == "email":
        return EmailNotifier(**payload)
    raise ValueError(f"unknown notifier type {kind!r}")
