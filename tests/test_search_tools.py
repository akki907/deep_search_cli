"""Tests for the network-backed tools. These never touch the network."""

import pytest

from react_loop.tools import (
    MAX_SEARCH_RESULTS,
    web_search,
    wikipedia_search,
    wikipedia_summary,
)


class FakeDDGS:
    """Records the arguments the tool passes, and returns canned hits."""

    calls: list[dict] = []

    def __init__(self, timeout=None) -> None:
        self.timeout = timeout

    def text(self, query, **kwargs):
        FakeDDGS.calls.append({"query": query, **kwargs})
        return [
            {"title": "First hit", "href": "https://example.com/1", "body": "A snippet."},
            {"title": "Second hit", "href": "https://example.com/2", "body": "More text."},
        ]


@pytest.fixture
def fake_ddgs(monkeypatch):
    """Patch the search engine and hand back the recorded calls."""
    FakeDDGS.calls = []
    monkeypatch.setattr("ddgs.DDGS", FakeDDGS)
    return FakeDDGS.calls


class FakeResponse:
    """Minimal stand-in for a requests Response."""

    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        """Never fails."""

    def json(self) -> dict:
        return self._payload


def fake_requests(monkeypatch, payload: dict) -> None:
    """Answer every MediaWiki call with ``payload``."""
    monkeypatch.setattr("requests.get", lambda *_a, **_kw: FakeResponse(payload))


def search_payload(*titles: str) -> dict:
    """Build a MediaWiki search response listing ``titles``."""
    return {"query": {"search": [{"title": t} for t in titles]}}


def page_payload(title: str, extract: str) -> dict:
    """Build a MediaWiki extracts response for one article."""
    return {"query": {"pages": [{"title": title, "extract": extract}]}}


def test_web_search_returns_titles_links_and_snippets(fake_ddgs):
    out = web_search.invoke({"query": "langgraph", "max_results": 2})

    assert "First hit" in out
    assert "https://example.com/1" in out
    assert "A snippet." in out
    assert fake_ddgs[0]["query"] == "langgraph"


def test_web_search_result_count_is_capped(fake_ddgs):
    """A model asking for everything must not be obeyed literally."""
    web_search.invoke({"query": "langgraph", "max_results": 99})

    assert fake_ddgs[0]["max_results"] == MAX_SEARCH_RESULTS


def test_web_search_pins_the_duckduckgo_backend(fake_ddgs):
    """The auto backend mixes in engines that reject generic clients."""
    web_search.invoke({"query": "langgraph"})

    assert fake_ddgs[0]["backend"] == "duckduckgo"


def test_web_search_reports_failure_as_text(monkeypatch):
    """A failed search must reach the model as text, not an exception."""

    class Broken:
        def __init__(self, **_kwargs) -> None:
            pass

        def text(self, *_args, **_kwargs):
            raise RuntimeError("No results found.")

    monkeypatch.setattr("ddgs.DDGS", Broken)

    out = web_search.invoke({"query": "langgraph"})

    assert out.startswith("Error: web search failed")
    assert "No results found." in out


def test_web_search_falls_back_to_google_after_primary_failure(monkeypatch):
    calls: list[str] = []

    class FlakyDDGS:
        def __init__(self, timeout=None) -> None:
            self.timeout = timeout

        def text(self, _query, **kwargs):
            calls.append(kwargs["backend"])
            if kwargs["backend"] == "duckduckgo":
                raise RuntimeError("No results found.")
            return [{"title": "Fallback hit", "href": "https://example.com", "body": "A result."}]

    monkeypatch.setattr("ddgs.DDGS", FlakyDDGS)

    out = web_search.invoke({"query": "SQLite PostgreSQL"})

    assert calls == ["duckduckgo", "google"]
    assert "Fallback hit" in out


def test_wikipedia_search_lists_titles_with_summaries(monkeypatch):
    payload = search_payload("Ada Lovelace", "Analytical Engine")

    def respond(url, params=None, **_kwargs):
        if params.get("list") == "search":
            return FakeResponse(payload)
        title = params["titles"]
        return FakeResponse(page_payload(title, f"About {title}."))

    monkeypatch.setattr("requests.get", respond)

    out = wikipedia_search.invoke({"query": "Ada Lovelace", "max_results": 2})

    assert "Ada Lovelace\nAbout Ada Lovelace." in out
    assert "Analytical Engine\nAbout Analytical Engine." in out


def test_wikipedia_search_reports_failure_as_text(monkeypatch):
    def boom(*_args, **_kwargs):
        raise RuntimeError("HTTP 429")

    monkeypatch.setattr("requests.get", boom)

    out = wikipedia_search.invoke({"query": "Ada Lovelace"})

    assert out.startswith("Error: Wikipedia search failed")
    assert "HTTP 429" in out


def test_wikipedia_search_reports_no_articles(monkeypatch):
    fake_requests(monkeypatch, search_payload())

    out = wikipedia_search.invoke({"query": "nothing at all"})

    assert out == "No Wikipedia articles for 'nothing at all'."


def test_wikipedia_summary_returns_the_intro_extract(monkeypatch):
    fake_requests(monkeypatch, page_payload("Ada Lovelace", "An English mathematician."))

    assert wikipedia_summary.invoke({"title": "Ada Lovelace"}) == "An English mathematician."


def test_wikipedia_summary_of_a_missing_article(monkeypatch):
    fake_requests(monkeypatch, {"query": {"pages": [{"title": "Nope", "missing": True}]}})

    out = wikipedia_summary.invoke({"title": "Nope"})

    assert out == "No Wikipedia summary for 'Nope'."


def test_wikipedia_sends_an_identifying_user_agent(monkeypatch):
    """Wikimedia answers a generic client with HTTP 429 instead of results."""
    seen: dict = {}

    def respond(_url, headers=None, **_kwargs):
        seen.update(headers or {})
        return FakeResponse(page_payload("Ada Lovelace", "An English mathematician."))

    monkeypatch.setattr("requests.get", respond)
    wikipedia_summary.invoke({"title": "Ada Lovelace"})

    assert "python-requests" not in seen.get("User-Agent", "")
    assert "react-loop" in seen["User-Agent"]
