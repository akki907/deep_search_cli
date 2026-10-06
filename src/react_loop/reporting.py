"""Render research answers as portable report formats."""

from __future__ import annotations

import html
import json
import re
from datetime import UTC, datetime
from typing import Any

from react_loop.research_report import parse_research_report


def _sources(answer: str) -> list[str]:
    """Extract source entries from the report's Sources section."""
    lines = answer.splitlines()
    in_sources = False
    found: list[str] = []
    for line in lines:
        if re.match(r"^#{1,6}\s+sources\s*$", line, re.IGNORECASE):
            in_sources = True
            continue
        if in_sources and re.match(r"^#{1,6}\s+", line):
            break
        if in_sources and line.strip():
            found.append(line.strip())
    return found
def _inline_html(text: str) -> str:
    escaped = html.escape(text)
    return re.sub(
        r"(https?://[^)\s]+)",
        r'<a href="\1">\1</a>',
        escaped,
    )


def render_report(answer: str, question: str, report_format: str) -> str:
    """Render an answer as Markdown, JSON, or standalone HTML."""
    report = parse_research_report(answer, question)
    if report_format == "markdown":
        return answer
    if report_format == "json":
        payload: dict[str, Any] = report.to_dict()
        payload["generated_at"] = datetime.now(UTC).isoformat(timespec="seconds")
        payload["sources"] = _sources(answer)
        payload["source_entries"] = report.sources
        return json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    if report_format == "html":
        return _markdown_to_html(report.answer, report.question)
    return answer


def _markdown_to_html(markdown: str, question: str) -> str:
    """Convert the report subset emitted by the research prompt to HTML."""
    output: list[str] = []
    in_list = False
    in_code = False
    paragraph: list[str] = []

    def flush_paragraph() -> None:
        if paragraph:
            output.append(f"<p>{_inline_html(' '.join(paragraph))}</p>")
            paragraph.clear()

    def close_list() -> None:
        nonlocal in_list
        if in_list:
            output.append("</ul>")
            in_list = False

    for raw_line in markdown.splitlines():
        line = raw_line.rstrip()
        if line.startswith("```"):
            flush_paragraph()
            close_list()
            if in_code:
                output.append("</code></pre>")
            else:
                output.append("<pre><code>")
            in_code = not in_code
            continue
        if in_code:
            output.append(html.escape(line) + "\n")
            continue
        heading = re.match(r"^(#{1,6})\s+(.*)$", line)
        if heading:
            flush_paragraph()
            close_list()
            level = len(heading.group(1))
            output.append(f"<h{level}>{_inline_html(heading.group(2))}</h{level}>")
            continue
        bullet = re.match(r"^\s*[-*]\s+(.*)$", line)
        if bullet:
            flush_paragraph()
            if not in_list:
                output.append("<ul>")
                in_list = True
            output.append(f"<li>{_inline_html(bullet.group(1))}</li>")
            continue
        if not line.strip():
            flush_paragraph()
            close_list()
            continue
        close_list()
        paragraph.append(line)
    flush_paragraph()
    close_list()
    if in_code:
        output.append("</code></pre>")
    title = html.escape(question or "Research report")
    body = "\n".join(output)
    return (
        "<!doctype html>\n<html lang=\"en\">\n<head>\n"
        "<meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
        f"<title>{title}</title>\n"
        "<style>body{font:16px/1.5 system-ui,sans-serif;max-width:900px;margin:2rem auto;padding:0 1rem;color:#222}"
        "pre{background:#f4f4f4;padding:1rem;overflow:auto}li{margin:.25rem 0}</style>\n"
        "</head>\n<body>\n"
        f"{body}\n</body>\n</html>\n"
    )
