"""Validated internal model for research reports."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


_SECTION_NAMES = {
    "executive summary": "summary",
    "findings": "findings",
    "catalysts and risk timeline": "catalysts",
    "price scenarios": "scenarios",
    "historical validation": "historical_validation",
    "caveats and open questions": "caveats",
    "sources": "sources",
}


@dataclass
class ResearchReport:
    """Structured report that retains the original answer for fidelity."""

    question: str
    answer: str
    as_of: str | None = None
    symbols: list[str] = field(default_factory=list)
    summary: list[str] = field(default_factory=list)
    findings: list[str] = field(default_factory=list)
    catalysts: list[str] = field(default_factory=list)
    scenarios: dict[str, Any] = field(default_factory=dict)
    historical_validation: list[str] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)
    sources: list[dict[str, Any]] = field(default_factory=list)

    def validate(self) -> None:
        """Validate required fields and any structured scenario probabilities."""
        if not self.question.strip():
            raise ValueError("research report question cannot be empty")
        if not self.answer.strip():
            raise ValueError("research report answer cannot be empty")
        for name, scenario in self.scenarios.items():
            if not isinstance(scenario, dict):
                continue
            probability = scenario.get("probability")
            if probability is None:
                continue
            if not isinstance(probability, (int, float)) or not 0 <= probability <= 1:
                raise ValueError(f"scenario {name!r} probability must be between 0 and 1")

    def to_dict(self) -> dict[str, Any]:
        """Return the stable JSON representation used by report exporters."""
        self.validate()
        return {
            "question": self.question,
            "as_of": self.as_of,
            "symbols": self.symbols,
            "summary": self.summary,
            "findings": self.findings,
            "catalysts": self.catalysts,
            "scenarios": self.scenarios,
            "historical_validation": self.historical_validation,
            "caveats": self.caveats,
            "sources": self.sources,
            "answer": self.answer,
        }


def _section_lines(answer: str) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {name: [] for name in _SECTION_NAMES.values()}
    current: str | None = None
    for line in answer.splitlines():
        heading = re.match(r"^#{1,6}\s+(.*?)\s*$", line)
        if heading:
            current = _SECTION_NAMES.get(heading.group(1).strip().lower())
            continue
        if current and line.strip():
            sections[current].append(line.strip())
    return sections

def parse_research_report(answer: str, question: str) -> ResearchReport:
    """Parse the report sections emitted by the research prompt."""
    sections = _section_lines(answer)
    sources = [
        {
            "id": index,
            "citation": line,
            "url": (
                match.group(0)
                if (match := re.search(r"https?://[^)\s]+", line))
                else None
            ),
            "source_type": "unknown",
        }
        for index, line in enumerate(sections["sources"], start=1)
    ]
    for source in sources:
        lower = source["citation"].lower()
        source["source_type"] = (
            "sec_filing"
            if "sec" in lower or "10-k" in lower or "10-q" in lower
            else "provider"
            if "yahoo" in lower or "provider" in lower
            else "web"
        )
    candidate_text = f"{question}\n{answer}"
    symbols = sorted(
        {
            match.upper()
            for match in re.findall(
                r"\b[A-Z]{2,5}(?:[.-][A-Z0-9]{1,5})?\b", candidate_text
            )
            if match.upper() not in {"THE", "AND", "FOR", "USD", "RSI", "SMA", "URL"}
        }
    )
    scenarios = {
        label.lower(): {"text": line}
        for line in sections["scenarios"]
        for label in re.findall(r"\b(Bear|Base|Bull)\b", line, re.IGNORECASE)
    }
    as_of_match = re.search(r"(?im)\bas[- ]of\s*:\s*([^;\n]+)", answer)
    report = ResearchReport(
        question=question,
        answer=answer,
        as_of=as_of_match.group(1).strip() if as_of_match else None,
        symbols=symbols,
        summary=sections["summary"],
        findings=sections["findings"],
        catalysts=sections["catalysts"],
        scenarios=scenarios,
        historical_validation=sections["historical_validation"],
        caveats=sections["caveats"],
        sources=sources,
    )
    report.validate()
    return report
