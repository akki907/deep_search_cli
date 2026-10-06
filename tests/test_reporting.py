"""Tests for portable report rendering and CLI export."""

import json

from react_loop.__main__ import main
from react_loop.reporting import render_report


def test_render_report_extracts_sources_as_json():
    answer = "# Executive summary\nA finding.\n\n# Sources\n1. Filing [1](https://example.test/filing)\n"

    payload = json.loads(render_report(answer, "Research Example", "json"))

    assert payload["question"] == "Research Example"
    assert payload["sources"] == ["1. Filing [1](https://example.test/filing)"]
    assert payload["answer"] == answer


def test_render_report_escapes_html():
    html = render_report("# Findings\n\n<script>alert(1)</script>", "Example", "html")

    assert "<h1>Findings</h1>" in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert "<script>alert(1)</script>" not in html


def test_cli_writes_json_report(tmp_path, capsys):
    output_path = tmp_path / "report.json"

    assert main(
        [
            "--demo",
            "--no-trace",
            "--plain",
            "--format",
            "json",
            "--output",
            str(output_path),
        ]
    ) == 0

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["question"]
    assert payload["answer"]
    assert f"Report written to {output_path}" in capsys.readouterr().out
