"""Unit tests for `evaluate_report.py` artifact and table generation."""
from pathlib import Path

from evaluate_report import export_vignettes_csv, generate_markdown_report


def test_generate_markdown_report_and_csv(tmp_path: Path):
    md_file = tmp_path / "EVALUATION_REPORT.md"
    csv_file = tmp_path / "vignettes_appendix.csv"

    out_md = generate_markdown_report(output_path=md_file)
    out_csv = export_vignettes_csv(path=csv_file)

    assert out_md.exists()
    content = out_md.read_text(encoding="utf-8")
    assert "## 1. Executive KPI Summary" in content
    assert "## 3. Rule Band vs. ML Second-Opinion Band Confusion Matrix" in content
    assert "## 4. Subgroup Fairness & Privacy-Skipper Robustness" in content
    assert "100%" in content

    assert out_csv.exists()
    rows = out_csv.read_text(encoding="utf-8").strip().splitlines()
    # Header + 60 vignettes = 61 lines
    assert len(rows) == 61