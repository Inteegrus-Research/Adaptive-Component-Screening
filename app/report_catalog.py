from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_ROOT = PROJECT_ROOT / "reports"


def list_report_folders() -> list[str]:
    if not REPORTS_ROOT.exists():
        return []
    return sorted([p.name for p in REPORTS_ROOT.iterdir() if p.is_dir()])
