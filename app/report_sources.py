from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_ROOT = PROJECT_ROOT / "reports"


def available_report_sources() -> list[str]:
    if not REPORTS_ROOT.exists():
        return []
    folders: list[str] = []
    for path in sorted(REPORTS_ROOT.iterdir()):
        if path.is_dir():
            folders.append(path.name)
    return folders
