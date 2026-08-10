"""Central paths and settings for the pipeline.

The competition data lives outside this package (one level up, in the
sibling `share/` and `evaluation/` folders that SIGNATE distributes). We
never hardcode project or file names here -- only the root locations.
"""

from __future__ import annotations

import os
from pathlib import Path

from datastel_rag.paths import resolve_child

# repo root = .../AI Engineering Challenge/pipeline ; data root = .../AI Engineering Challenge
PIPELINE_ROOT = Path(__file__).resolve().parents[2]
CHALLENGE_ROOT = PIPELINE_ROOT.parent

SHARE_ROOT = CHALLENGE_ROOT / "share"
SHARE_DRIVE_ROOT = resolve_child(SHARE_ROOT, "共有ドライブ")
QA_ROOT = resolve_child(SHARE_ROOT, "質問回答")
PROJECTS_ROOT = resolve_child(SHARE_DRIVE_ROOT, "プロジェクト")
INTERNAL_ROOT = resolve_child(SHARE_DRIVE_ROOT, "社内管理")

CACHE_DIR = PIPELINE_ROOT / "cache"
LOG_DIR = PIPELINE_ROOT / "logs"

for d in (CACHE_DIR, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)

ANTHROPIC_MODEL = os.environ.get("DATASTEL_ANTHROPIC_MODEL", "claude-sonnet-5")
GEMINI_DEV_MODEL = os.environ.get("DATASTEL_GEMINI_MODEL", "gemini-flash-latest")
MAX_ANSWER_TOKENS = 1000
