"""Central paths and settings for the pipeline.

The competition data lives outside this package (one level up, in the
sibling `share/` and `evaluation/` folders that SIGNATE distributes). We
never hardcode project or file names here -- only the root locations.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from datastel_rag.paths import resolve_child

# Every entrypoint imports config sooner or later, so load .env here once --
# google-genai (unlike the claude CLI subprocess) reads GEMINI_API_KEY
# straight from os.environ and won't pick up an unloaded .env on its own.
load_dotenv()

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
# "gemini-flash-latest" (Gemini 3.6 Flash) scored 0.733 on questions_valid.csv
# (92% Perfect on the questions it completed) vs. "gemini-flash-lite-latest"'s
# 0.533 -- meaningfully more capable at this task's multi-hop/vision/precise-
# extraction demands. Requires a paid (prepay-funded) project: the free tier's
# 5 req/min quota makes it impractical for a multi-turn tool-calling agent.
# Flash-Lite remains a cheap fallback if quota/budget gets tight.
GEMINI_MODEL = os.environ.get("DATASTEL_GEMINI_MODEL", "gemini-flash-latest")
MAX_ANSWER_TOKENS = 1000
