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
# google-genai (unlike the claude CLI subprocess) reads auth from os.environ
# (GEMINI_API_KEY, or GOOGLE_CLOUD_PROJECT + ADC in enterprise/Vertex mode)
# and won't pick up an unloaded .env on its own.
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
# gemini-3.6-flash scored 0.733 on questions_valid.csv (as gemini-flash-latest
# on the Developer API) vs. Flash-Lite's 0.533 -- meaningfully more capable at
# this task's multi-hop/vision/precise-extraction demands. Pin the Vertex /
# Enterprise model id rather than the Developer API "-latest" alias.
GEMINI_MODEL = os.environ.get("DATASTEL_GEMINI_MODEL", "gemini-3.6-flash")
MAX_ANSWER_TOKENS = 1000
