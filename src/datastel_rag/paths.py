"""Filesystem helpers that are safe with respect to Unicode normalization.

The distributed share drive was zipped on macOS, so directory/file names
containing dakuten/handakuten kana (e.g. `ドライブ`) are stored on disk in
NFD (decomposed) form. Any Japanese text typed directly into source code is
normally NFC (composed), so naive `Path(parent, "共有ドライブ")` construction
silently fails to match the real filesystem entry.

Rule: never hardcode a Japanese path segment and hand it straight to Path.
Discover the real entry with `resolve_child`/`iter_children`, or compare
display names via `to_nfc`.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Iterator
from pathlib import Path


def to_nfc(s: str) -> str:
    """Normalize for display, comparison, search-indexing, and glossary matching."""
    return unicodedata.normalize("NFC", s)


def resolve_child(parent: Path, name: str) -> Path:
    """Find `parent`'s child whose NFC-normalized name matches `name`.

    Use this instead of `parent / name` whenever `name` contains non-ASCII
    text you typed by hand (as opposed to a name obtained from `iterdir`).
    """
    target = to_nfc(name)
    for child in parent.iterdir():
        if to_nfc(child.name) == target:
            return child
    raise FileNotFoundError(f"No child named {name!r} (NFC) under {parent}")


def iter_children(parent: Path) -> Iterator[Path]:
    """Sorted iteration by NFC display name, for deterministic traversal order."""
    yield from sorted(parent.iterdir(), key=lambda p: to_nfc(p.name))


def display_name(p: Path) -> str:
    return to_nfc(p.name)
