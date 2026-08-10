"""Japanese tokenization for BM25. fugashi (+ bundled unidic-lite) needs no
external MeCab install, which keeps this portable across the dev machine
and whatever the 検収 environment turns out to be."""

from __future__ import annotations

import re

import fugashi

_tagger = fugashi.Tagger()
_ALNUM_RE = re.compile(r"[A-Za-z0-9_.]+")


def tokenize_ja(text: str) -> list[str]:
    """Word-level tokens, lowercased. Keeps alphanumeric runs (identifiers,
    codes like 'T09', column names) intact rather than let the morphological
    tokenizer fragment them."""
    tokens = []
    for m in _ALNUM_RE.finditer(text):
        tokens.append(m.group(0).lower())
    for word in _tagger(_ALNUM_RE.sub(" ", text)):
        surface = word.surface.strip()
        if surface:
            tokens.append(surface.lower())
    return tokens
