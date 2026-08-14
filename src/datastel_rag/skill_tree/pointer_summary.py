"""Deterministic, structure-only leaf summaries for the skill tree.

Not LLM-generated content summaries on purpose: this competition's questions
are overwhelmingly extractive (a specific cell value, a highlight color, a
contract clause number, a cross-project total) rather than gist-level. A
summary is lossy compression -- the moment a leaf's "summary" describes a
document instead of pointing into it, two things go wrong: (1) the exact
values these questions need are gone, and (2) the more plausible the summary
reads, the more likely the agent trusts it and never opens the real file,
which is a hallucination risk RAPTOR-style hierarchical summarization is
already known for on pinpoint questions.

So this only ever answers "where is X" (sheet names, column headers, heading
hierarchy, function names, slide count) never "what does X mean" -- entirely
derived from data ingest/dispatch.py's parsers already extracted, so it's
free (no LLM call), lossless (nothing summarized away), and generalizes to
any future file automatically (same parser, not per-file authored text).
"""

from __future__ import annotations

import re

from datastel_rag.catalog.glossary import Glossary
from datastel_rag.catalog.scanner import Catalog, FileEntry
from datastel_rag.ingest.dispatch import parse_entry

_DEF_CLASS_RE = re.compile(r"^\s*(?:async\s+def|def|class)\s+(\w+)", re.MULTILINE)
_MAX_LIST_ITEMS = 15
_MAX_ITEM_CHARS = 30


def _truncate_list(items: list[str], n: int = _MAX_LIST_ITEMS) -> str:
    shown = items[:n]
    suffix = f" (+{len(items) - n})" if len(items) > n else ""
    return str(shown) + suffix


def _funcs_and_classes(code_text: str) -> str:
    names = list(dict.fromkeys(_DEF_CLASS_RE.findall(code_text)))
    return ", ".join(names[:20]) + (f" (+{len(names) - 20})" if len(names) > 20 else "")


def pointer_summary(entry: FileEntry, catalog: Catalog, glossary: Glossary) -> str:
    try:
        doc = parse_entry(entry, catalog, glossary)
    except Exception as e:
        return f"{entry.name}（{entry.phase}）。[取得失敗: {type(e).__name__}]"

    if doc.parse_errors and not doc.blocks and not doc.images:
        return f"{entry.name}（{entry.phase}）。[解析エラー: {doc.parse_errors[0][:60]}]"

    bits: list[str] = []
    ext = entry.ext

    if ext == "xlsx":
        sheet_names = doc.meta.get("sheet_names", [])
        if sheet_names:
            bits.append(f"シート={sheet_names}")
        for b in doc.blocks:
            if b.kind == "table_summary":
                cols = b.extra.get("columns", [])
                bits.append(f"[{b.location.get('sheet')}]列={_truncate_list([str(c) for c in cols])}({b.extra.get('row_count')}行)")
            elif b.kind == "pivot_table":
                row_fields = b.extra.get("row_fields", [])
                data_fields = list((b.extra.get("data_fields") or {}).values())
                bits.append(f"[Pivot]グループ化={row_fields} 集計={data_fields}")
            elif b.kind == "filter":
                bits.append(f"[フィルタ]{b.text[:50]}")

    elif ext in ("csv", "tsv"):
        for b in doc.blocks:
            if b.kind == "table_summary":
                cols = b.extra.get("columns", [])
                bits.append(f"列={_truncate_list([str(c) for c in cols], 25)}({b.extra.get('row_count')}行)")

    elif ext == "docx":
        headings = [b.text.strip()[:_MAX_ITEM_CHARS] for b in doc.blocks if b.kind == "heading" and b.text.strip()]
        if headings:
            bits.append("見出し=" + _truncate_list(headings, 20))
        n_table_rows = sum(1 for b in doc.blocks if b.kind == "table_row")
        if n_table_rows:
            bits.append(f"表({n_table_rows}行分)")
        if doc.images:
            bits.append(f"埋め込み画像{len(doc.images)}件")

    elif ext == "pptx":
        n_slides = doc.meta.get("num_slides")
        if n_slides:
            bits.append(f"{n_slides}スライド")
        first_text_by_slide: dict[int, str] = {}
        for b in doc.blocks:
            slide = b.location.get("slide")
            if slide is not None and slide not in first_text_by_slide and b.text.strip():
                first_text_by_slide[slide] = b.text.strip().split("\n")[0][:_MAX_ITEM_CHARS]
        if first_text_by_slide:
            titles = [f"S{s}:{t}" for s, t in sorted(first_text_by_slide.items())[:_MAX_LIST_ITEMS]]
            bits.append("各スライド冒頭=" + " | ".join(titles))
        if doc.images:
            bits.append(f"埋め込み画像{len(doc.images)}件")

    elif ext == "ipynb":
        md_headers = [
            line.strip()[:_MAX_ITEM_CHARS]
            for b in doc.blocks
            if b.kind != "code"
            for line in b.text.split("\n")
            if line.strip().startswith("#")
        ]
        if md_headers:
            bits.append("見出し=" + _truncate_list(md_headers, 20))
        code_text = "\n".join(b.text for b in doc.blocks if b.kind == "code")
        funcs = _funcs_and_classes(code_text)
        if funcs:
            bits.append(f"関数/クラス={funcs}")

    elif ext == "py":
        code_text = "\n".join(b.text for b in doc.blocks if b.kind == "code")
        funcs = _funcs_and_classes(code_text)
        if funcs:
            bits.append(f"関数/クラス={funcs}")

    elif ext == "pdf":
        n_pages = doc.meta.get("num_pages")
        if n_pages:
            bits.append(f"{n_pages}ページ")
        page_firsts = []
        for b in doc.blocks[:20]:
            page = b.location.get("page")
            first_line = b.text.strip().split("\n")[0][:_MAX_ITEM_CHARS] if b.text.strip() else ""
            if first_line:
                page_firsts.append(f"p{page}:{first_line}")
        if page_firsts:
            bits.append("各ページ冒頭=" + " | ".join(page_firsts[:_MAX_LIST_ITEMS]))
        if any(img.caption == "thin_text" for img in doc.images):
            bits.append("画像/スキャンページあり")

    elif ext == "json":
        keys = doc.meta.get("top_level_keys")
        if keys:
            bits.append(f"トップレベルキー={keys}")

    elif ext == "md":
        headers = [line.strip()[:_MAX_ITEM_CHARS] for b in doc.blocks for line in b.text.split("\n") if line.strip().startswith("#")]
        if headers:
            bits.append("見出し=" + _truncate_list(headers, 20))

    base = " / ".join(bits) if bits else ""
    return f"{entry.name}（{entry.phase}）。{base}".strip("。 ") + "。" if base else f"{entry.name}（{entry.phase}）"
