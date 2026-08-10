"""Parsers for formats that are already plain text, or close to it:
.py/.md/.txt/.toml/.cfg/.ini (read verbatim) and .csv/.tsv/.json (read with
a light structural summary so large tabular files don't blow up the search
index -- the agent computes over them with the code-execution tool instead
of reading them out of the block/text index row by row).
"""

from __future__ import annotations

import json

import pandas as pd

from datastel_rag.ingest.models import Block, ParsedDocument

_MAX_FULL_DUMP_ROWS = 200
_MAX_CODE_CHARS = 200_000


def parse_code_or_text(abs_path: str, source_rel_path: str, project_key: str, ext: str) -> ParsedDocument:
    doc = ParsedDocument(source_rel_path=source_rel_path, project_key=project_key, ext=ext)
    try:
        text = open(abs_path, encoding="utf-8", errors="replace").read()
    except Exception as e:
        doc.parse_errors.append(f"open failed: {e}")
        return doc

    kind = "code" if ext == "py" else "paragraph"
    if len(text) > _MAX_CODE_CHARS:
        text = text[:_MAX_CODE_CHARS] + "\n... [truncated]"
        doc.parse_errors.append("truncated: file exceeds max size for full-text indexing")
    doc.blocks.append(Block(block_id="full", kind=kind, text=text, location={}))
    doc.meta = {"char_count": len(text)}
    return doc


def parse_json(abs_path: str, source_rel_path: str, project_key: str) -> ParsedDocument:
    doc = ParsedDocument(source_rel_path=source_rel_path, project_key=project_key, ext="json")
    try:
        with open(abs_path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        doc.parse_errors.append(f"open/parse failed: {e}")
        return doc

    text = json.dumps(data, ensure_ascii=False, indent=2)
    doc.blocks.append(Block(block_id="full", kind="code", text=text, location={}))
    top_keys = list(data.keys()) if isinstance(data, dict) else None
    doc.meta = {"top_level_keys": top_keys}
    return doc


def parse_csv_like(abs_path: str, source_rel_path: str, project_key: str, ext: str, sep: str) -> ParsedDocument:
    doc = ParsedDocument(source_rel_path=source_rel_path, project_key=project_key, ext=ext)
    try:
        df = pd.read_csv(abs_path, sep=sep, encoding="utf-8-sig")
    except Exception as e:
        doc.parse_errors.append(f"read failed: {e}")
        return doc

    columns = list(df.columns)
    doc.blocks.append(
        Block(
            block_id="summary",
            kind="table_summary",
            text=f"{ext.upper()} '{source_rel_path}': {len(df)}行 x {len(columns)}列。列: {columns}",
            location={},
            extra={"columns": columns, "row_count": len(df), "dtypes": {c: str(t) for c, t in df.dtypes.items()}},
        )
    )

    if len(df) <= _MAX_FULL_DUMP_ROWS:
        for i, row in df.iterrows():
            text = " | ".join("" if pd.isna(v) else str(v) for v in row)
            doc.blocks.append(Block(block_id=f"row{i}", kind="table_row", text=text, location={"row": int(i)}))

    doc.meta = {"columns": columns, "row_count": len(df)}
    return doc
