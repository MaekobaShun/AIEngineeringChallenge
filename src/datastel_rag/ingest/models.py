"""Common intermediate representation that every format-specific parser
emits, so the indexer/agent tools don't need to know about pptx vs docx vs
xlsx internals.

Design intent: a `Block` is one addressable, human-locatable unit of text
(a paragraph, a table row, a code cell, a chart's data label, ...). Keeping
`text` (flattened, for search) separate from `runs` (formatting detail)
means the search index stays simple while formatting-sensitive questions
("what's highlighted in yellow") can still be answered by scanning `runs`.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field


@dataclass
class FormattedRun:
    text: str
    bold: bool = False
    italic: bool = False
    underline: bool = False
    strike: bool = False
    font_color: str | None = None  # "RRGGBB"
    highlight: str | None = None  # named color, e.g. "yellow", or "RRGGBB"


@dataclass
class Block:
    block_id: str  # unique within the document, e.g. "slide2/shape1/para0"
    kind: str  # "paragraph" | "heading" | "table_row" | "cell" | "code" | "output" | "note" | "title" | "chart"
    text: str  # flattened plain text, used for search
    runs: list[FormattedRun] = field(default_factory=list)
    location: dict = field(default_factory=dict)  # human-readable: {"slide": 2}, {"sheet": "train", "cell": "C5"}, {"page": 3}
    extra: dict = field(default_factory=dict)  # format-specific extras (e.g. pivot definition, autofilter criteria)


@dataclass
class ImageAsset:
    image_id: str
    location: dict
    cache_path: str  # absolute path to the extracted image bytes on disk
    width: int | None = None
    height: int | None = None
    caption: str = ""  # nearby text/alt-text if available


@dataclass
class ParsedDocument:
    source_rel_path: str  # NFC path relative to the share-drive root
    project_key: str
    ext: str
    blocks: list[Block] = field(default_factory=list)
    images: list[ImageAsset] = field(default_factory=list)
    meta: dict = field(default_factory=dict)
    parse_errors: list[str] = field(default_factory=list)

    def plain_text(self) -> str:
        return "\n".join(b.text for b in self.blocks if b.text)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, s: str) -> "ParsedDocument":
        raw = json.loads(s)
        doc = cls(
            source_rel_path=raw["source_rel_path"],
            project_key=raw["project_key"],
            ext=raw["ext"],
            meta=raw.get("meta", {}),
            parse_errors=raw.get("parse_errors", []),
        )
        doc.blocks = [Block(**b) for b in raw.get("blocks", [])]
        for b in doc.blocks:
            b.runs = [FormattedRun(**r) for r in b.runs] if b.runs and isinstance(b.runs[0], dict) else b.runs
        doc.images = [ImageAsset(**i) for i in raw.get("images", [])]
        return doc
