"""Parses 社内用語集.docx into structured lookups.

Two things live in that document:
  1. Eight-ish general term tables (正式名称 / 社内用語 / 補足) covering
     document types, meeting jargon, contract/billing terms, data & modeling
     terms, metrics, visual/formatting shorthand, and internal-management
     vocabulary.
  2. One project-abbreviation table (案件名 / 主略称 / 別名候補 / 補足) that
     maps each client's full legal name to its canonical short code (used in
     questions, filenames, and password derivation) plus informal aliases.

Everything here is discovered from the document at runtime -- nothing about
specific terms or project names is hardcoded.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

import docx

from datastel_rag import config
from datastel_rag.paths import resolve_child, to_nfc


@dataclass
class TermEntry:
    canonical: str  # 正式名称
    code: str  # 社内用語 (abbreviation/code)
    note: str = ""  # 補足


@dataclass
class ProjectAlias:
    full_name: str  # 案件名 (legal entity name)
    code: str  # 主略称 (canonical short code, e.g. "KAEDE")
    aliases: list[str] = field(default_factory=list)  # 別名候補
    note: str = ""


@dataclass
class Glossary:
    terms: list[TermEntry] = field(default_factory=list)
    projects: list[ProjectAlias] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, s: str) -> "Glossary":
        raw = json.loads(s)
        return cls(
            terms=[TermEntry(**t) for t in raw["terms"]],
            projects=[ProjectAlias(**p) for p in raw["projects"]],
        )

    def expand_term(self, text: str) -> list[TermEntry]:
        """Terms whose code or canonical name appears in `text` (substring match)."""
        nfc = to_nfc(text)
        hits = []
        for t in self.terms:
            if t.code and t.code in nfc:
                hits.append(t)
            elif t.canonical and t.canonical in nfc:
                hits.append(t)
        return hits

    def find_project(self, text: str) -> ProjectAlias | None:
        """Best-effort project resolution: does any code/alias/full-name
        substring of a known project appear in `text`?"""
        nfc = to_nfc(text)
        candidates = []
        for p in self.projects:
            needles = [p.code, p.full_name, *p.aliases]
            for needle in needles:
                needle = needle.strip()
                if needle and needle in nfc:
                    candidates.append((len(needle), p))
                    break
        if not candidates:
            return None
        # prefer the longest matching needle (most specific)
        candidates.sort(key=lambda t: -t[0])
        return candidates[0][1]


_PROJECT_TABLE_HEADER = {"案件名", "主略称"}


def _clean_cell(s: str) -> str:
    return to_nfc(s).replace("\xa0", " ").strip()


def parse_glossary_docx(path) -> Glossary:
    d = docx.Document(str(path))
    glossary = Glossary()

    for table in d.tables:
        if not table.rows:
            continue
        header = [_clean_cell(c.text) for c in table.rows[0].cells]
        is_project_table = set(header[:2]) >= _PROJECT_TABLE_HEADER

        for row in table.rows[1:]:
            cells = [_clean_cell(c.text) for c in row.cells]
            if is_project_table:
                if len(cells) < 3 or not cells[0]:
                    continue
                full_name, code, alias_field = cells[0], cells[1], cells[2]
                note = cells[3] if len(cells) > 3 else ""
                aliases = [a.strip() for a in alias_field.split(",") if a.strip()]
                glossary.projects.append(ProjectAlias(full_name=full_name, code=code, aliases=aliases, note=note))
            else:
                if len(cells) < 2 or not cells[0]:
                    continue
                canonical, code = cells[0], cells[1]
                note = cells[2] if len(cells) > 2 else ""
                glossary.terms.append(TermEntry(canonical=canonical, code=code, note=note))

    return glossary


def find_glossary_docx_path():
    return resolve_child(config.INTERNAL_ROOT, "社内用語集.docx")


def load_or_build_glossary(refresh: bool = False) -> Glossary:
    cache_path = config.CACHE_DIR / "glossary.json"
    if cache_path.exists() and not refresh:
        return Glossary.from_json(cache_path.read_text(encoding="utf-8"))
    glossary = parse_glossary_docx(find_glossary_docx_path())
    cache_path.write_text(glossary.to_json(), encoding="utf-8")
    return glossary


if __name__ == "__main__":
    g = load_or_build_glossary(refresh=True)
    print(f"terms: {len(g.terms)}")
    print(f"projects: {len(g.projects)}")
    for p in g.projects:
        print(f"  {p.code:10s} {p.full_name}  aliases={p.aliases}")
