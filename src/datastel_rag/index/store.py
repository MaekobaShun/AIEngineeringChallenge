"""BM25 full-text search over every parsed document's blocks.

Retrieval strategy for this dataset is deliberately two-layered, not a
single global vector search:
  1. Most questions name a project (explicitly or via a 社内用語集 code/
     alias) -- `catalog.glossary.find_project` resolves that first, and
     `search(..., project_key=...)` scopes BM25 to just that project's
     blocks. This is far more precise than semantic search for this data,
     since answers usually live in one identifiable project's files.
  2. Cross-project questions (略称の一覧を挙げよ, 全案件の合計, ...) fall
     back to an unscoped search across everything.

No embedding model is wired in yet (see README) -- BM25 + project scoping
covers the keyword-heavy, project/file-named question style seen in the
sample questions; add a semantic layer later only if evaluation shows BM25
missing paraphrased queries.
"""

from __future__ import annotations

from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from datastel_rag.catalog.glossary import Glossary
from datastel_rag.catalog.scanner import Catalog
from datastel_rag.ingest.dispatch import parse_all
from datastel_rag.ingest.models import ParsedDocument
from datastel_rag.index.tokenize import tokenize_ja


@dataclass
class SearchHit:
    score: float
    project_key: str
    rel_path: str
    ext: str
    block_id: str
    kind: str
    text: str
    location: dict


class SearchIndex:
    def __init__(self) -> None:
        self._meta: list[dict] = []
        self._tokens: list[list[str]] = []
        self._bm25: BM25Okapi | None = None
        self.documents: dict[str, ParsedDocument] = {}  # rel_path -> full parsed doc

    def build(self, catalog: Catalog, glossary: Glossary, force: bool = False) -> None:
        self._meta.clear()
        self._tokens.clear()
        self.documents.clear()

        for entry, doc in parse_all(catalog, glossary, force=force):
            self.documents[entry.rel_path] = doc
            for block in doc.blocks:
                if not block.text.strip():
                    continue
                self._meta.append(
                    {
                        "project_key": entry.project_key,
                        "rel_path": entry.rel_path,
                        "ext": entry.ext,
                        "block_id": block.block_id,
                        "kind": block.kind,
                        "text": block.text,
                        "location": block.location,
                    }
                )
                self._tokens.append(tokenize_ja(block.text))

        self._bm25 = BM25Okapi(self._tokens) if self._tokens else None

    def search(self, query: str, project_key: str | None = None, kind: str | None = None, top_k: int = 10) -> list[SearchHit]:
        if self._bm25 is None:
            return []
        q_tokens = tokenize_ja(query)
        scores = self._bm25.get_scores(q_tokens)

        ranked_idx = sorted(range(len(scores)), key=lambda i: -scores[i])
        results: list[SearchHit] = []
        for i in ranked_idx:
            if scores[i] <= 0:
                break
            meta = self._meta[i]
            if project_key and meta["project_key"] != project_key:
                continue
            if kind and meta["kind"] != kind:
                continue
            results.append(SearchHit(score=float(scores[i]), **meta))
            if len(results) >= top_k:
                break
        return results

    def get_document(self, rel_path: str) -> ParsedDocument | None:
        return self.documents.get(rel_path)


_singleton: SearchIndex | None = None


def get_index(catalog: Catalog, glossary: Glossary, force: bool = False) -> SearchIndex:
    global _singleton
    if _singleton is None or force:
        _singleton = SearchIndex()
        _singleton.build(catalog, glossary, force=force)
    return _singleton


if __name__ == "__main__":
    from datastel_rag.catalog.glossary import load_or_build_glossary
    from datastel_rag.catalog.scanner import load_or_build_catalog

    cat = load_or_build_catalog()
    gl = load_or_build_glossary()
    idx = SearchIndex()
    idx.build(cat, gl)
    print(f"indexed blocks: {len(idx._meta)}")

    for q, proj in [("モビリティ需要の要因分析 マーカー", "AOSHIO"), ("PivotTable ALP 平均", None)]:
        project_key = None
        if proj:
            alias = next((p for p in gl.projects if p.code == proj), None)
            project_key = alias.full_name if alias else None
        hits = idx.search(q, project_key=project_key, top_k=5)
        print(f"\nquery={q!r} project={project_key}")
        for h in hits:
            print(f"  {h.score:.2f} {h.rel_path} [{h.block_id}] {h.text[:60]!r}")
