"""Routes a FileEntry to the right format-specific parser, transparently
handling decryption first and caching the resulting ParsedDocument to disk.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from datastel_rag import config
from datastel_rag.catalog.glossary import Glossary
from datastel_rag.catalog.scanner import Catalog, FileEntry
from datastel_rag.ingest import decrypt
from datastel_rag.ingest.docx_parser import parse_docx
from datastel_rag.ingest.models import ImageAsset, ParsedDocument
from datastel_rag.ingest.notebook_parser import parse_ipynb
from datastel_rag.ingest.pdf_parser import parse_pdf
from datastel_rag.ingest.pptx_parser import parse_pptx
from datastel_rag.ingest.simple_parsers import parse_code_or_text, parse_csv_like, parse_json
from datastel_rag.ingest.xlsx_parser import parse_xlsx

_TEXTLIKE_EXTS = {"md", "txt", "py", "toml", "cfg", "ini"}
_IMAGE_EXTS = {"png", "jpg", "jpeg", "gif", "bmp"}
# dependency lock files carry no QA-relevant content and can be huge; skip indexing them
_SKIP_EXTS = {"lock"}

_PARSED_CACHE_DIR = config.CACHE_DIR / "parsed"
_DECRYPTED_CACHE_DIR = config.CACHE_DIR / "decrypted"
_IMAGE_CACHE_ROOT = config.CACHE_DIR / "images"


def _safe_key(rel_path: str) -> str:
    digest = hashlib.sha1(rel_path.encode("utf-8")).hexdigest()[:16]
    tail = rel_path.replace("/", "_").replace("\\", "_")[-60:]
    return f"{digest}_{tail}"


def _parse_image_file(abs_path: str, source_rel_path: str, project_key: str, ext: str) -> ParsedDocument:
    doc = ParsedDocument(source_rel_path=source_rel_path, project_key=project_key, ext=ext)
    doc.images.append(ImageAsset(image_id="self", location={}, cache_path=abs_path))
    return doc


def parse_entry(entry: FileEntry, catalog: Catalog, glossary: Glossary, force: bool = False) -> ParsedDocument:
    key = _safe_key(entry.rel_path)
    cache_path = _PARSED_CACHE_DIR / f"{key}.json"
    if cache_path.exists() and not force:
        return ParsedDocument.from_json(cache_path.read_text(encoding="utf-8"))

    work_path = entry.abs_path
    doc: ParsedDocument | None = None

    if entry.is_encrypted:
        result = decrypt.decrypt_entry(entry, catalog, glossary)
        if not result.success:
            doc = ParsedDocument(source_rel_path=entry.rel_path, project_key=entry.project_key, ext=entry.ext)
            doc.parse_errors.append("encrypted; automatic password derivation failed (needs agent-level retry)")
        else:
            _DECRYPTED_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            tmp_path = _DECRYPTED_CACHE_DIR / f"{key}.{entry.ext}"
            tmp_path.write_bytes(result.data)
            work_path = str(tmp_path)

    if doc is None:
        image_dir = _IMAGE_CACHE_ROOT / key
        ext = entry.ext
        try:
            if ext == "docx":
                doc = parse_docx(work_path, entry.rel_path, entry.project_key)
            elif ext == "pptx":
                doc = parse_pptx(work_path, entry.rel_path, entry.project_key, image_dir)
            elif ext == "xlsx":
                doc = parse_xlsx(work_path, entry.rel_path, entry.project_key)
            elif ext == "pdf":
                doc = parse_pdf(work_path, entry.rel_path, entry.project_key, image_dir)
            elif ext == "ipynb":
                doc = parse_ipynb(work_path, entry.rel_path, entry.project_key, image_dir)
            elif ext == "json":
                doc = parse_json(work_path, entry.rel_path, entry.project_key)
            elif ext == "csv":
                doc = parse_csv_like(work_path, entry.rel_path, entry.project_key, ext, ",")
            elif ext == "tsv":
                doc = parse_csv_like(work_path, entry.rel_path, entry.project_key, ext, "\t")
            elif ext in _TEXTLIKE_EXTS:
                doc = parse_code_or_text(work_path, entry.rel_path, entry.project_key, ext)
            elif ext in _IMAGE_EXTS:
                doc = _parse_image_file(work_path, entry.rel_path, entry.project_key, ext)
            elif ext in _SKIP_EXTS:
                doc = ParsedDocument(source_rel_path=entry.rel_path, project_key=entry.project_key, ext=ext)
            else:
                doc = ParsedDocument(source_rel_path=entry.rel_path, project_key=entry.project_key, ext=ext)
                doc.parse_errors.append(f"unsupported extension: {ext}")
        except Exception as e:
            doc = ParsedDocument(source_rel_path=entry.rel_path, project_key=entry.project_key, ext=ext)
            doc.parse_errors.append(f"unhandled parser exception: {type(e).__name__}: {e}")

    _PARSED_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(doc.to_json(), encoding="utf-8")
    return doc


def parse_all(catalog: Catalog, glossary: Glossary, force: bool = False):
    for entry in catalog.iter_files():
        yield entry, parse_entry(entry, catalog, glossary, force=force)


if __name__ == "__main__":
    from datastel_rag.catalog.glossary import load_or_build_glossary
    from datastel_rag.catalog.scanner import load_or_build_catalog

    cat = load_or_build_catalog()
    gl = load_or_build_glossary()

    n_total = 0
    n_err = 0
    err_by_ext: dict[str, int] = {}
    for entry, doc in parse_all(cat, gl):
        n_total += 1
        if doc.parse_errors:
            n_err += 1
            err_by_ext[entry.ext] = err_by_ext.get(entry.ext, 0) + 1
            print(f"[ERR] {entry.rel_path}: {doc.parse_errors}")

    print(f"\ntotal={n_total} errors={n_err}")
    print("errors by ext:", err_by_ext)
