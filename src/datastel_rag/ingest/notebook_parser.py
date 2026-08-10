"""nbformat-based parser: code cells, markdown cells, and their outputs
(text/stream and image) in execution order."""

from __future__ import annotations

import base64
import hashlib
from pathlib import Path

import nbformat

from datastel_rag.ingest.models import Block, ImageAsset, ParsedDocument


def _save_b64_image(b64_data: str, cache_dir: Path, image_id: str, ext: str = "png") -> str:
    cache_dir.mkdir(parents=True, exist_ok=True)
    raw = base64.b64decode(b64_data)
    digest = hashlib.sha1(raw).hexdigest()[:12]
    out_path = cache_dir / f"{image_id.replace('/', '_')}_{digest}.{ext}"
    if not out_path.exists():
        out_path.write_bytes(raw)
    return str(out_path)


def parse_ipynb(abs_path: str, source_rel_path: str, project_key: str, image_cache_dir: Path) -> ParsedDocument:
    doc = ParsedDocument(source_rel_path=source_rel_path, project_key=project_key, ext="ipynb")
    try:
        nb = nbformat.read(abs_path, as_version=4)
    except Exception as e:
        doc.parse_errors.append(f"open failed: {e}")
        return doc

    for ci, cell in enumerate(nb.cells):
        loc = {"cell": ci, "cell_type": cell.cell_type}
        src = cell.get("source", "")
        if src.strip():
            kind = "code" if cell.cell_type == "code" else "paragraph"
            doc.blocks.append(Block(block_id=f"cell{ci}/source", kind=kind, text=src, location=loc))

        for oi, output in enumerate(cell.get("outputs", []) or []):
            otype = output.get("output_type")
            if otype in ("stream",):
                text = "".join(output.get("text", []))
                if text.strip():
                    doc.blocks.append(Block(block_id=f"cell{ci}/out{oi}", kind="output", text=text, location=loc))
            elif otype in ("execute_result", "display_data"):
                data = output.get("data", {})
                if "text/plain" in data:
                    text = "".join(data["text/plain"])
                    if text.strip():
                        doc.blocks.append(Block(block_id=f"cell{ci}/out{oi}/text", kind="output", text=text, location=loc))
                for mime, b64 in data.items():
                    if mime.startswith("image/"):
                        ext = mime.split("/")[-1]
                        img_id = f"cell{ci}/out{oi}"
                        try:
                            path = _save_b64_image(b64 if isinstance(b64, str) else "".join(b64), image_cache_dir, img_id, ext)
                            doc.images.append(ImageAsset(image_id=img_id, location=loc, cache_path=path))
                        except Exception as e:
                            doc.parse_errors.append(f"image decode failed at {img_id}: {e}")
            elif otype == "error":
                text = "\n".join(output.get("traceback", []))
                if text.strip():
                    doc.blocks.append(Block(block_id=f"cell{ci}/out{oi}/error", kind="output", text=text, location=loc))

    doc.meta = {"num_cells": len(nb.cells), "kernel": nb.get("metadata", {}).get("kernelspec", {}).get("name", "")}
    return doc
