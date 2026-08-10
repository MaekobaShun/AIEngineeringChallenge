"""PyMuPDF (fitz) based parser.

Text is extracted normally where present. Every page is *also* rasterized
to a PNG and exposed as an ImageAsset, regardless of whether text
extraction succeeded -- this covers image-only/scanned pages (社内用語集の
"画像PDF / IMG-PDF" = "OCR前提PDF") and watermarked pages uniformly: instead
of running a local OCR engine, the agent reads the rendered page directly
with vision when text extraction looks thin or the question needs visual
formatting (highlight color, layout) that text extraction can't see.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import fitz  # pymupdf

from datastel_rag.ingest.models import Block, ImageAsset, ParsedDocument

_RENDER_ZOOM = 1.5  # ~108 DPI -- kept modest so a page render stays well under
# the agent transport's message-size limit once base64-encoded for the Read tool


def parse_pdf(abs_path: str, source_rel_path: str, project_key: str, image_cache_dir: Path) -> ParsedDocument:
    doc = ParsedDocument(source_rel_path=source_rel_path, project_key=project_key, ext="pdf")
    try:
        pdf = fitz.open(abs_path)
    except Exception as e:
        doc.parse_errors.append(f"open failed: {e}")
        return doc

    image_cache_dir.mkdir(parents=True, exist_ok=True)

    for page_idx in range(pdf.page_count):
        page = pdf[page_idx]
        text = page.get_text("text")
        if text.strip():
            doc.blocks.append(
                Block(
                    block_id=f"page{page_idx}/text",
                    kind="paragraph",
                    text=text,
                    location={"page": page_idx + 1},
                    extra={"char_count": len(text.strip())},
                )
            )

        try:
            pix = page.get_pixmap(matrix=fitz.Matrix(_RENDER_ZOOM, _RENDER_ZOOM))
            png_bytes = pix.tobytes("png")
            digest = hashlib.sha1(png_bytes).hexdigest()[:12]
            out_path = image_cache_dir / f"page{page_idx}_{digest}.png"
            if not out_path.exists():
                out_path.write_bytes(png_bytes)
            doc.images.append(
                ImageAsset(
                    image_id=f"page{page_idx}/render",
                    location={"page": page_idx + 1},
                    cache_path=str(out_path),
                    width=pix.width,
                    height=pix.height,
                    caption="thin_text" if len(text.strip()) < 20 else "",
                )
            )
        except Exception as e:
            doc.parse_errors.append(f"render failed on page {page_idx}: {e}")

    doc.meta = {"num_pages": pdf.page_count}
    pdf.close()
    return doc
