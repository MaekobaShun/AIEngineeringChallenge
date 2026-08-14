"""Normalizes embedded-image bytes to a real raster format (PNG) before
they're cached and eventually handed to a vision model.

Office documents often embed charts/diagrams as WMF or EMF (Windows
Metafile / Enhanced Metafile) -- vector formats GDI can render but that
are not valid input to any vision API. Saving that blob to disk with a
".png" extension (matching its declared content type, or just guessed from
the filename) does not make it a PNG; Gemini/Vertex correctly rejects it
("Provided image is not valid") when actually sent. Confirmed live: a
salary-comparison table embedded as EMF in 東都's データサイエンティスト調査.docx
triggered exactly this on two different test100 questions, both essential-
data images the agent otherwise had no way to read.

Pillow's WMF/EMF plugin only works via the Windows GDI backend, so this
conversion is Windows-only -- same platform constraint as the rest of this
codebase (see README's COM-automation note). Non-Windows: PIL raises and we
fall back to passing the original bytes through unchanged.
"""

from __future__ import annotations

import io

from PIL import Image

_METAFILE_CONTENT_TYPES = {"image/x-emf", "image/x-wmf", "image/emf", "image/wmf"}
_METAFILE_MAGIC_PREFIXES = (
    b"\x01\x00\x00\x00",  # EMF: iType=EMR_HEADER
    b"\xd7\xcd\xc6\x9a",  # WMF: placeable header signature
)


def _looks_like_metafile(blob: bytes, content_type: str | None) -> bool:
    if content_type and content_type.lower() in _METAFILE_CONTENT_TYPES:
        return True
    return blob[:4] in _METAFILE_MAGIC_PREFIXES


def normalize_image_bytes(blob: bytes, content_type: str | None = None) -> tuple[bytes, str]:
    """Returns (bytes_safe_to_send_to_a_vision_api, file_extension_without_dot).

    Passes standard raster formats (PNG/JPEG/...) through untouched. Converts
    WMF/EMF to PNG via Pillow. On conversion failure (e.g. non-Windows, or a
    genuinely corrupt blob), returns the original bytes with a best-guess
    extension -- callers still get *something* cacheable, just not
    guaranteed vision-API-safe in that fallback case.
    """
    if not _looks_like_metafile(blob, content_type):
        ext = (content_type.split("/")[-1] if content_type else "png").lower()
        return blob, ext

    try:
        img = Image.open(io.BytesIO(blob))
        img.load()
        out = io.BytesIO()
        img.convert("RGB").save(out, format="PNG")
        return out.getvalue(), "png"
    except Exception:
        return blob, "emf"
