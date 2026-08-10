"""Extension classification and cheap file-type sniffing."""

from __future__ import annotations

from pathlib import Path

OLE_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
ZIP_MAGIC = b"PK\x03\x04"

OFFICE_EXTS = {"docx", "xlsx", "pptx"}
TEXT_EXTS = {"md", "txt", "py", "json", "csv", "tsv", "toml", "cfg", "ini", "lock"}
NOTEBOOK_EXTS = {"ipynb"}
PDF_EXTS = {"pdf"}
IMAGE_EXTS = {"png", "jpg", "jpeg", "gif", "bmp"}

IGNORED_DIR_NAMES = {"__pycache__", ".ipynb_checkpoints", ".git", ".venv"}


def is_temp_or_lock_file(name: str) -> bool:
    return name.startswith("~$") or name.startswith(".~")


def sniff_is_ole(path: Path) -> bool:
    """True if the file is an OLE Compound File (legacy binary container,
    also used by MS-CFB-encrypted OOXML files)."""
    try:
        with open(path, "rb") as f:
            return f.read(8) == OLE_MAGIC
    except OSError:
        return False


def is_office_file(ext: str) -> bool:
    return ext.lower().lstrip(".") in OFFICE_EXTS


def is_encrypted_office_file(path: Path, ext: str) -> bool:
    """An .xlsx/.docx/.pptx is normally a zip (starts with PK). If instead
    it sniffs as OLE, it has been saved through MS-CFB encryption and needs
    a password before any zip-based parser (openpyxl/python-docx/python-pptx)
    can touch it."""
    if not is_office_file(ext):
        return False
    return sniff_is_ole(path)
