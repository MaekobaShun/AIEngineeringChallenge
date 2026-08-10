"""Walks the share drive and builds a file inventory.

This is purely structural discovery -- no project- or file-specific
knowledge is encoded here. Anything about *what* a file means (its role,
its abbreviation, ...) is layered on top by the glossary matcher.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

from datastel_rag import config
from datastel_rag.ingest.fileformats import (
    IGNORED_DIR_NAMES,
    is_encrypted_office_file,
    is_temp_or_lock_file,
)
from datastel_rag.paths import display_name, iter_children, to_nfc


@dataclass
class FileEntry:
    project_key: str  # NFC display name of the owning project folder ("" for internal-management files)
    phase: str | None  # top-level phase folder under the project, e.g. "00.提案"
    rel_path: str  # NFC path relative to the share-drive root, "/"-separated
    abs_path: str  # actual filesystem path (may be NFD) -- use this for I/O
    name: str  # NFC file name
    ext: str  # lowercase, without dot
    size: int
    is_encrypted: bool


@dataclass
class ProjectEntry:
    key: str  # NFC folder name, e.g. "医療法人社団 恒一会 かえで総合病院"
    files: list[FileEntry] = field(default_factory=list)


@dataclass
class Catalog:
    projects: list[ProjectEntry] = field(default_factory=list)
    internal_files: list[FileEntry] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, s: str) -> "Catalog":
        raw = json.loads(s)
        return cls(
            projects=[ProjectEntry(key=p["key"], files=[FileEntry(**f) for f in p["files"]]) for p in raw["projects"]],
            internal_files=[FileEntry(**f) for f in raw["internal_files"]],
        )

    def iter_files(self):
        for f in self.internal_files:
            yield f
        for p in self.projects:
            yield from p.files


def _walk_files(root, project_key: str, share_root) -> list[FileEntry]:
    out: list[FileEntry] = []

    def _rec(dir_path, phase: str | None):
        for child in iter_children(dir_path):
            name = display_name(child)
            if child.is_dir():
                if name in IGNORED_DIR_NAMES:
                    continue
                next_phase = phase if phase is not None else (name if _looks_like_phase(name) else None)
                _rec(child, next_phase)
            else:
                if is_temp_or_lock_file(name):
                    continue
                ext = child.suffix.lower().lstrip(".")
                rel = to_nfc(str(child.relative_to(share_root)).replace("\\", "/"))
                out.append(
                    FileEntry(
                        project_key=project_key,
                        phase=phase,
                        rel_path=rel,
                        abs_path=str(child),
                        name=name,
                        ext=ext,
                        size=child.stat().st_size,
                        is_encrypted=is_encrypted_office_file(child, ext),
                    )
                )

    _rec(root, None)
    return out


def _looks_like_phase(name: str) -> bool:
    # phase folders are named like "00.提案" .. "06.報告書"
    return len(name) >= 3 and name[:2].isdigit() and name[2] == "."


def build_catalog() -> Catalog:
    cat = Catalog()

    cat.internal_files = _walk_files(config.INTERNAL_ROOT, project_key="", share_root=config.SHARE_DRIVE_ROOT)

    for project_dir in iter_children(config.PROJECTS_ROOT):
        if not project_dir.is_dir():
            continue
        key = display_name(project_dir)
        files = _walk_files(project_dir, project_key=key, share_root=config.SHARE_DRIVE_ROOT)
        cat.projects.append(ProjectEntry(key=key, files=files))

    return cat


def load_or_build_catalog(refresh: bool = False) -> Catalog:
    cache_path = config.CACHE_DIR / "catalog.json"
    if cache_path.exists() and not refresh:
        return Catalog.from_json(cache_path.read_text(encoding="utf-8"))
    cat = build_catalog()
    cache_path.write_text(cat.to_json(), encoding="utf-8")
    return cat


if __name__ == "__main__":
    catalog = load_or_build_catalog(refresh=True)
    n_files = sum(len(p.files) for p in catalog.projects) + len(catalog.internal_files)
    print(f"projects: {len(catalog.projects)}")
    print(f"internal files: {len(catalog.internal_files)}")
    print(f"total files: {n_files}")
    for p in catalog.projects:
        print(f"  - {p.key}: {len(p.files)} files")
