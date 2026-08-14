"""Provider-agnostic tool logic, shared by the Claude Agent SDK wrapper
(tools.py) and the Gemini function-calling wrapper (gemini_agent.py).

Each *_impl function does the actual work and returns plain data (usually
a str). The provider-specific modules only handle schema declaration and
wrapping the return value into whatever shape that provider's tool-calling
protocol expects -- Claude wants {"content": [...]}, Gemini wants a
JSON-serializable dict for the function response.

get_document_impl separately returns image cache paths (rather than only
mentioning them in the text) because the two providers surface images
differently: Claude has a built-in Read tool the agent can call on any
path itself; Gemini has no such built-in, so gemini_agent.py exposes a
dedicated view_image tool and attaches the bytes as inline image content.
"""

from __future__ import annotations

import builtins
import contextlib
import difflib
import io
import mimetypes
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from datastel_rag import config
from datastel_rag.catalog.glossary import Glossary
from datastel_rag.catalog.scanner import Catalog
from datastel_rag.ingest import decrypt
from datastel_rag.ingest.dispatch import parse_entry
from datastel_rag.index.store import SearchIndex
from datastel_rag.skill_tree.tree import SkillTree

SCRATCH_DIR = config.CACHE_DIR / "scratch"

_MAX_DOC_CHARS = 60_000
_MAX_SNIPPET_CHARS = 350


@dataclass
class ToolContext:
    index: SearchIndex
    catalog: Catalog
    glossary: Glossary
    capture: dict = field(default_factory=dict)
    skill_tree: SkillTree | None = None
    retrieval_mode: str = "bm25"  # bm25 | skill_nav


def resolve_project_key(glossary: Glossary, text: str | None) -> str | None:
    if not text:
        return None
    alias = next((p for p in glossary.projects if p.code == text or p.full_name == text), None)
    if alias:
        return alias.full_name
    resolved = glossary.find_project(text)
    return resolved.full_name if resolved else text


def search_documents_impl(ctx: ToolContext, query: str, project: str | None = None, top_k: int = 8) -> str:
    project_key = resolve_project_key(ctx.glossary, project)
    hits = ctx.index.search(query, project_key=project_key, top_k=int(top_k or 8))
    if not hits:
        return "該当する結果が見つかりませんでした。project指定を外す、またはクエリを変えて再検索してください。"
    lines = []
    for h in hits:
        snippet = h.text[:_MAX_SNIPPET_CHARS]
        lines.append(f"[score={h.score:.1f}] {h.rel_path}\n  block={h.block_id} kind={h.kind} location={h.location}\n  {snippet}")
    return "\n\n".join(lines)


def _format_document(rel_path: str, doc) -> tuple[str, list[str]]:
    lines = [f"=== {rel_path} (.{doc.ext}) ==="]
    if doc.parse_errors:
        lines.append(f"[注意: 解析時の問題: {doc.parse_errors}]")
    for b in doc.blocks:
        flags = []
        for r in b.runs:
            if r.bold:
                flags.append("太字")
            if r.italic:
                flags.append("イタリック")
            if r.underline:
                flags.append("下線")
            if r.highlight:
                flags.append(f"ハイライト:{r.highlight}")
            if r.font_color:
                flags.append(f"文字色:{r.font_color}")
        flag_str = f" <{'/'.join(sorted(set(flags)))}>" if flags else ""
        lines.append(f"[{b.block_id} {b.kind} {b.location}]{flag_str} {b.text}")

    image_paths = [img.cache_path for img in doc.images]
    if image_paths:
        lines.append("\n--- 画像/レンダリング済みページ ---")
        for img in doc.images:
            lines.append(f"  {img.location} -> {img.cache_path}")

    text = "\n".join(lines)
    if len(text) > _MAX_DOC_CHARS:
        text = text[:_MAX_DOC_CHARS] + f"\n... [切り詰め: 全{len(lines)}ブロック中一部のみ表示。必要ならsearch_documentsで該当箇所に絞り込むこと]"
    return text, image_paths


def _get_parsed_document(ctx: ToolContext, rel_path: str):
    doc = ctx.index.get_document(rel_path)
    if doc is not None:
        return doc
    entry = ctx.catalog.find_by_rel_path(rel_path)
    if entry is None:
        return None
    return parse_entry(entry, ctx.catalog, ctx.glossary)


def get_document_impl(ctx: ToolContext, rel_path: str) -> tuple[str, list[str]]:
    """Returns (formatted_text, image_cache_paths)."""
    doc = _get_parsed_document(ctx, rel_path)
    if doc is None:
        return f"ファイルが見つかりません: {rel_path}", []
    return _format_document(rel_path, doc)


def diff_documents_impl(ctx: ToolContext, rel_path_a: str, rel_path_b: str) -> str:
    """Line-level diff between two documents' extracted text blocks (e.g. an
    old vs new version of the same file). Deterministic (difflib), not the
    model eyeballing two long documents itself.

    Caveat confirmed in practice: this only diffs extracted text, so a block
    that looks "added" may just be a picture (e.g. an embedded chart/table
    image) in the old version being re-rendered as native text/shapes in the
    new one -- same content, different format, not a substantive change. One
    real case: text diff showed a "4.1-4.5 breakdown" table as newly added,
    but the old file's slide had the identical table baked into a PICTURE
    shape the whole time (confirmed by opening that image directly) -- so the
    real answer was "no substantive change", not "section added". Always
    check for a same-location image on the old side via view_image before
    trusting an apparent addition."""
    doc_a = _get_parsed_document(ctx, rel_path_a)
    if doc_a is None:
        return f"ファイルが見つかりません: {rel_path_a}"
    doc_b = _get_parsed_document(ctx, rel_path_b)
    if doc_b is None:
        return f"ファイルが見つかりません: {rel_path_b}"

    texts_a = [b.text for b in doc_a.blocks if b.text.strip()]
    texts_b = [b.text for b in doc_b.blocks if b.text.strip()]
    diff_lines = list(difflib.unified_diff(texts_a, texts_b, fromfile=rel_path_a, tofile=rel_path_b, lineterm=""))
    if not diff_lines:
        return "テキストブロックの構成に差分はありません(内容は同一)。"

    text = "\n".join(diff_lines)
    if len(text) > _MAX_DOC_CHARS:
        text = text[:_MAX_DOC_CHARS] + "\n...[切り詰め: 差分が大きいため一部のみ表示]"
    return (
        "行頭 '-' は旧版のみ、'+' は新版のみに存在する行(=追加/削除された内容)。"
        "レイアウト変更で同じ内容がブロックの区切り方だけ変わった箇所もノイズとして出るため、"
        "本当に内容が変わった箇所かを見極めること。\n\n" + text
    )


def list_children_impl(ctx: ToolContext, node_id: str | None = None) -> str:
    if ctx.skill_tree is None:
        return "skill tree が読み込まれていません。retrieval_mode=skill_nav で起動してください。"
    return ctx.skill_tree.list_children(node_id or "root")


def resolve_project_impl(ctx: ToolContext, text: str) -> str:
    p = ctx.glossary.find_project(text)
    if p is None:
        return "一致する案件が見つかりませんでした。list_projectsで一覧を確認してください。"
    return f"正式名称: {p.full_name}\n主略称: {p.code}\n別名候補: {p.aliases}\n補足: {p.note}"


def list_projects_impl(ctx: ToolContext) -> str:
    return "\n".join(f"{p.code}: {p.full_name} (別名: {', '.join(p.aliases)})" for p in ctx.glossary.projects)


def expand_glossary_terms_impl(ctx: ToolContext, text: str) -> str:
    hits = ctx.glossary.expand_term(text)
    if not hits:
        return "該当する社内用語は見つかりませんでした。"
    return "\n".join(f"{h.code} = {h.canonical}" + (f" ({h.note})" if h.note else "") for h in hits)


def list_project_files_impl(ctx: ToolContext, project: str) -> str:
    project_key = resolve_project_key(ctx.glossary, project)
    proj = ctx.catalog.find_project(project_key) if project_key else None
    if proj is None:
        return f"案件が見つかりません: {project}"
    return "\n".join(f"{f.rel_path}  (ext={f.ext}, phase={f.phase}, encrypted={f.is_encrypted})" for f in proj.files)


def _make_restricted_open(scratch_dir: Path, real_open):
    """run_python is meant for read-only pandas/numpy computation over the
    share drive's raw tables. It's still plain exec(), and an agent has
    legitimately used it to crop an image with PIL for a closer look, so
    writes are confined to a scratch dir under cache/ instead of landing
    wherever the process cwd happens to be.

    Shadowing `open` only in the exec namespace's __builtins__ does NOT
    catch writes made by library code the exec'd script calls into (e.g.
    PIL.Image.save(), pandas.DataFrame.to_csv()) -- those resolve `open`
    through the real `builtins` module, not the caller's local namespace.
    So this patches `builtins.open` itself for the duration of the call
    (see the try/finally in run_python_impl) to actually catch every path.
    """

    def restricted_open(file, mode="r", *args, **kwargs):
        if any(m in mode for m in ("w", "a", "x", "+")):
            p = Path(file)
            p = p if p.is_absolute() else scratch_dir / p
            p = p.resolve()
            scratch_resolved = scratch_dir.resolve()
            if p != scratch_resolved and scratch_resolved not in p.parents:
                raise PermissionError(f"run_pythonからの書き込みは{scratch_dir}配下のみ許可されています: {file}")
            p.parent.mkdir(parents=True, exist_ok=True)
            return real_open(p, mode, *args, **kwargs)
        return real_open(file, mode, *args, **kwargs)

    return restricted_open


def run_python_impl(ctx: ToolContext, code: str) -> str:
    def read_table(rel_path: str, sheet=None):
        entry = ctx.catalog.find_by_rel_path(rel_path)
        if entry is None:
            raise FileNotFoundError(rel_path)
        if entry.ext == "csv":
            return pd.read_csv(entry.abs_path, encoding="utf-8-sig")
        if entry.ext == "tsv":
            return pd.read_csv(entry.abs_path, sep="\t", encoding="utf-8-sig")
        if entry.ext == "xlsx":
            return pd.read_excel(entry.abs_path, sheet_name=sheet if sheet is not None else 0)
        raise ValueError(f"read_tableが対応していない拡張子です: {entry.ext}")

    def resolve_path(rel_path: str) -> str:
        """The real, OS-openable absolute path for any file in the catalog --
        the share drive was zipped on macOS, so directory names with dakuten/
        handakuten kana (e.g. 共有ドライブ) are stored NFD-decomposed on disk.
        A hand-typed path literal in generated code is normally NFC and
        silently fails to match (PackageNotFoundError / FileNotFoundError),
        even though the file is right there -- confirmed live: an agent's own
        pptx.Presentation(hardcoded_path) calls failed this way on every
        attempt while investigating a diff_documents result, and it never
        surfaced as anything other than empty/no-diff output. Always go
        through this (or read_table) instead of typing a path yourself."""
        entry = ctx.catalog.find_by_rel_path(rel_path)
        if entry is None:
            raise FileNotFoundError(rel_path)
        return entry.abs_path

    SCRATCH_DIR.mkdir(parents=True, exist_ok=True)

    global_ns = {
        "pd": pd,
        "np": np,
        "read_table": read_table,
        "resolve_path": resolve_path,
        "SCRATCH_DIR": str(SCRATCH_DIR),
    }
    local_ns: dict = {}
    buf = io.StringIO()
    real_open = builtins.open
    real_cwd = os.getcwd()
    # Belt and suspenders against relative-path writes escaping the scratch dir:
    # builtins.open patching alone doesn't catch every write path -- Path.write_bytes()/
    # write_text() and pathlib.Path.open() go through io.open()/os-level calls, not the
    # `builtins.open` name, so they sailed straight past the patch below and once
    # dumped ~45 PNGs (cropped chart images from a real question) into the pipeline
    # root as a literal "SCRATCH_DIR" folder. chdir is a process-wide (not per-call)
    # property, but exec() here is fully synchronous with no `await` inside it, so no
    # other coroutine can run on this single-threaded asyncio loop while we're inside
    # it -- safe under the concurrency model this is actually used with.
    os.chdir(SCRATCH_DIR)
    builtins.open = _make_restricted_open(SCRATCH_DIR, real_open)
    try:
        with contextlib.redirect_stdout(buf):
            exec(code, global_ns, local_ns)
    except Exception as e:
        return f"実行エラー: {type(e).__name__}: {e}\n\n--- stdout ---\n{buf.getvalue()}"
    finally:
        builtins.open = real_open
        os.chdir(real_cwd)

    out = buf.getvalue()
    if "result" in local_ns:
        out += f"\nresult = {local_ns['result']!r}"
    if not out.strip():
        out = "(出力なし。print()するか`result`変数に代入してください)"
    return out


def attempt_decrypt_impl(ctx: ToolContext, rel_path: str, extra_passwords: list[str] | None = None) -> str:
    entry = ctx.catalog.find_by_rel_path(rel_path)
    if entry is None:
        return f"見つかりません: {rel_path}"
    result = decrypt.decrypt_entry(entry, ctx.catalog, ctx.glossary, extra_passwords=extra_passwords or [])
    if result.success:
        return f"復号成功(password={result.password})。get_documentツールで内容を取得してください。"
    return f"復号失敗({result.candidates_tried}件試行)。extra_passwordsに別の候補を指定して再試行できます。"


def submit_answer_impl(ctx: ToolContext, answer: str) -> str:
    ctx.capture["answer"] = answer
    return "回答を受け付けました。"


def view_image_impl(path: str) -> tuple[bytes, str]:
    """Raw bytes + best-guess MIME type for an image/rendered-page cache
    path. Used by the Gemini orchestrator's view_image tool to attach
    vision content -- Claude uses its built-in Read tool instead."""
    mime, _ = mimetypes.guess_type(path)
    with open(path, "rb") as f:
        data = f.read()
    return data, mime or "image/png"
