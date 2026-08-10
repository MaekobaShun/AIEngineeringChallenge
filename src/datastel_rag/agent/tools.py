"""In-process MCP tools exposed to the agent.

Design: the agent gets exactly the surface it needs to answer a question
from the raw share drive --

- search_documents / get_document: BM25 retrieval + full-document read,
  including paths to extracted/rendered images. The agent opens those
  paths with the built-in Read tool (which handles images/PDFs natively)
  for anything that needs actual vision -- charts, embedded pictures,
  scanned/watermarked PDF pages, the seating chart. No separate
  image-passing plumbing needed.
- resolve_project / expand_terms / list_projects: 社内用語集-backed lookups,
  so the agent expands abbreviations instead of guessing.
- list_project_files: browse a project's raw file tree.
- run_python: pandas/numpy computation over a project's raw tables, for
  aggregation questions -- this is how "programmatic computation, not
  hardcoded" is satisfied for numeric/filter questions.
- attempt_decrypt: retry hook for the rare file neither password scheme
  in ingest/decrypt.py cracks automatically.
- submit_answer: the only way the agent's final answer is captured; this
  avoids scraping free-text assistant messages for the "real" answer.
"""

from __future__ import annotations

import contextlib
import io

import numpy as np
import pandas as pd
from claude_agent_sdk import SdkMcpTool, tool

from datastel_rag.catalog.glossary import Glossary
from datastel_rag.catalog.scanner import Catalog
from datastel_rag.ingest import decrypt
from datastel_rag.ingest.dispatch import parse_entry
from datastel_rag.index.store import SearchIndex

_MAX_DOC_CHARS = 60_000
_MAX_SNIPPET_CHARS = 350


def _resolve_project_key(glossary: Glossary, text: str | None) -> str | None:
    if not text:
        return None
    alias = next((p for p in glossary.projects if p.code == text or p.full_name == text), None)
    if alias:
        return alias.full_name
    resolved = glossary.find_project(text)
    return resolved.full_name if resolved else text


def build_tools(index: SearchIndex, catalog: Catalog, glossary: Glossary, capture: dict) -> list[SdkMcpTool]:
    @tool(
        "search_documents",
        "共有ドライブ全体または特定案件内をBM25全文検索する。案件名・略称・キーワードで絞り込み可能。まず案件を特定してから検索するのが基本。",
        {
            "query": str,
            "project": "案件の主略称(例: KAEDE)または正式名称。省略時は全案件横断検索",
            "top_k": "返す件数(既定8)",
        },
    )
    async def search_documents(args):
        query = args["query"]
        project_key = _resolve_project_key(glossary, args.get("project"))
        top_k = int(args.get("top_k") or 8)
        hits = index.search(query, project_key=project_key, top_k=top_k)
        if not hits:
            return {"content": [{"type": "text", "text": "該当する結果が見つかりませんでした。project指定を外す、またはクエリを変えて再検索してください。"}]}
        lines = []
        for h in hits:
            snippet = h.text[:_MAX_SNIPPET_CHARS]
            lines.append(f"[score={h.score:.1f}] {h.rel_path}\n  block={h.block_id} kind={h.kind} location={h.location}\n  {snippet}")
        return {"content": [{"type": "text", "text": "\n\n".join(lines)}]}

    @tool(
        "get_document",
        "search_documentsで見つけたファイルの全文(書式情報つき)を取得する。画像/レンダリング済みページのパスも返すので、視覚的な読解が必要な場合はReadツールでそのパスを開くこと。",
        {"rel_path": "search_documentsの結果に出てくるrel_path(共有ドライブルートからの相対パス)"},
    )
    async def get_document(args):
        rel_path = args["rel_path"]
        doc = index.get_document(rel_path)
        if doc is None:
            entry = catalog.find_by_rel_path(rel_path)
            if entry is None:
                return {"content": [{"type": "text", "text": f"ファイルが見つかりません: {rel_path}"}], "is_error": True}
            doc = parse_entry(entry, catalog, glossary)

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
        if doc.images:
            lines.append("\n--- 画像/レンダリング済みページ(視覚的な読解が必要な場合はReadツールでこのパスを開く) ---")
            for img in doc.images:
                lines.append(f"  {img.location} -> {img.cache_path}")

        text = "\n".join(lines)
        if len(text) > _MAX_DOC_CHARS:
            text = text[:_MAX_DOC_CHARS] + f"\n... [切り詰め: 全{len(lines)}ブロック中一部のみ表示。必要ならsearch_documentsで該当箇所に絞り込むこと]"
        return {"content": [{"type": "text", "text": text}]}

    @tool(
        "resolve_project",
        "質問文に出てくる案件名・略称・別名から、社内用語集で定義されている正式な案件名と主略称を解決する。",
        {"text": "質問文またはその一部(案件名・略称を含む文字列)"},
    )
    async def resolve_project(args):
        p = glossary.find_project(args["text"])
        if p is None:
            return {"content": [{"type": "text", "text": "一致する案件が見つかりませんでした。list_projectsで一覧を確認してください。"}]}
        return {"content": [{"type": "text", "text": f"正式名称: {p.full_name}\n主略称: {p.code}\n別名候補: {p.aliases}\n補足: {p.note}"}]}

    @tool("list_projects", "全案件の正式名称・主略称・別名の一覧を返す。", {})
    async def list_projects(_args):
        lines = [f"{p.code}: {p.full_name} (別名: {', '.join(p.aliases)})" for p in glossary.projects]
        return {"content": [{"type": "text", "text": "\n".join(lines)}]}

    @tool(
        "expand_glossary_terms",
        "テキスト中の社内用語・略称(文書種別/会議用語/契約用語/データ分析用語/評価指標/書式用語など)を正式名称に展開する。",
        {"text": "略称・社内用語を含む可能性のあるテキスト"},
    )
    async def expand_glossary_terms(args):
        hits = glossary.expand_term(args["text"])
        if not hits:
            return {"content": [{"type": "text", "text": "該当する社内用語は見つかりませんでした。"}]}
        lines = [f"{h.code} = {h.canonical}" + (f" ({h.note})" if h.note else "") for h in hits]
        return {"content": [{"type": "text", "text": "\n".join(lines)}]}

    @tool(
        "list_project_files",
        "指定した案件のフォルダ構成・ファイル一覧(拡張子・フェーズ・暗号化有無つき)を返す。",
        {"project": "案件の主略称または正式名称"},
    )
    async def list_project_files(args):
        project_key = _resolve_project_key(glossary, args["project"])
        proj = catalog.find_project(project_key) if project_key else None
        if proj is None:
            return {"content": [{"type": "text", "text": f"案件が見つかりません: {args['project']}"}], "is_error": True}
        lines = [f"{f.rel_path}  (ext={f.ext}, phase={f.phase}, encrypted={f.is_encrypted})" for f in proj.files]
        return {"content": [{"type": "text", "text": "\n".join(lines)}]}

    @tool(
        "run_python",
        "案件の生データ(csv/tsv/xlsx)に対してpandas/numpyで集計・計算を行う。"
        "read_table(rel_path, sheet=None)でDataFrameを読み込める。print()の出力、または`result`変数の値が結果として返る。",
        {"code": "実行するPythonコード"},
    )
    async def run_python(args):
        code = args["code"]

        def read_table(rel_path: str, sheet=None):
            entry = catalog.find_by_rel_path(rel_path)
            if entry is None:
                raise FileNotFoundError(rel_path)
            if entry.ext == "csv":
                return pd.read_csv(entry.abs_path, encoding="utf-8-sig")
            if entry.ext == "tsv":
                return pd.read_csv(entry.abs_path, sep="\t", encoding="utf-8-sig")
            if entry.ext == "xlsx":
                return pd.read_excel(entry.abs_path, sheet_name=sheet if sheet is not None else 0)
            raise ValueError(f"read_tableが対応していない拡張子です: {entry.ext}")

        global_ns = {"pd": pd, "np": np, "read_table": read_table, "__builtins__": __builtins__}
        local_ns: dict = {}
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                exec(code, global_ns, local_ns)
        except Exception as e:
            return {
                "content": [{"type": "text", "text": f"実行エラー: {type(e).__name__}: {e}\n\n--- stdout ---\n{buf.getvalue()}"}],
                "is_error": True,
            }

        out = buf.getvalue()
        if "result" in local_ns:
            out += f"\nresult = {local_ns['result']!r}"
        if not out.strip():
            out = "(出力なし。print()するか`result`変数に代入してください)"
        return {"content": [{"type": "text", "text": out}]}

    @tool(
        "attempt_decrypt",
        "get_documentが暗号化により失敗した場合に使う。追加のパスワード候補を指定して復号を再試行する。",
        {"rel_path": "対象ファイルのrel_path", "extra_passwords": "試すパスワード候補のリスト(任意)"},
    )
    async def attempt_decrypt(args):
        entry = catalog.find_by_rel_path(args["rel_path"])
        if entry is None:
            return {"content": [{"type": "text", "text": f"見つかりません: {args['rel_path']}"}], "is_error": True}
        result = decrypt.decrypt_entry(entry, catalog, glossary, extra_passwords=args.get("extra_passwords") or [])
        if result.success:
            return {"content": [{"type": "text", "text": f"復号成功(password={result.password})。get_documentツールで内容を取得してください。"}]}
        return {
            "content": [{"type": "text", "text": f"復号失敗({result.candidates_tried}件試行)。extra_passwordsに別の候補を指定して再試行できます。"}],
            "is_error": True,
        }

    @tool(
        "submit_answer",
        "最終回答を提出する。質問への回答生成が完了したら必ずこれを1回呼び出すこと。これがそのまま採点対象の回答文字列になる。",
        {"answer": "日本語での最終回答。指定された形式・単位・丸め方・順序規則に従うこと。"},
    )
    async def submit_answer(args):
        capture["answer"] = args["answer"]
        return {"content": [{"type": "text", "text": "回答を受け付けました。"}]}

    return [
        search_documents,
        get_document,
        resolve_project,
        list_projects,
        expand_glossary_terms,
        list_project_files,
        run_python,
        attempt_decrypt,
        submit_answer,
    ]
