"""In-process MCP tools exposed to the Claude Agent SDK. Thin wrappers
around agent/tool_core.py -- see that module for what each tool actually
does; this module only adapts return values to the SDK's
{"content": [{"type": "text", ...}]} shape.

Only the built-in Read tool is additionally allowed (see orchestrator.py):
get_document returns paths to extracted/rendered images, and the agent
opens those with Read (which handles images/PDFs natively) for anything
needing actual vision -- no separate image-passing plumbing needed here,
unlike the Gemini path (gemini_agent.py) which has no built-in file reader.
"""

from __future__ import annotations

from claude_agent_sdk import SdkMcpTool, tool

from datastel_rag.agent import tool_core as core
from datastel_rag.agent.tool_core import ToolContext
from datastel_rag.catalog.glossary import Glossary
from datastel_rag.catalog.scanner import Catalog
from datastel_rag.index.store import SearchIndex


def _text(s: str, is_error: bool = False) -> dict:
    result = {"content": [{"type": "text", "text": s}]}
    if is_error:
        result["is_error"] = True
    return result


def build_tools(index: SearchIndex, catalog: Catalog, glossary: Glossary, capture: dict) -> list[SdkMcpTool]:
    ctx = ToolContext(index=index, catalog=catalog, glossary=glossary, capture=capture)

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
        return _text(core.search_documents_impl(ctx, args["query"], args.get("project"), int(args.get("top_k") or 8)))

    @tool(
        "get_document",
        "search_documentsで見つけたファイルの全文(書式情報つき)を取得する。画像/レンダリング済みページのパスも返すので、視覚的な読解が必要な場合はReadツールでそのパスを開くこと。",
        {"rel_path": "search_documentsの結果に出てくるrel_path(共有ドライブルートからの相対パス)"},
    )
    async def get_document(args):
        text, _image_paths = core.get_document_impl(ctx, args["rel_path"])
        return _text(text)

    @tool(
        "diff_documents",
        "2つのファイル(旧版と新版など)のテキストをdifflibで機械的に比較し、行単位の差分を返す。"
        "「新旧版で実質的な変更点を挙げよ」系の質問では、目視比較だけに頼らず必ずこれを使うこと"
        "(2つの長い文書を読み比べるだけでは、丸ごと追加されたセクションを見落とすことがある)。",
        {"rel_path_a": "比較元(例: 旧版)のrel_path", "rel_path_b": "比較先(例: 新版)のrel_path"},
    )
    async def diff_documents(args):
        return _text(core.diff_documents_impl(ctx, args["rel_path_a"], args["rel_path_b"]))

    @tool(
        "resolve_project",
        "質問文に出てくる案件名・略称・別名から、社内用語集で定義されている正式な案件名と主略称を解決する。",
        {"text": "質問文またはその一部(案件名・略称を含む文字列)"},
    )
    async def resolve_project(args):
        return _text(core.resolve_project_impl(ctx, args["text"]))

    @tool("list_projects", "全案件の正式名称・主略称・別名の一覧を返す。", {})
    async def list_projects(_args):
        return _text(core.list_projects_impl(ctx))

    @tool(
        "expand_glossary_terms",
        "テキスト中の社内用語・略称(文書種別/会議用語/契約用語/データ分析用語/評価指標/書式用語など)を正式名称に展開する。",
        {"text": "略称・社内用語を含む可能性のあるテキスト"},
    )
    async def expand_glossary_terms(args):
        return _text(core.expand_glossary_terms_impl(ctx, args["text"]))

    @tool(
        "list_project_files",
        "指定した案件のフォルダ構成・ファイル一覧(拡張子・フェーズ・暗号化有無つき)を返す。",
        {"project": "案件の主略称または正式名称"},
    )
    async def list_project_files(args):
        return _text(core.list_project_files_impl(ctx, args["project"]))

    @tool(
        "run_python",
        "案件の生データ(csv/tsv/xlsx)に対してpandas/numpyで集計・計算を行う。"
        "read_table(rel_path, sheet=None)でDataFrameを読み込める。print()の出力、または`result`変数の値が結果として返る。"
        "画像の切り出し等でファイルを保存したい場合はSCRATCH_DIR配下のパスに書き込むこと(それ以外への書き込みは拒否される)。",
        {"code": "実行するPythonコード"},
    )
    async def run_python(args):
        return _text(core.run_python_impl(ctx, args["code"]))

    @tool(
        "attempt_decrypt",
        "get_documentが暗号化により失敗した場合に使う。追加のパスワード候補を指定して復号を再試行する。",
        {"rel_path": "対象ファイルのrel_path", "extra_passwords": "試すパスワード候補のリスト(任意)"},
    )
    async def attempt_decrypt(args):
        return _text(core.attempt_decrypt_impl(ctx, args["rel_path"], args.get("extra_passwords")))

    @tool(
        "submit_answer",
        "最終回答を提出する。質問への回答生成が完了したら必ずこれを1回呼び出すこと。これがそのまま採点対象の回答文字列になる。",
        {"answer": "日本語での最終回答。指定された形式・単位・丸め方・順序規則に従うこと。"},
    )
    async def submit_answer(args):
        return _text(core.submit_answer_impl(ctx, args["answer"]))

    return [
        search_documents,
        get_document,
        diff_documents,
        resolve_project,
        list_projects,
        expand_glossary_terms,
        list_project_files,
        run_python,
        attempt_decrypt,
        submit_answer,
    ]
