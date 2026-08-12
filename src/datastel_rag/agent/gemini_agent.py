"""Gemini function-calling orchestrator -- the Gemini counterpart to
agent/orchestrator.py (Claude Agent SDK). There is no equivalent "agent
SDK" on the Gemini side, so this hand-rolls the tool-call loop: send
contents -> read function_call parts off the response -> dispatch to
agent/tool_core.py -> feed function_response parts back -> repeat until
submit_answer is called or max_turns is hit.

Vision handling differs from the Claude path on purpose: Claude Code ships
a built-in Read tool that opens any path (images/PDFs) directly, so
get_document just mentions image paths in its text and the agent calls
Read itself. Gemini has no such built-in, so a dedicated view_image tool
is declared here; when the agent calls it, the image bytes are read and
attached as an inline_data Part alongside the function_response in the
same turn.
"""

from __future__ import annotations

import asyncio
import os
import re
from dataclasses import dataclass, field

from google import genai
from google.genai import errors, types

from datastel_rag import config
from datastel_rag.agent import tool_core as core
from datastel_rag.agent.prompts import SYSTEM_PROMPT_GEMINI, user_prompt
from datastel_rag.agent.tool_core import ToolContext
from datastel_rag.catalog.glossary import Glossary
from datastel_rag.catalog.scanner import Catalog
from datastel_rag.index.store import SearchIndex


def _schema(properties: dict, required: list[str] | None = None) -> dict:
    return {"type": "object", "properties": properties, "required": required or []}


_STR = {"type": "string"}
_INT = {"type": "integer"}

_FUNCTION_DECLARATIONS = [
    types.FunctionDeclaration(
        name="search_documents",
        description="共有ドライブ全体または特定案件内をBM25全文検索する。案件名・略称・キーワードで絞り込み可能。まず案件を特定してから検索するのが基本。",
        parameters_json_schema=_schema(
            {
                "query": _STR,
                "project": {**_STR, "description": "案件の主略称(例: KAEDE)または正式名称。省略時は全案件横断検索"},
                "top_k": {**_INT, "description": "返す件数(既定8)"},
            },
            ["query"],
        ),
    ),
    types.FunctionDeclaration(
        name="get_document",
        description="search_documentsで見つけたファイルの全文(書式情報つき)を取得する。画像/レンダリング済みページがある場合はview_imageツールで開いて確認すること。",
        parameters_json_schema=_schema({"rel_path": _STR}, ["rel_path"]),
    ),
    types.FunctionDeclaration(
        name="view_image",
        description="画像ファイル、またはget_documentが返すレンダリング済みページ(PDFページ等)のキャッシュパスを開いて視覚的に確認する。",
        parameters_json_schema=_schema({"path": _STR}, ["path"]),
    ),
    types.FunctionDeclaration(
        name="resolve_project",
        description="質問文に出てくる案件名・略称・別名から、社内用語集で定義されている正式な案件名と主略称を解決する。",
        parameters_json_schema=_schema({"text": _STR}, ["text"]),
    ),
    types.FunctionDeclaration(
        name="list_projects",
        description="全案件の正式名称・主略称・別名の一覧を返す。",
        parameters_json_schema=_schema({}),
    ),
    types.FunctionDeclaration(
        name="expand_glossary_terms",
        description="テキスト中の社内用語・略称(文書種別/会議用語/契約用語/データ分析用語/評価指標/書式用語など)を正式名称に展開する。",
        parameters_json_schema=_schema({"text": _STR}, ["text"]),
    ),
    types.FunctionDeclaration(
        name="list_project_files",
        description="指定した案件のフォルダ構成・ファイル一覧(拡張子・フェーズ・暗号化有無つき)を返す。",
        parameters_json_schema=_schema({"project": _STR}, ["project"]),
    ),
    types.FunctionDeclaration(
        name="run_python",
        description=(
            "案件の生データ(csv/tsv/xlsx)に対してpandas/numpyで集計・計算を行う。"
            "read_table(rel_path, sheet=None)でDataFrameを読み込める。print()の出力、または`result`変数の値が結果として返る。"
            "画像の切り出し等でファイルを保存したい場合はSCRATCH_DIR配下のパスに書き込むこと(それ以外への書き込みは拒否される)。"
        ),
        parameters_json_schema=_schema({"code": _STR}, ["code"]),
    ),
    types.FunctionDeclaration(
        name="attempt_decrypt",
        description="get_documentが暗号化により失敗した場合に使う。追加のパスワード候補を指定して復号を再試行する。",
        parameters_json_schema=_schema(
            {"rel_path": _STR, "extra_passwords": {"type": "array", "items": _STR}},
            ["rel_path"],
        ),
    ),
    types.FunctionDeclaration(
        name="submit_answer",
        description="最終回答を提出する。質問への回答生成が完了したら必ずこれを1回呼び出すこと。これがそのまま採点対象の回答文字列になる。",
        parameters_json_schema=_schema({"answer": _STR}, ["answer"]),
    ),
]

_TOOLS = [types.Tool(function_declarations=_FUNCTION_DECLARATIONS)]


def _dispatch(ctx: ToolContext, name: str, args: dict) -> tuple[dict, bytes | None, str | None]:
    """Returns (function_response_payload, optional image bytes, optional mime_type)."""
    try:
        if name == "search_documents":
            return {"result": core.search_documents_impl(ctx, args["query"], args.get("project"), int(args.get("top_k") or 8))}, None, None
        if name == "get_document":
            text, _image_paths = core.get_document_impl(ctx, args["rel_path"])
            return {"result": text}, None, None
        if name == "view_image":
            data, mime = core.view_image_impl(args["path"])
            return {"result": "画像を添付しました。"}, data, mime
        if name == "resolve_project":
            return {"result": core.resolve_project_impl(ctx, args["text"])}, None, None
        if name == "list_projects":
            return {"result": core.list_projects_impl(ctx)}, None, None
        if name == "expand_glossary_terms":
            return {"result": core.expand_glossary_terms_impl(ctx, args["text"])}, None, None
        if name == "list_project_files":
            return {"result": core.list_project_files_impl(ctx, args["project"])}, None, None
        if name == "run_python":
            return {"result": core.run_python_impl(ctx, args["code"])}, None, None
        if name == "attempt_decrypt":
            return {"result": core.attempt_decrypt_impl(ctx, args["rel_path"], args.get("extra_passwords"))}, None, None
        if name == "submit_answer":
            return {"result": core.submit_answer_impl(ctx, args["answer"])}, None, None
        return {"error": f"unknown tool: {name}"}, None, None
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}, None, None


@dataclass
class AnswerResult:
    question: str
    answer: str
    num_turns: int = 0
    cost_usd: float = 0.0
    total_tokens: int = 0
    session_id: str | None = None
    is_error: bool = False
    error_detail: str | None = None
    transcript: list = field(default_factory=list, repr=False)


_MAX_RETRIES = 6
_DEFAULT_BACKOFF_S = 20.0


def _retry_delay_seconds(e: errors.ClientError) -> float | None:
    """Google's free-tier RPM limits are tight enough (e.g. 5 req/min) that
    a single question's multi-turn loop routinely hits 429s -- this isn't
    an edge case to fail fast on. The error response includes a RetryInfo
    with the server's own suggested wait; prefer that over guessing."""
    try:
        details = e.response_json.get("error", {}).get("details", [])
        for d in details:
            if d.get("@type", "").endswith("RetryInfo"):
                m = re.match(r"([\d.]+)s", d.get("retryDelay", ""))
                if m:
                    return float(m.group(1))
    except Exception:
        pass
    return None


async def _generate_with_retry(client, model_name, contents, gen_config):
    last_err = None
    for attempt in range(_MAX_RETRIES):
        try:
            return await client.aio.models.generate_content(model=model_name, contents=contents, config=gen_config)
        except errors.ClientError as e:
            last_err = e
            if e.status != "RESOURCE_EXHAUSTED" and getattr(e, "code", None) != 429:
                raise
            delay = _retry_delay_seconds(e) or _DEFAULT_BACKOFF_S * (attempt + 1)
            await asyncio.sleep(delay + 1)
    raise last_err


def _final_text(content) -> str | None:
    if content is None or not content.parts:
        return None
    texts = [p.text for p in content.parts if getattr(p, "text", None)]
    return "\n".join(texts) if texts else None


async def answer_question_async(
    question: str,
    index: SearchIndex,
    catalog: Catalog,
    glossary: Glossary,
    max_turns: int = 50,
    max_budget_usd: float = 1.5,  # unused for Gemini (no per-call cost API); kept for interface parity
    model: str | None = None,
) -> AnswerResult:
    capture: dict = {}
    ctx = ToolContext(index=index, catalog=catalog, glossary=glossary, capture=capture)
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    model_name = model or config.GEMINI_MODEL

    gen_config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT_GEMINI,
        tools=_TOOLS,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )

    contents: list = [types.Content(role="user", parts=[types.Part.from_text(text=user_prompt(question))])]
    result = AnswerResult(question=question, answer="")

    try:
        for turn in range(max_turns):
            response = await _generate_with_retry(client, model_name, contents, gen_config)
            if response.usage_metadata and response.usage_metadata.total_token_count:
                result.total_tokens += response.usage_metadata.total_token_count

            if not response.candidates:
                result.is_error = True
                result.error_detail = f"no candidates (prompt_feedback={response.prompt_feedback})"
                break

            candidate = response.candidates[0]
            result.num_turns = turn + 1
            if candidate.content is None:
                result.is_error = True
                result.error_detail = f"empty content (finish_reason={candidate.finish_reason})"
                break
            contents.append(candidate.content)

            fn_calls = [p.function_call for p in candidate.content.parts if p.function_call]
            if not fn_calls:
                fallback = _final_text(candidate.content)
                if fallback:
                    capture.setdefault("_fallback_text", fallback)
                break

            response_parts = []
            for fc in fn_calls:
                fn_result, image_bytes, mime = _dispatch(ctx, fc.name, dict(fc.args or {}))
                response_parts.append(types.Part.from_function_response(name=fc.name, response=fn_result))
                if image_bytes:
                    response_parts.append(types.Part.from_bytes(data=image_bytes, mime_type=mime))
            contents.append(types.Content(role="user", parts=response_parts))

            if "answer" in capture:
                break
        else:
            result.is_error = True
            result.error_detail = f"max_turns ({max_turns}) reached without submit_answer"
    except Exception as e:
        result.is_error = True
        result.error_detail = f"{type(e).__name__}: {e}"

    answer = capture.get("answer")
    if answer is None and not result.is_error:
        answer = capture.get("_fallback_text")
    if not answer:
        answer = "わかりません"
    result.answer = answer
    result.transcript = contents
    return result


def answer_question(
    question: str,
    index: SearchIndex,
    catalog: Catalog,
    glossary: Glossary,
    **kwargs,
) -> AnswerResult:
    return asyncio.run(answer_question_async(question, index, catalog, glossary, **kwargs))
