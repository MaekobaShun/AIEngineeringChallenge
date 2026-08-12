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
from datastel_rag.agent.prompts import build_system_prompt, user_prompt
from datastel_rag.agent.tool_core import ToolContext
from datastel_rag.catalog.glossary import Glossary
from datastel_rag.catalog.scanner import Catalog
from datastel_rag.index.store import SearchIndex
from datastel_rag.skill_tree.tree import SkillTree, load_skill_tree


def _schema(properties: dict, required: list[str] | None = None) -> dict:
    return {"type": "object", "properties": properties, "required": required or []}


_STR = {"type": "string"}
_INT = {"type": "integer"}

_FD_SEARCH = types.FunctionDeclaration(
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
)
_FD_LIST_CHILDREN = types.FunctionDeclaration(
    name="list_children",
    description=(
        "スキルツリーの子ノード一覧を返す。最初は node_id='root'。"
        "案件→フェーズ→ファイルと降り、リーフの rel_path を得たら get_document へ。"
        "行き止まりなら parent_id に戻る。"
    ),
    parameters_json_schema=_schema(
        {"node_id": {**_STR, "description": "ノードID。省略時は root"}},
    ),
)
_FD_GET_DOCUMENT = types.FunctionDeclaration(
    name="get_document",
    description="rel_path で指定したファイルの全文(書式情報つき)を取得する。画像/レンダリング済みページがある場合はview_imageツールで開いて確認すること。",
    parameters_json_schema=_schema({"rel_path": _STR}, ["rel_path"]),
)
_FD_VIEW_IMAGE = types.FunctionDeclaration(
    name="view_image",
    description="画像ファイル、またはget_documentが返すレンダリング済みページ(PDFページ等)のキャッシュパスを開いて視覚的に確認する。",
    parameters_json_schema=_schema({"path": _STR}, ["path"]),
)
_FD_RESOLVE_PROJECT = types.FunctionDeclaration(
    name="resolve_project",
    description="質問文に出てくる案件名・略称・別名から、社内用語集で定義されている正式な案件名と主略称を解決する。",
    parameters_json_schema=_schema({"text": _STR}, ["text"]),
)
_FD_LIST_PROJECTS = types.FunctionDeclaration(
    name="list_projects",
    description="全案件の正式名称・主略称・別名の一覧を返す。",
    parameters_json_schema=_schema({}),
)
_FD_EXPAND = types.FunctionDeclaration(
    name="expand_glossary_terms",
    description="テキスト中の社内用語・略称(文書種別/会議用語/契約用語/データ分析用語/評価指標/書式用語など)を正式名称に展開する。",
    parameters_json_schema=_schema({"text": _STR}, ["text"]),
)
_FD_LIST_FILES = types.FunctionDeclaration(
    name="list_project_files",
    description="指定した案件のフォルダ構成・ファイル一覧(拡張子・フェーズ・暗号化有無つき)を返す。",
    parameters_json_schema=_schema({"project": _STR}, ["project"]),
)
_FD_RUN_PYTHON = types.FunctionDeclaration(
    name="run_python",
    description=(
        "案件の生データ(csv/tsv/xlsx)に対してpandas/numpyで集計・計算を行う。"
        "read_table(rel_path, sheet=None)でDataFrameを読み込める。print()の出力、または`result`変数の値が結果として返る。"
        "画像の切り出し等でファイルを保存したい場合はSCRATCH_DIR配下のパスに書き込むこと(それ以外への書き込みは拒否される)。"
    ),
    parameters_json_schema=_schema({"code": _STR}, ["code"]),
)
_FD_DECRYPT = types.FunctionDeclaration(
    name="attempt_decrypt",
    description="get_documentが暗号化により失敗した場合に使う。追加のパスワード候補を指定して復号を再試行する。",
    parameters_json_schema=_schema(
        {"rel_path": _STR, "extra_passwords": {"type": "array", "items": _STR}},
        ["rel_path"],
    ),
)
_FD_SUBMIT = types.FunctionDeclaration(
    name="submit_answer",
    description="最終回答を提出する。質問への回答生成が完了したら必ずこれを1回呼び出すこと。これがそのまま採点対象の回答文字列になる。",
    parameters_json_schema=_schema({"answer": _STR}, ["answer"]),
)


def _tools_for_mode(retrieval_mode: str) -> list[types.Tool]:
    """skill_nav is an isolated toolset switch for ablation testing: BM25 and
    flat file listing are omitted so the agent cannot choose search vs
    navigate, to measure navigation alone. hybrid (the production default)
    gives the agent both search_documents and list_children together, so it
    can fall back to navigation on its own when keyword search comes up
    short -- that's autonomous tool selection, unlike skill_nav's forced
    route."""
    common_tail = [
        _FD_GET_DOCUMENT,
        _FD_VIEW_IMAGE,
        _FD_RESOLVE_PROJECT,
        _FD_LIST_PROJECTS,
        _FD_EXPAND,
        _FD_RUN_PYTHON,
        _FD_DECRYPT,
        _FD_SUBMIT,
    ]
    if retrieval_mode == "skill_nav":
        decls = [_FD_LIST_CHILDREN, *common_tail]
    elif retrieval_mode == "hybrid":
        decls = [_FD_SEARCH, _FD_LIST_FILES, _FD_LIST_CHILDREN, *common_tail]
    else:
        decls = [_FD_SEARCH, _FD_LIST_FILES, *common_tail]
    return [types.Tool(function_declarations=decls)]


def _dispatch(ctx: ToolContext, name: str, args: dict) -> tuple[dict, bytes | None, str | None]:
    """Returns (function_response_payload, optional image bytes, optional mime_type)."""
    try:
        if name == "search_documents":
            if ctx.retrieval_mode == "skill_nav":
                return {"error": "skill_nav mode: search_documents is disabled (forced navigate). Use list_children."}, None, None
            return {"result": core.search_documents_impl(ctx, args["query"], args.get("project"), int(args.get("top_k") or 8))}, None, None
        if name == "list_children":
            return {"result": core.list_children_impl(ctx, args.get("node_id"))}, None, None
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
            if ctx.retrieval_mode == "skill_nav":
                return {"error": "skill_nav mode: list_project_files is disabled. Use list_children."}, None, None
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
    input_tokens: int = 0
    output_tokens: int = 0
    session_id: str | None = None
    is_error: bool = False
    error_detail: str | None = None
    transcript: list = field(default_factory=list, repr=False)


# USD per 1M tokens, (input, output). Keyed by substring match against the
# resolved model_version the API reports (not the "-latest" alias we call
# with, which doesn't tell us which concrete model actually served the
# request). Real $ tracking matters here because -- unlike the Claude Agent
# SDK, which reports total_cost_usd per call -- the Gemini API gives us
# nothing but raw token counts, so without this every run silently cost
# $0.00 on paper while real money was being spent.
_PRICING_PER_MTOK = {
    "gemini-3.6-flash": (1.50, 7.50),
    "gemini-3.5-flash-lite": (0.30, 2.50),
    "gemini-3.1-flash-lite": (0.30, 2.50),
    "gemini-3-flash": (1.50, 7.50),
}
_DEFAULT_PRICING = (1.50, 7.50)  # conservative: assume full Flash pricing if the model is unrecognized


def _price_for_model(model_version: str) -> tuple[float, float]:
    for key, price in _PRICING_PER_MTOK.items():
        if key in model_version:
            return price
    return _DEFAULT_PRICING


_MAX_RETRIES = 6
_DEFAULT_BACKOFF_S = 20.0


def _quota_details(e: errors.ClientError) -> tuple[float | None, bool]:
    """Returns (suggested_retry_delay_seconds, is_per_day_quota).

    Free-tier quota comes in at least two shapes we've hit in practice: a
    per-minute RPM cap (worth waiting out -- routinely triggered mid-question
    by a multi-turn tool-calling loop) and a per-day request cap (500/day
    seen on gemini-3.5-flash-lite). Retrying a per-day exhaustion is
    pointless -- the suggested delay is short (seconds) but the quota won't
    actually refill for hours, so it's better to fail this question fast
    than burn max_turns retrying a wait that won't help.
    """
    try:
        details = e.response_json.get("error", {}).get("details", [])
        delay = None
        is_per_day = False
        for d in details:
            if d.get("@type", "").endswith("RetryInfo"):
                m = re.match(r"([\d.]+)s", d.get("retryDelay", ""))
                if m:
                    delay = float(m.group(1))
            if d.get("@type", "").endswith("QuotaFailure"):
                for v in d.get("violations", []):
                    if "PerDay" in v.get("quotaId", ""):
                        is_per_day = True
        return delay, is_per_day
    except Exception:
        return None, False


class DailyQuotaExhausted(Exception):
    pass


async def _generate_with_retry(client, model_name, contents, gen_config):
    last_err = None
    for attempt in range(_MAX_RETRIES):
        try:
            return await client.aio.models.generate_content(model=model_name, contents=contents, config=gen_config)
        except errors.ClientError as e:
            last_err = e
            if e.status != "RESOURCE_EXHAUSTED" and getattr(e, "code", None) != 429:
                raise
            delay, is_per_day = _quota_details(e)
            if is_per_day:
                raise DailyQuotaExhausted(str(e)) from e
            await asyncio.sleep((delay or _DEFAULT_BACKOFF_S * (attempt + 1)) + 1)
    raise last_err


def _final_text(content) -> str | None:
    if content is None or not content.parts:
        return None
    texts = [p.text for p in content.parts if getattr(p, "text", None)]
    return "\n".join(texts) if texts else None


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def _make_client() -> genai.Client:
    """Gemini Developer API (API key) or Gemini Enterprise / Vertex AI (ADC).

    Enterprise mode bills to the GCP project, so the $300 Welcome credit
    applies. The Developer API key path cannot spend that credit.
    """
    if _env_flag("GOOGLE_GENAI_USE_ENTERPRISE") or _env_flag("GOOGLE_GENAI_USE_VERTEXAI"):
        project = os.environ.get("GOOGLE_CLOUD_PROJECT")
        if not project:
            raise RuntimeError(
                "GOOGLE_GENAI_USE_ENTERPRISE is set but GOOGLE_CLOUD_PROJECT is empty"
            )
        return genai.Client(
            enterprise=True,
            project=project,
            location=os.environ.get("GOOGLE_CLOUD_LOCATION") or "global",
        )
    return genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))


async def answer_question_async(
    question: str,
    index: SearchIndex,
    catalog: Catalog,
    glossary: Glossary,
    max_turns: int = 50,
    max_budget_usd: float = 1.5,  # unused for Gemini (no per-call cost API); kept for interface parity
    model: str | None = None,
    retrieval_mode: str = "hybrid",
    skill_tree: SkillTree | None = None,
) -> AnswerResult:
    capture: dict = {}
    tree = skill_tree
    if retrieval_mode in ("skill_nav", "hybrid") and tree is None:
        tree = load_skill_tree()
    ctx = ToolContext(
        index=index,
        catalog=catalog,
        glossary=glossary,
        capture=capture,
        skill_tree=tree,
        retrieval_mode=retrieval_mode,
    )
    client = _make_client()
    model_name = model or config.GEMINI_MODEL

    gen_config = types.GenerateContentConfig(
        system_instruction=build_system_prompt(image_tool="view_image", retrieval_mode=retrieval_mode),
        tools=_tools_for_mode(retrieval_mode),
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )

    contents: list = [types.Content(role="user", parts=[types.Part.from_text(text=user_prompt(question))])]
    result = AnswerResult(question=question, answer="")

    try:
        for turn in range(max_turns):
            response = await _generate_with_retry(client, model_name, contents, gen_config)
            if response.usage_metadata:
                um = response.usage_metadata
                result.total_tokens += um.total_token_count or 0
                in_tok = um.prompt_token_count or 0
                out_tok = um.candidates_token_count or 0
                result.input_tokens += in_tok
                result.output_tokens += out_tok
                in_price, out_price = _price_for_model(response.model_version or model_name)
                result.cost_usd += (in_tok / 1_000_000) * in_price + (out_tok / 1_000_000) * out_price

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
                # Two things Part.from_function_response() doesn't do, both needed for
                # Vertex specifically (the Developer/AI Studio API tolerated the naive
                # version fine):
                #  1. It drops the call's `id`, correlating replies to calls by `name`
                #     alone -- ambiguous once a turn has >1 call (even to the same tool).
                #  2. There's no way to attach media through it. view_image's image used
                #     to ride along as a separate sibling Part(inline_data=...) after the
                #     function_response Part; Vertex rejects that shape. Inline data
                #     belongs *inside* the FunctionResponse's own `parts`
                #     (FunctionResponsePart(inline_data=FunctionResponseBlob(...))), not
                #     as a top-level Part next to it.
                # This combination is the confirmed cause of the intermittent Vertex 400
                # "Requests ending with a model turn are not supported" errors: the
                # malformed turn gets rejected/dropped, leaving the conversation ending on
                # the prior model turn from the server's point of view.
                fr_parts = None
                if image_bytes:
                    fr_parts = [
                        types.FunctionResponsePart(
                            inline_data=types.FunctionResponseBlob(mime_type=mime, data=image_bytes)
                        )
                    ]
                response_parts.append(
                    types.Part(
                        function_response=types.FunctionResponse(id=fc.id, name=fc.name, response=fn_result, parts=fr_parts)
                    )
                )
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
