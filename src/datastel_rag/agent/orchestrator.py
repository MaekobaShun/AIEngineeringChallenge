"""Runs one question through the Claude Agent SDK loop and captures the
answer submitted via the submit_answer tool.

Tool availability is deliberately narrow: only the built-in `Read` tool
(for opening images/PDFs/rendered pages the custom tools point at) plus our
own MCP server. Everything else built-in (Bash, Write, Edit, WebFetch,
WebSearch, Task, ...) is unavailable, since the competition rules require
answers to come only from the provided share-drive data, and this pipeline
should never write to or execute arbitrary commands against the user's
machine.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import claude_agent_sdk as sdk

from datastel_rag import config
from datastel_rag.agent.prompts import SYSTEM_PROMPT, user_prompt
from datastel_rag.agent.tools import build_tools
from datastel_rag.catalog.glossary import Glossary
from datastel_rag.catalog.scanner import Catalog
from datastel_rag.index.store import SearchIndex


@dataclass
class AnswerResult:
    question: str
    answer: str
    num_turns: int = 0
    cost_usd: float = 0.0
    session_id: str | None = None
    is_error: bool = False
    error_detail: str | None = None
    transcript: list = field(default_factory=list, repr=False)


def _build_options(model: str, max_turns: int, max_budget_usd: float, tools: list) -> sdk.ClaudeAgentOptions:
    server = sdk.create_sdk_mcp_server("datastel", tools=tools)
    return sdk.ClaudeAgentOptions(
        system_prompt=SYSTEM_PROMPT,
        tools=["Read"],
        mcp_servers={"datastel": server},
        strict_mcp_config=True,
        permission_mode="bypassPermissions",
        cwd=str(config.CHALLENGE_ROOT),
        model=model,
        max_turns=max_turns,
        max_budget_usd=max_budget_usd,
        setting_sources=[],
    )


async def answer_question_async(
    question: str,
    index: SearchIndex,
    catalog: Catalog,
    glossary: Glossary,
    max_turns: int = 40,
    max_budget_usd: float = 1.0,
    model: str | None = None,
) -> AnswerResult:
    capture: dict = {}
    tools = build_tools(index, catalog, glossary, capture)
    options = _build_options(model or config.ANTHROPIC_MODEL, max_turns, max_budget_usd, tools)

    transcript = []
    result = AnswerResult(question=question, answer="")
    try:
        async for msg in sdk.query(prompt=user_prompt(question), options=options):
            transcript.append(msg)
            if isinstance(msg, sdk.ResultMessage):
                result.num_turns = msg.num_turns
                result.cost_usd = msg.total_cost_usd or 0.0
                result.session_id = msg.session_id
                result.is_error = bool(msg.is_error)
                if msg.is_error:
                    result.error_detail = msg.result
    except Exception as e:
        # a single question's transport/API failure (billing, rate limit,
        # transient network error, ...) must not take down a whole batch run
        result.is_error = True
        result.error_detail = f"{type(e).__name__}: {e}"

    answer = capture.get("answer")
    if answer is None and not result.is_error:
        # fallback: last assistant text, in case the model forgot to call submit_answer.
        # Only trusted when the run itself didn't error -- on error the last
        # assistant text is often a synthetic message (e.g. a billing error),
        # not a real answer.
        for msg in reversed(transcript):
            if isinstance(msg, sdk.AssistantMessage):
                texts = [b.text for b in msg.content if isinstance(b, sdk.TextBlock)]
                if texts:
                    answer = "\n".join(texts)
                    break
    if not answer:
        answer = "わかりません"

    result.answer = answer
    result.transcript = transcript
    return result


def answer_question(
    question: str,
    index: SearchIndex,
    catalog: Catalog,
    glossary: Glossary,
    **kwargs,
) -> AnswerResult:
    return asyncio.run(answer_question_async(question, index, catalog, glossary, **kwargs))
