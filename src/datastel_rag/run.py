"""Entrypoint: reads a questions CSV (index,question), answers each one via
the agent, and writes predictions.csv (index,answer -- no header, matching
evaluation/src/validator.py's expected format).

Usage:
    uv run python -m datastel_rag.run --questions ../share/質問回答/questions_valid.csv --out submit/predictions.csv
    uv run python -m datastel_rag.run --questions ... --out ... --limit 5   # smoke test on the first 5 rows
    uv run python -m datastel_rag.run --questions ... --out ... --indices 2,5,10 --retrieval-mode skill_nav
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

import pandas as pd

from datastel_rag import config
from datastel_rag.catalog.glossary import load_or_build_glossary
from datastel_rag.catalog.scanner import load_or_build_catalog
from datastel_rag.index.store import SearchIndex
from datastel_rag.skill_tree.tree import SkillTree, load_skill_tree

_PROVIDERS = {}


def _get_provider(name: str):
    if name not in _PROVIDERS:
        if name == "claude":
            from datastel_rag.agent.orchestrator import answer_question_async
        elif name == "gemini":
            from datastel_rag.agent.gemini_agent import answer_question_async
        else:
            raise ValueError(f"unknown provider: {name}")
        _PROVIDERS[name] = answer_question_async
    return _PROVIDERS[name]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--questions", required=True, help="questions CSV path (columns: index,question)")
    p.add_argument("--out", required=True, help="output predictions.csv path")
    p.add_argument("--provider", choices=["gemini", "claude"], default="gemini")
    p.add_argument("--log", default=None, help="JSONL run log path (default: logs/run_<timestamp>.jsonl)")
    p.add_argument("--limit", type=int, default=None, help="only answer the first N rows (smoke testing)")
    p.add_argument(
        "--indices",
        default=None,
        help="comma-separated question indices to run (e.g. 2,5,10,18,20,29)",
    )
    p.add_argument(
        "--retrieval-mode",
        choices=["hybrid", "bm25", "skill_nav"],
        default="hybrid",
        help=(
            "hybrid (default, production): search_documents + list_children both available, "
            "agent autonomously falls back to folder navigation when search comes up short. "
            "bm25: search_documents toolset only (no navigation fallback) -- ablation baseline. "
            "skill_nav: forced folder-tree navigation only (BM25 disabled) -- ablation baseline, "
            "route_forced rather than autonomous."
        ),
    )
    p.add_argument("--concurrency", type=int, default=3)
    p.add_argument("--max-turns", type=int, default=60)
    p.add_argument("--max-budget-usd", type=float, default=1.5, help="Claude only; Gemini has no per-call cost API")
    p.add_argument("--model", default=None)
    p.add_argument("--refresh-index", action="store_true", help="force re-parse everything instead of using the cache")
    return p.parse_args()


async def _run_one(sem, idx_num, question, index, catalog, glossary, args, log_f, skill_tree):
    answer_question_async = _get_provider(args.provider)
    async with sem:
        start = time.time()
        kwargs = dict(
            max_turns=args.max_turns,
            max_budget_usd=args.max_budget_usd,
            model=args.model,
        )
        if args.provider == "gemini":
            kwargs["retrieval_mode"] = args.retrieval_mode
            kwargs["skill_tree"] = skill_tree
        elif args.retrieval_mode != "bm25":
            raise SystemExit(f"--retrieval-mode {args.retrieval_mode} is currently wired for --provider gemini only")
        result = await answer_question_async(
            question,
            index,
            catalog,
            glossary,
            **kwargs,
        )
        elapsed = time.time() - start
        default_model = config.GEMINI_MODEL if args.provider == "gemini" else config.ANTHROPIC_MODEL
        record = {
            "index": idx_num,
            "question": question,
            "answer": result.answer,
            "provider": args.provider,
            "retrieval_mode": args.retrieval_mode,
            "route_policy": {
                "skill_nav": "forced_navigate_first",
                "hybrid": "autonomous_search_or_navigate",
                "bm25": "bm25_search_only",
            }[args.retrieval_mode],
            "experiment_note": (
                "Toolset switch test: BM25 disabled, navigate-first prompt forced. "
                "NOT autonomous agent routing between BM25 and skill tree."
                if args.retrieval_mode == "skill_nav"
                else None
            ),
            "num_turns": result.num_turns,
            "cost_usd": result.cost_usd,
            "input_tokens": getattr(result, "input_tokens", None),
            "output_tokens": getattr(result, "output_tokens", None),
            "elapsed_s": round(elapsed, 1),
            "is_error": result.is_error,
            "error_detail": result.error_detail,
            "session_id": result.session_id,
            "model": args.model or default_model,
        }
        log_f.write(json.dumps(record, ensure_ascii=False) + "\n")
        log_f.flush()
        print(f"[{idx_num}] turns={result.num_turns} cost=${result.cost_usd:.3f} err={result.is_error} :: {result.answer[:80]!r}")
        return idx_num, result.answer


async def main_async(args):
    questions_df = pd.read_csv(args.questions)
    if args.indices:
        wanted = {int(x.strip()) for x in args.indices.split(",") if x.strip()}
        questions_df = questions_df[questions_df["index"].isin(wanted)]
    if args.limit:
        questions_df = questions_df.head(args.limit)

    catalog = load_or_build_catalog(refresh=args.refresh_index)
    glossary = load_or_build_glossary(refresh=args.refresh_index)
    index = SearchIndex()
    index.build(catalog, glossary, force=args.refresh_index)

    skill_tree: SkillTree | None = None
    if args.retrieval_mode in ("skill_nav", "hybrid"):
        skill_tree = load_skill_tree()
        print(f"{args.retrieval_mode} tree loaded: nodes={len(skill_tree.nodes)} meta={skill_tree.meta.get('version')}")

    log_path = Path(args.log) if args.log else config.LOG_DIR / f"run_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    sem = asyncio.Semaphore(args.concurrency)
    with open(log_path, "w", encoding="utf-8") as log_f:
        tasks = [
            _run_one(sem, int(row["index"]), row["question"], index, catalog, glossary, args, log_f, skill_tree)
            for _, row in questions_df.iterrows()
        ]
        results = await asyncio.gather(*tasks)

    # evaluation/src/validator.py pre-checks the file line-by-line before any
    # real CSV parsing, so an embedded newline (breaking one row across two
    # physical lines) is fatal even though it'd be valid CSV -- collapse to
    # single-line answers defensively (the prompt already asks for this).
    sanitized = [(i, " / ".join(a.replace("\r\n", "\n").split("\n")).strip()) for i, a in results]
    out_df = pd.DataFrame(sorted(sanitized, key=lambda t: t[0]), columns=["index", "answer"])
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, header=False, index=False)
    print(f"\nwrote {len(out_df)} rows to {out_path}")
    print(f"log: {log_path}")


def main():
    args = parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
