"""Entrypoint: reads a questions CSV (index,question), answers each one via
the agent, and writes predictions.csv (index,answer -- no header, matching
evaluation/src/validator.py's expected format).

Usage:
    uv run python -m datastel_rag.run --questions ../share/質問回答/questions_valid.csv --out submit/predictions.csv
    uv run python -m datastel_rag.run --questions ... --out ... --limit 5   # smoke test on the first 5 rows
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

import pandas as pd

from datastel_rag import config
from datastel_rag.agent.orchestrator import answer_question_async
from datastel_rag.catalog.glossary import load_or_build_glossary
from datastel_rag.catalog.scanner import load_or_build_catalog
from datastel_rag.index.store import SearchIndex


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--questions", required=True, help="questions CSV path (columns: index,question)")
    p.add_argument("--out", required=True, help="output predictions.csv path")
    p.add_argument("--log", default=None, help="JSONL run log path (default: logs/run_<timestamp>.jsonl)")
    p.add_argument("--limit", type=int, default=None, help="only answer the first N rows (smoke testing)")
    p.add_argument("--concurrency", type=int, default=3)
    p.add_argument("--max-turns", type=int, default=50)
    p.add_argument("--max-budget-usd", type=float, default=1.5)
    p.add_argument("--model", default=None)
    p.add_argument("--refresh-index", action="store_true", help="force re-parse everything instead of using the cache")
    return p.parse_args()


async def _run_one(sem, idx_num, question, index, catalog, glossary, args, log_f):
    async with sem:
        start = time.time()
        result = await answer_question_async(
            question,
            index,
            catalog,
            glossary,
            max_turns=args.max_turns,
            max_budget_usd=args.max_budget_usd,
            model=args.model,
        )
        elapsed = time.time() - start
        record = {
            "index": idx_num,
            "question": question,
            "answer": result.answer,
            "num_turns": result.num_turns,
            "cost_usd": result.cost_usd,
            "elapsed_s": round(elapsed, 1),
            "is_error": result.is_error,
            "error_detail": result.error_detail,
            "session_id": result.session_id,
            "model": args.model or config.ANTHROPIC_MODEL,
        }
        log_f.write(json.dumps(record, ensure_ascii=False) + "\n")
        log_f.flush()
        print(f"[{idx_num}] turns={result.num_turns} cost=${result.cost_usd:.3f} err={result.is_error} :: {result.answer[:80]!r}")
        return idx_num, result.answer


async def main_async(args):
    questions_df = pd.read_csv(args.questions)
    if args.limit:
        questions_df = questions_df.head(args.limit)

    catalog = load_or_build_catalog(refresh=args.refresh_index)
    glossary = load_or_build_glossary(refresh=args.refresh_index)
    index = SearchIndex()
    index.build(catalog, glossary, force=args.refresh_index)

    log_path = Path(args.log) if args.log else config.LOG_DIR / f"run_{time.strftime('%Y%m%d_%H%M%S')}.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    sem = asyncio.Semaphore(args.concurrency)
    with open(log_path, "w", encoding="utf-8") as log_f:
        tasks = [
            _run_one(sem, int(row["index"]), row["question"], index, catalog, glossary, args, log_f)
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
