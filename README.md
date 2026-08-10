# datastel-rag

Agentic RAG pipeline for the AI Engineering Challenge (data-astel fictional
consulting firm). Reads the raw share-drive folder distributed by the
competition and answers the question set with cited, computed answers --
no per-question or per-file hardcoding.

## Setup

```bash
uv sync
cp .env.example .env  # fill in ANTHROPIC_API_KEY / OPENAI_API_KEY
```

Data is expected one level up, at `../share/` (as distributed).

## Layout

- `src/datastel_rag/paths.py` -- NFC/NFD-safe path helpers. The share drive
  was zipped on macOS, so kana with dakuten (e.g. `ドライブ`) are stored
  NFD-decomposed on disk; never hand-type a Japanese path segment.
- `src/datastel_rag/catalog/` -- `scanner.py` walks the share drive into a
  file inventory; `glossary.py` parses 社内用語集.docx into term/abbreviation
  lookups (including the project full-name <-> 主略称 mapping).
- `src/datastel_rag/ingest/` -- format-specific parsers producing a common
  intermediate representation, plus `decrypt.py` for the password-derivation
  rule that guards some office files.
- `src/datastel_rag/index/` -- hybrid (BM25 + catalog-routed) search over
  parsed documents.
- `src/datastel_rag/agent/` -- Claude Agent SDK orchestration and tools.

## Known open issue

`ingest/decrypt.py` implements the documented password rule
(`DA-[案件略号]-[開始年月日8桁]-[拡張子コード]`) with date candidates
harvested from sibling files. It reliably cracks some files (verified against
かえで's `スケジュール.xlsx`) but not all (かえで's `契約書*.docx` resists
every extension-code variant tried so far, despite the abbreviation and date
being independently confirmed). Treat this as an agent-callable tool with
extra candidate passwords, not a guaranteed pre-processing step.
