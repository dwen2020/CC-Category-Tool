# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A **local, single-user tool for tracking personal credit-card spending across categories.**
The user uploads PDF statements from **multiple banks / multiple cards**, the tool extracts
each transaction (date, raw merchant descriptor, amount), assigns a spending **category**, and
shows totals per category over time. Runs entirely on the user's machine.

## Commands

Set up (one-time, PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .
Copy-Item .env.example .env
# then edit .env and fill in FUELIX_API_KEY, FUELIX_BASE_URL, and FUELIX_MODEL
```

Run:

- Parse a statement PDF: `python -m cc_tool parse path\to\statement.pdf`
- JSON output: `python -m cc_tool parse path\to\statement.pdf --json`
- Override the model for one run: `python -m cc_tool parse path\to\statement.pdf --model anthropic/claude-3.5-sonnet`

## Stack

- Python (>=3.10), packaged with `pyproject.toml` + Hatchling.
- `openai` SDK pointed at TELUS Fuelix (OpenRouter-compatible). Credentials in `.env`
  via `python-dotenv`: `FUELIX_API_KEY`, `FUELIX_BASE_URL`, `FUELIX_MODEL`.
- `pdfplumber` for PDF text extraction (chat-completions endpoints don't accept PDFs
  directly the way Gemini does — text-first is the universally compatible path).
  Extraction uses `layout=True` to preserve rough column alignment for tabular
  statements.
- `pydantic` v2 for the parser's structured-output schema (`ParseResult` /
  `TransactionRow` in `src/cc_tool/schema.py`). The schema is sent to the model via
  the system prompt; the response comes back as JSON (`response_format={"type": "json_object"}`)
  and is validated by Pydantic on the way out.
- No database, no web UI, no categorizer in the current build. The architecture for
  those parts lives in `ARCHITECTURE.md` / `FLOW.md`; this build implements only the
  parse-and-reconcile slice.

Swapping the LLM backend later (local model, different provider) means writing a new
`StatementParser` subclass in `src/cc_tool/parser.py`; nothing else changes.
