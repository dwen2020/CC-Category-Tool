"""StatementParser interface and concrete implementations.

Future backends (deterministic per-bank, etc.) implement the same
StatementParser ABC and slot in without touching the rest of the app."""

from __future__ import annotations

import io
import json
import os
from abc import ABC, abstractmethod

from .schema import ParseResult, parse_json_schema


PARSE_INSTRUCTIONS = """\
You are parsing a credit card statement. Return a single JSON object. No markdown, no commentary.

--- FIELD RULES ---

date: TRANSACTION date (the "Trans date" column, not "Post date"). Always ISO
YYYY-MM-DD. Infer the year from the statement period.
  "Apr 23" in an April-May 2026 statement → "2026-04-23"

descriptor: Merchant name exactly as printed. Include store numbers and
city/province. EXCLUDE any spend-category label that appears in a separate
column on the same line (e.g. "Restaurants", "Transportation").
  "COFFEE SHOP VANCOUVER BC  Restaurants  12.34" → descriptor "COFFEE SHOP VANCOUVER BC"

amount_cents: Signed integer cents. All amounts print as positive in the PDF;
assign sign by type:
  purchases / fees / interest / cash advances → POSITIVE  ($18.90 → 1890)
  cardholder payments / merchant credits / refunds → NEGATIVE  ($529.42 payment → -52942)

transaction_type: "purchase" | "payment" | "refund" | "fee" | "interest"

--- WHAT TO INCLUDE ---

Include ALL dated line items: purchases, payments, fees, credits, interest.
Cardholder payments ARE transactions — include them.
EXCLUDE account summary lines: previous balance, new balance, minimum payment
due, available credit, credit limit.

--- OTHER FIELDS ---

issuer: bank/card name as printed (e.g. "Simplii Financial", "Canadian Tire Bank").
period_start / period_end: statement period in YYYY-MM-DD.
printed_total_cents: the single charges total printed for this period — look for
labels like "Total charges", "Total Purchases", "Total for [card number]", "New Purchases".
If none of those labels appear, fall back to "New Balance".
Integer cents. Example: $308.44 → 30844. Null only if no dollar total of any kind is present.
Do NOT use minimum payment due or available credit.
"""


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract text from a PDF.

    For each page, prefer structured table extraction (less noise for the model).
    Fall back to layout-preserving text for pages that have no detectable tables
    (summary sections, headers, etc. that live outside tables).
    """
    import pdfplumber

    pages: list[str] = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            if tables:
                rows: list[str] = []
                for table in tables:
                    for row in table:
                        if row and any(cell for cell in row if cell and cell.strip()):
                            rows.append(" | ".join(cell.strip() if cell else "" for cell in row))
                pages.append("\n".join(rows))
            else:
                pages.append(page.extract_text(layout=True) or "")
    return "\n\n--- PAGE BREAK ---\n\n".join(pages)


class StatementParser(ABC):
    @abstractmethod
    def parse(self, pdf_bytes: bytes, *, debug: bool = False) -> ParseResult: ...


class FuelixStatementParser(StatementParser):
    """Primary parser — uses the TELUS Fuelix (OpenRouter-compatible) API."""

    def __init__(self, model: str | None = None):
        from openai import OpenAI

        api_key = os.environ.get("FUELIX_API_KEY")
        base_url = os.environ.get("FUELIX_BASE_URL")
        mdl = model or os.environ.get("FUELIX_MODEL")

        if not api_key:
            raise RuntimeError("FUELIX_API_KEY is not set. Add it to .env.")
        if not base_url:
            raise RuntimeError("FUELIX_BASE_URL is not set. Add it to .env.")
        if not mdl:
            raise RuntimeError(
                "FUELIX_MODEL is not set. Add it to .env or pass --model."
            )

        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model = mdl

    def parse(self, pdf_bytes: bytes, *, debug: bool = False) -> ParseResult:
        text = extract_text_from_pdf(pdf_bytes)
        schema = parse_json_schema()

        system_prompt = (
            PARSE_INSTRUCTIONS
            + "\n\nThe JSON object you return must conform to this schema:\n"
            + json.dumps(schema, indent=2)
        )

        if debug:
            print(f"[debug] extracted text length: {len(text)} chars (~{len(text)//4} tokens est.)")

        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": (
                        "Below is the text extracted from a credit card "
                        "statement PDF. Parse it.\n\n"
                        f"{text}"
                    ),
                },
            ],
            response_format={"type": "json_object"},
        )

        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("Empty response from model.")

        if debug:
            usage = response.usage
            print(f"[debug] prompt_tokens={usage.prompt_tokens}  completion_tokens={usage.completion_tokens}  finish_reason={response.choices[0].finish_reason}")
            print(f"[debug] raw response (first 2000 chars):\n{content[:2000]}")

        return ParseResult.model_validate_json(content)


class OllamaStatementParser(StatementParser):
    """Local fallback parser — uses a model served by Ollama."""

    # Enough for a full multi-page statement including the system prompt and schema.
    _NUM_CTX = 32768

    def __init__(self, model: str | None = None):
        mdl = model or os.environ.get("OLLAMA_MODEL")
        if not mdl:
            raise RuntimeError(
                "OLLAMA_MODEL is not set. Add it to .env or pass --model."
            )
        self._model = mdl

    def parse(self, pdf_bytes: bytes, *, debug: bool = False) -> ParseResult:
        text = extract_text_from_pdf(pdf_bytes)
        schema = parse_json_schema()

        system_prompt = (
            PARSE_INSTRUCTIONS
            + "\n\nThe JSON object you return must conform to this schema:\n"
            + json.dumps(schema, indent=2)
        )

        if debug:
            print(f"[debug] extracted text length: {len(text)} chars (~{len(text)//4} tokens est.)")
            print(f"[debug] requesting num_ctx={self._NUM_CTX}")

        import ollama
        # trust_env=False prevents the corporate proxy from intercepting localhost.
        client = ollama.Client(host="http://localhost:11434", trust_env=False)
        response = client.chat(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": (
                        "Below is the text extracted from a credit card "
                        "statement PDF. Parse it.\n\n"
                        f"{text}"
                    ),
                },
            ],
            format=schema,
            options={"num_ctx": self._NUM_CTX},
        )

        if debug:
            print(
                f"[debug] prompt_tokens={response.prompt_eval_count}"
                f"  completion_tokens={response.eval_count}"
                f"  done_reason={response.done_reason}"
            )
            print(f"[debug] raw response (first 2000 chars):\n{response.message.content[:2000]}")

        content = response.message.content
        if not content:
            raise RuntimeError("Empty response from model.")
        return ParseResult.model_validate_json(content)
