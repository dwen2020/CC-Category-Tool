"""StatementParser interface and Ollama implementation.

Future backends (deterministic per-bank, etc.) implement the same
StatementParser ABC and slot in without touching the rest of the app."""

from __future__ import annotations

import io
import json
import os
from abc import ABC, abstractmethod

from .schema import ParseResult


PARSE_INSTRUCTIONS = """\
You are parsing a credit card statement.

Extract every line-item transaction (date, descriptor, amount), the statement's
printed transactions total if shown, and the issuer name and statement period
if shown.

Rules:
- amount_cents is a SIGNED integer in CENTS. Charges and purchases are POSITIVE.
  Credits, refunds, and payments received are NEGATIVE.
- date is ISO format YYYY-MM-DD. If the statement only shows MM/DD, infer the
  year from the statement period.
- descriptor is the merchant string exactly as printed, including any store
  numbers or city/state suffixes. Do not normalize or clean it.
- Do NOT include summary lines (previous balance, new balance, payment due,
  minimum payment, available credit) as transactions.
- Interest charges are transactions only if they appear in the dated
  transactions list.
- printed_total_cents is the sum the statement itself prints (commonly labeled
  "Transactions", "Total Purchases", or "Total this period"). Null if absent.

Return a single JSON object that conforms to the provided schema. Do not wrap
it in markdown, do not add commentary.
"""


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract text from a PDF, preserving rough column layout via pdfplumber's
    layout=True option. Statements are tabular; layout preservation matters."""
    import pdfplumber

    pages: list[str] = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            text = page.extract_text(layout=True) or ""
            pages.append(text)
    return "\n\n--- PAGE BREAK ---\n\n".join(pages)


class StatementParser(ABC):
    @abstractmethod
    def parse(self, pdf_bytes: bytes) -> ParseResult: ...


_OLLAMA_BASE_URL = "http://localhost:11434/v1"


class OllamaStatementParser(StatementParser):
    def __init__(self, model: str | None = None):
        import httpx
        from openai import OpenAI

        mdl = model or os.environ.get("OLLAMA_MODEL")
        if not mdl:
            raise RuntimeError(
                "OLLAMA_MODEL is not set. Add it to .env or pass --model."
            )
        # trust_env=False prevents corporate proxies from intercepting localhost traffic.
        self._client = OpenAI(
            api_key="ollama",
            base_url=_OLLAMA_BASE_URL,
            http_client=httpx.Client(trust_env=False),
        )
        self._model = mdl

    def parse(self, pdf_bytes: bytes) -> ParseResult:
        text = extract_text_from_pdf(pdf_bytes)
        schema = ParseResult.model_json_schema()

        system_prompt = (
            PARSE_INSTRUCTIONS
            + "\n\nThe JSON object you return must conform to this schema:\n"
            + json.dumps(schema, indent=2)
        )

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
        return ParseResult.model_validate_json(content)
