"""Spending-category assignment for parsed transactions.

Design (see the conversation that produced it):

  - The LLM does the actual thinking. It already knows what "SQ *BLUE BOTTLE" or
    "PAYPAL *STEAM" is, which a from-scratch classifier or an embedding model only
    approximates -- and we have no labeled training data, so a supervised model is
    a non-starter.
  - A per-merchant cache sits in front of the LLM purely as MEMOIZATION, not as a
    rules engine. The first time a normalized merchant is seen it costs one LLM
    call; every later occurrence is free. New merchants for a statement are batched
    into a single request.
  - The cache key is the normalized merchant (see normalize.normalize_merchant),
    so store numbers / gateway prefixes collapse to one entry across banks/cards.
  - User overrides always win and are written back with source="user", so a manual
    correction is permanent.

Only `purchase` rows are categorized. Payments, refunds, fees, and interest are
identified by `transaction_type` and are left with `category = None`.

Swapping the LLM backend = a new LLMCategorizer subclass; nothing else changes
(mirrors the StatementParser design in parser.py).
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from pathlib import Path

from .categories import CATEGORIES, CATEGORY_SET
from .normalize import normalize_merchant
from .schema import ParseResult, TransactionRow


CATEGORIZE_INSTRUCTIONS = """\
You categorize credit-card merchants for a personal spending tracker.

You are given a JSON array of merchant descriptor strings. Assign EACH one to
exactly one category from this closed list -- never invent a category:

{categories}

Guidance:
- Pick the single best fit based on what the merchant primarily sells.
- "Dining" covers restaurants, cafes, coffee shops, bars, fast food, and food
  delivery. "Groceries" is supermarkets and grocery stores only.
- "Transport" covers gas, transit, rideshare, parking, tolls, and taxis.
- "Entertainment" covers streaming, games, movies, events, and subscriptions.
- Use "Other" ONLY when nothing else plausibly fits. Do not overuse it.

Return a single JSON object mapping each input merchant string, EXACTLY as given,
to its category string. No markdown, no commentary. Example:
{{"TIM HORTONS": "Dining", "PETRO-CANADA": "Transport"}}
"""


class CategorizerError(RuntimeError):
    pass


class MerchantCache:
    """On-disk memo of normalized-merchant -> category.

    Format: {"<normalized merchant>": {"category": "<cat>", "source": "llm|user"}}.
    """

    def __init__(self, path: Path):
        self._path = path
        self._data: dict[str, dict[str, str]] = {}
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            try:
                self._data = json.loads(self._path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                # A corrupt cache should not break categorization; start fresh.
                self._data = {}

    def save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(self._data, indent=2, sort_keys=True), encoding="utf-8"
        )

    def get(self, merchant_key: str) -> str | None:
        entry = self._data.get(merchant_key)
        return entry["category"] if entry else None

    def set(self, merchant_key: str, category: str, *, source: str) -> None:
        self._data[merchant_key] = {"category": category, "source": source}

    def set_override(self, raw_merchant: str, category: str) -> None:
        """Record a user correction (wins over any LLM answer, persisted)."""
        if category not in CATEGORY_SET:
            raise CategorizerError(
                f"'{category}' is not a valid category. Choose from: {', '.join(CATEGORIES)}"
            )
        self.set(normalize_merchant(raw_merchant), category, source="user")
        self.save()


class LLMCategorizer(ABC):
    """Backend-agnostic categorization over a batch of merchant strings."""

    def categorize_merchants(self, merchants: list[str]) -> dict[str, str]:
        """Map each input merchant string to a valid category.

        Any answer outside the closed category set (or any merchant the model
        omits) is coerced to "Other" so callers always get a complete, valid map.
        """
        if not merchants:
            return {}

        system = CATEGORIZE_INSTRUCTIONS.format(
            categories="\n".join(f"- {c}" for c in CATEGORIES)
        )
        user = json.dumps(merchants, ensure_ascii=False)
        raw = self._complete(system, user)
        if not raw:
            raise CategorizerError("Empty response from categorization model.")

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as e:
            raise CategorizerError(f"Model did not return valid JSON: {e}") from e
        if not isinstance(parsed, dict):
            raise CategorizerError("Model response was not a JSON object.")

        result: dict[str, str] = {}
        for m in merchants:
            cat = parsed.get(m)
            result[m] = cat if cat in CATEGORY_SET else "Other"
        return result

    @abstractmethod
    def _complete(self, system: str, user: str) -> str | None:
        """Return the model's raw text response to (system, user)."""
        ...


class FuelixCategorizer(LLMCategorizer):
    """Primary categorizer -- TELUS Fuelix (OpenRouter-compatible) API."""

    def __init__(self, model: str | None = None):
        from openai import OpenAI

        api_key = os.environ.get("FUELIX_API_KEY")
        base_url = os.environ.get("FUELIX_BASE_URL")
        mdl = model or os.environ.get("FUELIX_MODEL")

        if not api_key:
            raise CategorizerError("FUELIX_API_KEY is not set. Add it to .env.")
        if not base_url:
            raise CategorizerError("FUELIX_BASE_URL is not set. Add it to .env.")
        if not mdl:
            raise CategorizerError("FUELIX_MODEL is not set. Add it to .env or pass --model.")

        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model = mdl

    def _complete(self, system: str, user: str) -> str | None:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content


class OllamaCategorizer(LLMCategorizer):
    """Local fallback categorizer -- a model served by Ollama."""

    def __init__(self, model: str | None = None):
        mdl = model or os.environ.get("OLLAMA_MODEL")
        if not mdl:
            raise CategorizerError("OLLAMA_MODEL is not set. Add it to .env or pass --model.")
        self._model = mdl

    def _complete(self, system: str, user: str) -> str | None:
        import ollama

        # trust_env=False prevents the corporate proxy from intercepting localhost.
        client = ollama.Client(host="http://localhost:11434", trust_env=False)
        response = client.chat(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            format="json",
        )
        return response.message.content


def default_cache_path() -> Path:
    """Location of the persistent merchant cache.

    Honors CC_TOOL_CACHE if set; otherwise ~/.cc_tool/merchant_categories.json.
    """
    env = os.environ.get("CC_TOOL_CACHE")
    if env:
        return Path(env)
    return Path.home() / ".cc_tool" / "merchant_categories.json"


class Categorizer:
    """Cache-fronted, LLM-backed categorization of a parsed statement."""

    def __init__(self, llm: LLMCategorizer, cache: MerchantCache | None = None):
        self._llm = llm
        self._cache = cache or MerchantCache(default_cache_path())

    def categorize(self, result: ParseResult, *, debug: bool = False) -> ParseResult:
        """Assign a spending category to every `purchase` row, in place.

        Cache hits are free; the batch of cache-missing merchants goes to the LLM
        in one call and the answers are written back to the cache.
        """
        purchases = [r for r in result.rows if r.transaction_type == "purchase"]

        # normalized key -> category, seeded from the cache.
        key_to_cat: dict[str, str] = {}
        misses: dict[str, str] = {}  # normalized key -> a representative raw descriptor
        for row in purchases:
            key = normalize_merchant(row.descriptor)
            cached = self._cache.get(key)
            if cached is not None:
                key_to_cat[key] = cached
            elif key and key not in misses:
                misses[key] = row.descriptor

        if debug:
            print(
                f"[debug] categorize: {len(purchases)} purchase rows, "
                f"{len(key_to_cat)} cache hits, {len(misses)} merchants to classify"
            )

        if misses:
            # Query on the raw representative descriptor (more context for the
            # model than the stripped key), then store under the normalized key.
            merchant_list = list(misses.values())
            answers = self._llm.categorize_merchants(merchant_list)
            for key, raw in misses.items():
                cat = answers.get(raw, "Other")
                key_to_cat[key] = cat
                self._cache.set(key, cat, source="llm")
            self._cache.save()

        for row in purchases:
            row.category = key_to_cat.get(normalize_merchant(row.descriptor))

        return result
