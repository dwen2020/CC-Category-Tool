"""Spending-category assignment for parsed transactions.

Design:

  - A local DistilBERT classifier (see training/distilbert-uncased-trainer.ipynb) does the
    categorization -- fine-tuned on real merchant descriptors, runs entirely on
    the user's machine, no API key or network call required.
  - A per-merchant cache sits in front of it purely as MEMOIZATION, not as a
    rules engine: the first time a normalized merchant is seen it costs one
    inference call; every later occurrence is free. New merchants for a
    statement are batched into a single forward pass.
  - The cache key is the normalized merchant (see normalize.normalize_merchant),
    so store numbers / gateway prefixes collapse to one entry across banks/cards.
  - Every categorization carries a confidence score (the model's softmax
    probability for its chosen category), shown alongside the category in the
    "All merchants" browser (storage.all_merchants()) so a human can judge how
    much to trust it while scanning. User overrides always win and are written
    back with source="user", so a manual correction is permanent.

Only `purchase` rows are categorized. Payments, refunds, fees, and interest are
identified by `transaction_type` and are left with `category = None`.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path

from .categories import CATEGORIES, CATEGORY_SET
from .normalize import normalize_merchant
from .schema import ParseResult, TransactionRow


class CategorizerError(RuntimeError):
    pass


class MerchantCache:
    """On-disk memo of normalized-merchant -> category.

    Format: {"<normalized merchant>": {"category": "<cat>", "confidence": <float|null>, "source": "model|user"}}.
    """

    def __init__(self, path: Path):
        self._path = path
        self._data: dict[str, dict] = {}
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

    def get_confidence(self, merchant_key: str) -> float | None:
        entry = self._data.get(merchant_key)
        return entry.get("confidence") if entry else None

    def set(
        self, merchant_key: str, category: str, *, confidence: float | None = None, source: str
    ) -> None:
        self._data[merchant_key] = {
            "category": category,
            "confidence": confidence,
            "source": source,
        }

    def set_override(self, raw_merchant: str, category: str) -> None:
        """Record a user correction (wins over any model answer, persisted)."""
        if category not in CATEGORY_SET:
            raise CategorizerError(
                f"'{category}' is not a valid category. Choose from: {', '.join(CATEGORIES)}"
            )
        self.set(normalize_merchant(raw_merchant), category, confidence=1.0, source="user")
        self.save()


def default_model_path() -> Path:
    """Location of a locally-installed trained DistilBERT model folder.

    Honors CC_TOOL_MODEL_PATH if set; otherwise ~/.cc_tool/models/distilbert-merchant.
    This is checked first; if nothing is there, DistilBertCategorizer falls back
    to downloading the model from the Hub (see resolve_model_source below).
    """
    env = os.environ.get("CC_TOOL_MODEL_PATH")
    if env:
        return Path(env)
    return Path.home() / ".cc_tool" / "models" / "distilbert-merchant"


# Public Hub repo hosting the trained weights, used when no local copy exists.
# transformers caches the download under ~/.cache/huggingface after the first
# run, so this only costs a network call once per machine.
HF_MODEL_REPO_ID = "Dluvhugging/cc-tool-merchant-distilbert"


def resolve_model_source() -> str:
    """A local path if one is installed, else the public Hub repo id.

    AutoTokenizer/AutoModelForSequenceClassification.from_pretrained() accept
    either form, so callers don't need to know which one this resolved to.
    """
    local = default_model_path()
    if local.exists():
        return str(local)
    return HF_MODEL_REPO_ID


# Loaded (model, tokenizer) pairs, keyed by resolved model path. Loading
# DistilBERT from disk costs real time; callers (e.g. webapp.py) build a fresh
# DistilBertCategorizer per import, so this avoids reloading weights every time.
_MODEL_CACHE: dict[str, tuple] = {}
# Guards the check-then-load below. Without it, two callers racing to build a
# DistilBertCategorizer before either has populated the cache (e.g. the
# webapp's drop-folder watcher and an upload request firing on the same file,
# see webapp.py) both attempt to load the model from disk concurrently, which
# can throw -- and the loser's exception then silently disables categorization
# for that import.
_MODEL_LOAD_LOCK = threading.Lock()


class DistilBertCategorizer:
    """Local DistilBERT-backed categorization of a batch of merchant strings."""

    def __init__(self, model_path: Path | None = None):
        import torch  # noqa: F401  (import validated here, used in categorize_merchants)

        path = str(model_path) if model_path else resolve_model_source()
        with _MODEL_LOAD_LOCK:
            if path not in _MODEL_CACHE:
                from transformers import AutoModelForSequenceClassification, AutoTokenizer

                try:
                    tokenizer = AutoTokenizer.from_pretrained(path)
                    model = AutoModelForSequenceClassification.from_pretrained(path)
                except OSError as exc:
                    raise CategorizerError(
                        f"Could not load a model from '{path}' (local path does not "
                        f"exist and it was not downloadable from the Hub). Set "
                        "CC_TOOL_MODEL_PATH to a local model folder, or check your "
                        f"network connection. Original error: {exc}"
                    ) from exc
                model.eval()

                id2label = {int(i): label for i, label in model.config.id2label.items()}
                trained_labels = set(id2label.values())
                if trained_labels != CATEGORY_SET:
                    raise CategorizerError(
                        "Model label set does not match categories.CATEGORY_SET. "
                        f"Model has: {sorted(trained_labels)}. "
                        f"Expected: {sorted(CATEGORY_SET)}."
                    )
                _MODEL_CACHE[path] = (model, tokenizer, id2label)

            self._model, self._tokenizer, self._id2label = _MODEL_CACHE[path]

    def categorize_merchants(self, merchants: list[str]) -> dict[str, tuple[str, float]]:
        """Map each input merchant string to (category, confidence).

        Confidence is the model's softmax probability for its top class.
        """
        if not merchants:
            return {}

        import torch

        inputs = self._tokenizer(
            merchants,
            truncation=True,
            max_length=32,
            padding=True,
            return_tensors="pt",
            # DistilBERT has no segment/token-type embeddings; forward() rejects
            # token_type_ids outright if the tokenizer includes them.
            return_token_type_ids=False,
        )
        with torch.no_grad():
            logits = self._model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)
            confidences, indices = probs.max(dim=-1)

        result: dict[str, tuple[str, float]] = {}
        for merchant, idx, conf in zip(merchants, indices.tolist(), confidences.tolist()):
            result[merchant] = (self._id2label[idx], conf)
        return result


def default_cache_path() -> Path:
    """Location of the persistent merchant cache.

    Honors CC_TOOL_CACHE if set; otherwise ~/.cc_tool/merchant_categories.json.
    """
    env = os.environ.get("CC_TOOL_CACHE")
    if env:
        return Path(env)
    return Path.home() / ".cc_tool" / "merchant_categories.json"


class Categorizer:
    """Cache-fronted, DistilBERT-backed categorization of a parsed statement."""

    def __init__(self, model: DistilBertCategorizer, cache: MerchantCache | None = None):
        self._model = model
        self._cache = cache or MerchantCache(default_cache_path())

    def categorize(self, result: ParseResult, *, debug: bool = False) -> ParseResult:
        """Assign a spending category (and confidence) to every `purchase` row,
        in place.

        Cache hits are free; the batch of cache-missing merchants goes through
        the model in one forward pass and the answers are written back to the
        cache.
        """
        purchases = [r for r in result.rows if r.transaction_type == "purchase"]

        # normalized key -> (category, confidence), seeded from the cache.
        key_to_result: dict[str, tuple[str, float | None]] = {}
        misses: dict[str, str] = {}  # normalized key -> a representative raw descriptor
        for row in purchases:
            key = normalize_merchant(row.descriptor)
            cached = self._cache.get(key)
            if cached is not None:
                key_to_result[key] = (cached, self._cache.get_confidence(key))
            elif key and key not in misses:
                # Descriptors arrive already clean (the parser drops the issuer's
                # spending-category column), so the raw descriptor is what we send.
                misses[key] = row.descriptor

        if debug:
            print(
                f"[debug] categorize: {len(purchases)} purchase rows, "
                f"{len(key_to_result)} cache hits, {len(misses)} merchants to classify"
            )

        if misses:
            # Classify on the raw representative descriptor (matches the
            # model's training input), then store under the normalized key.
            merchant_list = list(misses.values())
            answers = self._model.categorize_merchants(merchant_list)
            for key, raw in misses.items():
                cat, conf = answers.get(raw, ("Other", 0.0))
                key_to_result[key] = (cat, conf)
                self._cache.set(key, cat, confidence=conf, source="model")
            self._cache.save()

        for row in purchases:
            cat, conf = key_to_result.get(normalize_merchant(row.descriptor), (None, None))
            row.category = cat
            row.confidence = conf

        return result
