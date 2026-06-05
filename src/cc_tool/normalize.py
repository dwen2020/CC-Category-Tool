"""Merchant-string normalization.

Used as both the learned-cache key and the basis for fuzzy matching. Any change
to the rules here invalidates existing cache entries — keep this the single
source of truth and never duplicate the logic elsewhere."""

import re

_PREFIX_STRIPS = re.compile(
    r"^(POS\s+|SQ\s*\*\s*|TST\s*\*\s*|PAYPAL\s*\*\s*|GOOGLE\s*\*\s*)",
    re.IGNORECASE,
)
_STORE_NUMBER = re.compile(r"\s+#?\d{3,}\b")
_WHITESPACE = re.compile(r"\s+")


def normalize_merchant(raw: str) -> str:
    if not raw:
        return ""
    s = raw.upper().strip()
    s = _PREFIX_STRIPS.sub("", s)
    s = _STORE_NUMBER.sub(" ", s)
    s = _WHITESPACE.sub(" ", s).strip()
    return s
