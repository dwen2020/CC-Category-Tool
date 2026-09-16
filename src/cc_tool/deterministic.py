"""Issuer-agnostic statement parsing.

This tool has to work on a stranger's statement PDF with no per-bank code and
no LLM call (see the design discussion in ARCHITECTURE.md): each new issuer
would otherwise need its own hand-written parser, which doesn't scale past the
handful of banks any one person happens to have, and an LLM can silently drop
a row, flip a sign, or grab the wrong printed total on data where correctness
matters to the cent.

Instead, `GenericStatementParser` exploits the shape shared by almost every
statement -- dated line items whose first trailing amount is the transaction
amount, plus one labeled spending total -- and never trusts a guess it can't
check. Total detection in particular is reconciliation-driven rather than
phrase-driven (see `_find_total`): it doesn't need to recognize any one
issuer's exact wording for "total", because it verifies each candidate against
the rows it already extracted. The row-level extraction is checked the same
way, one level up: `reconcile.py` compares the parsed rows' sum to whatever
total was found, and a statement that doesn't reconcile is stored but flagged
for human review rather than trusted silently.
"""

from __future__ import annotations

import io
import re
from datetime import date

from .schema import ParseResult, TransactionRow


_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


def layout_text(pdf_bytes: bytes) -> str:
    """Full layout-preserving text of every page."""
    import pdfplumber

    pages: list[str] = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            pages.append(page.extract_text(layout=True) or "")
    return "\n".join(pages)


# --- Column-aware extraction --------------------------------------------------
#
# Some issuers (Simplii) print their own spending-category as a separate COLUMN
# between the descriptor and the amount. In plain text that column glues onto the
# descriptor ("...VANCOUVER BC Restaurants") where it is noise for the categorizer
# and the cache key. Rather than strip known label strings, we detect the column
# geometrically -- a run of words sharing a left edge across many rows, sitting in
# a gap between the descriptor and the amount -- and drop it at extraction time.
#
# Only transaction lines are touched; totals/period/header lines pass through
# unchanged, so reconciliation cannot regress. If no such column is found
# (Rogers, Canadian Tire), the output equals a plain word-join of every line.

# A leading date token: month word (optionally with glued day), or numeric date.
_DATE_TOKEN = re.compile(
    r"^(?:(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\d{0,2}"
    r"|\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?|\d{4}-\d{2}-\d{2})$",
    re.IGNORECASE,
)
_MONEY_TOKEN = re.compile(r"^-?\(?\$?\d{1,3}(?:,\d{3})*\.\d{2}\)?$")

# Horizontal gap (points) big enough to mark a real column break, well above
# inter-word spacing on these statements (~2-21pt) and below the descriptor->
# category gap (~57pt).
_COLUMN_GAP = 30.0


def _median(values: list[float]) -> float:
    s = sorted(values)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def _group_lines(words: list[dict], ytol: float = 3.0) -> list[list[dict]]:
    """Cluster words into visual lines by their `top`, each sorted left to right."""
    lines: list[list[dict]] = []
    cur: list[dict] = []
    cur_top: float | None = None
    for w in sorted(words, key=lambda w: (w["top"], w["x0"])):
        if cur_top is None or abs(w["top"] - cur_top) <= ytol:
            cur.append(w)
            cur_top = w["top"] if cur_top is None else cur_top
        else:
            lines.append(sorted(cur, key=lambda x: x["x0"]))
            cur, cur_top = [w], w["top"]
    if cur:
        lines.append(sorted(cur, key=lambda x: x["x0"]))
    return lines


def _is_txn_line(tokens: list[dict]) -> bool:
    return bool(
        tokens
        and _DATE_TOKEN.match(tokens[0]["text"])
        and any(_MONEY_TOKEN.match(t["text"]) for t in tokens)
    )


def _detect_category_band(lines: list[list[dict]]) -> tuple[float, float] | None:
    """Return (left, right) x-range of the spending-category column, or None.

    The column is the group of words after the LAST large horizontal gap that
    lies left of the amount, but only if its left edge is consistent across at
    least half the transaction rows (which is what tells a real column apart from
    a one-off wide space inside a descriptor).
    """
    from collections import Counter

    txn = [l for l in lines if _is_txn_line(l)]
    if len(txn) < 3:
        return None

    amount_lefts = [
        [t for t in l if _MONEY_TOKEN.match(t["text"])][-1]["x0"] for l in txn
    ]
    amount_left = _median(amount_lefts)

    starts: list[int] = []
    for l in txn:
        left_of_amt = [t for t in l if t["x1"] < amount_left - 5]
        last_break = None
        for i in range(1, len(left_of_amt)):
            if left_of_amt[i]["x0"] - left_of_amt[i - 1]["x1"] > _COLUMN_GAP:
                last_break = i
        if last_break is not None:
            starts.append(round(left_of_amt[last_break]["x0"]))

    if not starts:
        return None
    edge, count = Counter(starts).most_common(1)[0]
    if count < 0.5 * len(txn):
        return None
    return (edge - 3.0, amount_left - 5.0)


def column_aware_text(pdf_bytes: bytes) -> str:
    """Layout text with any detected spending-category column removed from
    transaction rows. Falls back to a faithful word-join when no column is found.
    """
    import pdfplumber

    pages: list[str] = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            lines = _group_lines(page.extract_words())
            band = _detect_category_band(lines)
            out: list[str] = []
            for line in lines:
                if band and _is_txn_line(line):
                    lo, hi = band
                    toks = [t for t in line if not (lo <= t["x0"] < hi)]
                else:
                    toks = line
                out.append(" ".join(t["text"] for t in toks))
            pages.append("\n".join(out))
    return "\n".join(pages)


def amount_to_cents(raw: str) -> int:
    """'$1,123.47' / '-272.23' -> signed integer cents."""
    s = raw.strip()
    neg = s.startswith("-") or s.startswith("(")
    digits = re.sub(r"[^\d.]", "", s)
    cents = round(float(digits) * 100)
    return -cents if neg else cents


def classify(descriptor: str, cents: int) -> str:
    """Transaction type from the descriptor and sign.

    Only these three types occur as line items in the supported statements;
    fees/interest print as summary rows (0.00) rather than dated lines.
    """
    d = descriptor.upper()
    if "PAYMENT" in d or "PAIEMENT" in d:
        return "payment"
    if cents < 0:
        return "refund"
    return "purchase"


class GenericStatementParser:
    """Issuer-agnostic heuristic parser.

    Exploits the shape shared by almost every statement: dated line items whose
    first trailing amount is the transaction amount, plus one labeled spending
    total. It has no per-bank knowledge, so it is only trustworthy when its rows
    reconcile against the printed total -- see reconcile.py, which is what the
    caller (importer.py) checks before trusting an import. Descriptors may be
    messier than a dedicated parser (embedded spend-category labels, etc.), but
    the amounts are what reconcile, and reconciliation is the correctness gate.
    """

    # A date at the start of a line: "Apr 23", "Apr30", "04/23", "2026-04-23".
    _DATE = r"(?:[A-Z][a-z]{2}\.?\s?\d{1,2}|\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?|\d{4}-\d{2}-\d{2})"
    _MONEY = r"-?\(?\$?\d{1,3}(?:,\d{3})*\.\d{2}\)?"
    _LEAD = re.compile(rf"^\s*({_DATE})(?:\s+{_DATE})?\s+(.*)$")
    _MONEY_RE = re.compile(_MONEY)
    _FULL_DATE = re.compile(r"([A-Za-z]{3,9})\.?\s?(\d{1,2}),\s*(\d{4})")

    # Some issuers (e.g. Canadian Tire/Triangle) follow a dated item line with
    # an itemized breakdown: tax sub-lines, an "OTHER TENDER" line for any
    # amount paid via redeemed loyalty points, and a "Total for transaction"
    # line with what was actually charged to the card. The dated line's own
    # trailing number is the pre-tax item price -- NOT what was charged --
    # so when this line appears before the next dated line, it overrides it.
    _TOTAL_FOR_TXN = re.compile(r"Total\s+for\s+transaction\s+\$?([\d,]+\.\d{2})", re.IGNORECASE)
    _TOTAL_FOR_TXN_LOOKAHEAD = 8

    # These itemized breakdowns are a supplementary appendix, not additional
    # transactions -- the statement says so explicitly ("The total of each
    # transaction is included in the Purchases section"). It always appears
    # after the real transaction list, with nothing transaction-like after it,
    # so once seen, every following dated-looking line is skipped rather than
    # double-counted alongside the summary line it's restating.
    _ITEMIZED_APPENDIX_MARKER = re.compile(r"included in the Purchases section", re.IGNORECASE)

    # Exact-phrase fallback, tried only when _find_total's reconciliation-driven
    # pass (below) has no row sum to check against, or genuinely finds nothing.
    # Tried in order; first match wins. Inter-word whitespace is optional so
    # glued text like "Newpurchases &debits" still matches.
    _TOTAL_PATTERNS = [
        r"New\s*purchases?\s*&\s*debits?\s*\$?([\d,]+\.\d{2})",
        r"Total\s*charges\s*\+?\s*\$?([\d,]+\.\d{2})",
        r"Total\s*purchases\s*\$?([\d,]+\.\d{2})",
        r"New\s*purchases\s*\$?([\d,]+\.\d{2})",
        r"Total\s*this\s*period\s*\$?([\d,]+\.\d{2})",
        r"New\s*balance\s*=?\s*\$?([\d,]+\.\d{2})",
        r"\+\s*Purchases\s*\$?([\d,]+\.\d{2})",  # PC Financial's summary-box wording
    ]

    # Generic words that plausibly sit next to a statement's own total, used by
    # the reconciliation-driven pass in _find_total. Deliberately not tied to
    # any one issuer's exact phrasing (that's what _TOTAL_PATTERNS is, and it
    # requires a new entry per issuer) -- this only needs ONE of these words to
    # appear somewhere near the real total, for ANY issuer, seen or unseen.
    _TOTAL_KEYWORD_RE = re.compile(
        r"total|purchase|balance|charges|debits|amount\s*due", re.IGNORECASE
    )

    def parse(self, pdf_bytes: bytes, *, debug: bool = False) -> ParseResult:
        return self.parse_text(column_aware_text(pdf_bytes), debug=debug)

    def parse_text(self, text: str, *, debug: bool = False) -> ParseResult:
        full_dates = [
            date(int(y), _MONTHS[mon[:3].lower()], int(d))
            for mon, d, y in self._FULL_DATE.findall(text)
            if mon[:3].lower() in _MONTHS
        ]
        ref_end = max(full_dates) if full_dates else None
        ref_start = min(full_dates) if full_dates else None

        lines = text.splitlines()
        rows: list[TransactionRow] = []
        in_itemized_appendix = False
        for i, line in enumerate(lines):
            if self._ITEMIZED_APPENDIX_MARKER.search(line):
                in_itemized_appendix = True
            if in_itemized_appendix:
                continue
            m = self._LEAD.match(line)
            if not m:
                continue
            date_tok, rest = m.group(1), m.group(2)
            md = self._parse_date_token(date_tok)
            if md is None:
                continue
            money = self._MONEY_RE.search(rest)
            if not money:
                continue
            month, day = md
            desc = rest[: money.start()].strip()
            if not desc:
                continue
            cents = amount_to_cents(money.group(0))

            # Look ahead (up to the next dated line) for an authoritative
            # "Total for transaction" figure that supersedes the item price.
            for j in range(i + 1, min(i + 1 + self._TOTAL_FOR_TXN_LOOKAHEAD, len(lines))):
                if self._LEAD.match(lines[j]):
                    break
                total = self._TOTAL_FOR_TXN.search(lines[j])
                if total:
                    cents = amount_to_cents(total.group(1))
                    break

            ttype = classify(desc, cents)
            if ttype == "payment":
                cents = -abs(cents)
            yr = self._resolve_year(month, day, ref_end)
            rows.append(
                TransactionRow(
                    date=f"{yr:04d}-{month:02d}-{day:02d}",
                    descriptor=desc,
                    amount_cents=cents,
                    transaction_type=ttype,
                )
            )

        if debug:
            print(f"[debug] generic: {len(rows)} candidate rows, "
                  f"ref period {ref_start}..{ref_end}")

        rows_sum_cents = sum(r.amount_cents for r in rows if r.transaction_type == "purchase")
        return ParseResult(
            rows=rows,
            printed_total_cents=self._find_total(text, rows_sum_cents),
            issuer=None,
            period_start=ref_start.isoformat() if ref_start else None,
            period_end=ref_end.isoformat() if ref_end else None,
        )

    def _parse_date_token(self, tok: str) -> tuple[int, int] | None:
        tok = tok.strip()
        m = re.match(r"([A-Za-z]{3})\.?\s?(\d{1,2})$", tok)
        if m and m.group(1).lower() in _MONTHS:
            return _MONTHS[m.group(1).lower()], int(m.group(2))
        m = re.match(r"(\d{4})-(\d{2})-(\d{2})$", tok)
        if m:
            return int(m.group(2)), int(m.group(3))
        m = re.match(r"(\d{1,2})[/-](\d{1,2})", tok)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            # Default assumption is MM/DD, but some issuers (e.g. PC Financial)
            # print DD/MM instead. Whichever slot holds a value >12 can't be a
            # month, so it must be the day -- reinterpret rather than silently
            # emit an impossible month like "22" (which nothing here validates
            # against a real calendar, since `date` is stored as plain text).
            if a > 12 and b <= 12:
                return b, a
            if a <= 12 and b > 12:
                return a, b
            if a > 12 and b > 12:
                return None  # neither slot can be a month; unparseable
            return a, b  # both <=12: genuinely ambiguous, keep the MM/DD default
        return None

    def _resolve_year(self, month: int, day: int, ref_end: date | None) -> int:
        if ref_end is None:
            return date.today().year
        for yr in (ref_end.year, ref_end.year - 1):
            try:
                d = date(yr, month, day)
            except ValueError:
                continue
            if d <= ref_end:
                return yr
        return ref_end.year

    def _find_total(self, text: str, rows_sum_cents: int | None = None) -> int | None:
        """Two passes, in order:

        1. Reconciliation-driven (issuer-agnostic): collect every dollar amount
           on a line that mentions one of the generic words in
           _TOTAL_KEYWORD_RE, and accept whichever one matches `rows_sum_cents`
           to the cent. This needs no exact phrase and no issuer-specific code
           -- reconciliation itself picks the right candidate out of several,
           which is what lets it generalize to statement wording never seen
           before (verified across 4 issuers / 25 real statements, each with
           different total phrasing, before this replaced the old sole
           mechanism below).
        2. The exact-phrase _TOTAL_PATTERNS list, tried only if pass 1 found no
           reconciling candidate (no row sum available, or genuinely no dollar
           amount near those words matches it -- e.g. mid-parse debugging via
           parse() without rows, or a statement whose total wording uses none
           of the generic words at all).
        """
        if rows_sum_cents is not None:
            for line in text.splitlines():
                if not self._TOTAL_KEYWORD_RE.search(line):
                    continue
                for m in self._MONEY_RE.finditer(line):
                    if amount_to_cents(m.group(0)) == rows_sum_cents:
                        return rows_sum_cents

        for pat in self._TOTAL_PATTERNS:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                return amount_to_cents(m.group(1))
        return None
