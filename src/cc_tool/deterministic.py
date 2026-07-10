"""Deterministic, per-issuer statement parsers.

Rationale: extracting dates and amounts from a credit-card statement is an exact
task (cents must reconcile) over rigidly structured text. Regex/positional
parsing is more reliable there than an LLM, which can silently drop a row, flip a
sign, or grab the wrong printed total.

The tradeoff is coverage: each issuer needs its own parser. To handle new cards
that don't have one yet, `DeterministicStatementParser` detects the issuer and
raises `UnknownIssuerError` when none matches. `AutoStatementParser` wraps that
and falls back to an LLM parser, so a brand-new statement still parses (just
without the determinism guarantee) until a dedicated parser is added.

Adding a new issuer = one `IssuerParser` subclass registered in `_ISSUERS`;
nothing else changes.
"""

from __future__ import annotations

import io
import re
from abc import ABC, abstractmethod
from datetime import date

from .schema import ParseResult, TransactionRow
from .parser import StatementParser


class UnknownIssuerError(RuntimeError):
    """No registered deterministic parser recognized the statement."""


_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


def layout_text(pdf_bytes: bytes) -> str:
    """Full layout-preserving text of every page.

    Unlike the table-first extraction in parser.py, this keeps the raw
    transaction rows intact (that extraction dropped them on some layouts).
    """
    import pdfplumber

    pages: list[str] = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            pages.append(page.extract_text(layout=True) or "")
    return "\n".join(pages)


def amount_to_cents(raw: str) -> int:
    """'$1,123.47' / '-272.23' -> signed integer cents."""
    s = raw.strip()
    neg = s.startswith("-") or s.startswith("(")
    digits = re.sub(r"[^\d.]", "", s)
    cents = round(float(digits) * 100)
    return -cents if neg else cents


def resolve_year(month: int, day: int, p_start: date, p_end: date) -> int:
    """Pick the year (start or end of the period) that puts month/day inside it.

    Handles statements that straddle a year boundary (Dec -> Jan).
    """
    for yr in {p_start.year, p_end.year}:
        try:
            d = date(yr, month, day)
        except ValueError:
            continue
        if p_start <= d <= p_end:
            return yr
    return p_start.year


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


class IssuerParser(ABC):
    """One concrete statement format."""

    name: str

    @abstractmethod
    def matches(self, text: str) -> bool: ...

    @abstractmethod
    def parse(self, text: str) -> ParseResult: ...


class CanadianTireParser(IssuerParser):
    name = "Canadian Tire Bank"

    _PERIOD = re.compile(
        r"For the period:\s*([A-Za-z]+ \d{1,2}, \d{4})\s*to\s*([A-Za-z]+ \d{1,2}, \d{4})"
    )
    # Trans-date, post-date, descriptor, first dollar amount. The right-hand
    # legal column bleeds onto these lines after the amount, so we do NOT anchor
    # the end and take the FIRST amount that follows the descriptor.
    _ROW = re.compile(
        r"^\s*([A-Z][a-z]{2})\s+(\d{1,2})\s+[A-Z][a-z]{2}\s+\d{1,2}\s+"
        r"(.+?)\s+(-?\$?[\d,]+\.\d{2})(?:\s|$)"
    )
    _TOTAL = re.compile(r"Total charges\s+\$?([\d,]+\.\d{2})")

    def matches(self, text: str) -> bool:
        return "Canadian Tire Bank" in text or "Triangle World Elite" in text

    def parse(self, text: str) -> ParseResult:
        return _parse_two_column(
            text,
            issuer=self.name,
            period_re=self._PERIOD,
            row_re=self._ROW,
            total_re=self._TOTAL,
            spaced_dates=True,
        )


class SimpliiParser(IssuerParser):
    name = "Simplii Financial"

    _PERIOD = re.compile(
        r"([A-Za-z]+ \d{1,2})\s*to\s*([A-Za-z]+ \d{1,2}, \d{4})"
    )
    # Descriptor carries an embedded spend-category before the amount; the row
    # ends at the amount (no trailing column bleed on this layout).
    _ROW = re.compile(
        r"^\s*([A-Z][a-z]{2})\s+(\d{1,2})\s+[A-Z][a-z]{2}\s+\d{1,2}\s+"
        r"(.+?)\s+(-?[\d,]+\.\d{2})\s*$"
    )
    _TOTAL = re.compile(r"Total charges\s*\+?\s*\$?([\d,]+\.\d{2})")

    # Longest first so multi-word categories strip before their substrings.
    _CATEGORIES = [
        "Personal and Household Expenses",
        "Professional and Financial Services",
        "Hotel, Entertainment and Recreation",
        "Retail and Grocery",
        "Other Transactions",
        "Transportation",
        "Restaurants",
    ]

    def matches(self, text: str) -> bool:
        return "Simplii Financial" in text

    def parse(self, text: str) -> ParseResult:
        p_start, p_end = _simplii_period(text, self._PERIOD)
        rows: list[TransactionRow] = []
        for line in text.splitlines():
            m = self._ROW.match(line)
            if not m:
                continue
            mon, day, desc, amt = m.groups()
            month = _MONTHS.get(mon.lower())
            if month is None:
                continue
            desc = self._strip_category(desc.strip())
            cents = amount_to_cents(amt)
            ttype = classify(desc, cents)
            if ttype == "payment":
                cents = -abs(cents)
            yr = resolve_year(month, int(day), p_start, p_end)
            rows.append(
                TransactionRow(
                    date=f"{yr:04d}-{month:02d}-{int(day):02d}",
                    descriptor=desc,
                    amount_cents=cents,
                    transaction_type=ttype,
                )
            )
        total = self._TOTAL.search(text)
        return ParseResult(
            rows=rows,
            printed_total_cents=amount_to_cents(total.group(1)) if total else None,
            issuer=self.name,
            period_start=p_start.isoformat(),
            period_end=p_end.isoformat(),
        )

    def _strip_category(self, desc: str) -> str:
        for cat in self._CATEGORIES:
            if desc.endswith(cat):
                return desc[: -len(cat)].strip()
        return desc


class RogersBankParser(IssuerParser):
    name = "Rogers Bank"

    # Dates print with no space: "Apr19,2026-May18,2026".
    _PERIOD = re.compile(
        r"Statement Period\s*([A-Za-z]+\d{1,2},\d{4})\s*-\s*([A-Za-z]+\d{1,2},\d{4})"
    )
    # Trans/post dates are also glued: "Apr30 May1 ...".
    _ROW = re.compile(
        r"^\s*([A-Z][a-z]{2})(\d{1,2})\s+[A-Z][a-z]{2}\d{1,2}\s+"
        r"(.+?)\s+(-?[\d,]+\.\d{2})\s*$"
    )
    _TOTAL = re.compile(r"Newpurchases\s*&debits\s*\$?([\d,]+\.\d{2})")

    def matches(self, text: str) -> bool:
        return "Rogers Bank" in text or "rogersbank.com" in text

    def parse(self, text: str) -> ParseResult:
        p_start, p_end = _rogers_period(text, self._PERIOD)
        rows: list[TransactionRow] = []
        for line in text.splitlines():
            m = self._ROW.match(line)
            if not m:
                continue
            mon, day, desc, amt = m.groups()
            month = _MONTHS.get(mon.lower())
            if month is None:
                continue
            desc = desc.strip()
            cents = amount_to_cents(amt)
            ttype = classify(desc, cents)
            if ttype == "payment":
                cents = -abs(cents)
            yr = resolve_year(month, int(day), p_start, p_end)
            rows.append(
                TransactionRow(
                    date=f"{yr:04d}-{month:02d}-{int(day):02d}",
                    descriptor=desc,
                    amount_cents=cents,
                    transaction_type=ttype,
                )
            )
        total = self._TOTAL.search(text)
        return ParseResult(
            rows=rows,
            printed_total_cents=amount_to_cents(total.group(1)) if total else None,
            issuer=self.name,
            period_start=p_start.isoformat(),
            period_end=p_end.isoformat(),
        )


def _parse_two_column(
    text: str,
    *,
    issuer: str,
    period_re: re.Pattern,
    row_re: re.Pattern,
    total_re: re.Pattern,
    spaced_dates: bool,
) -> ParseResult:
    """Shared body for spaced-date layouts whose period reads 'Mon D, YYYY to Mon D, YYYY'."""
    pm = period_re.search(text)
    p_start = _parse_long_date(pm.group(1)) if pm else None
    p_end = _parse_long_date(pm.group(2)) if pm else None
    if p_start is None or p_end is None:
        # Without a period we can't resolve years; refuse rather than guess.
        raise UnknownIssuerError(f"{issuer}: could not read statement period.")

    rows: list[TransactionRow] = []
    for line in text.splitlines():
        m = row_re.match(line)
        if not m:
            continue
        mon, day, desc, amt = m.groups()
        month = _MONTHS.get(mon.lower())
        if month is None:
            continue
        desc = desc.strip()
        cents = amount_to_cents(amt)
        ttype = classify(desc, cents)
        if ttype == "payment":
            cents = -abs(cents)
        yr = resolve_year(month, int(day), p_start, p_end)
        rows.append(
            TransactionRow(
                date=f"{yr:04d}-{month:02d}-{int(day):02d}",
                descriptor=desc,
                amount_cents=cents,
                transaction_type=ttype,
            )
        )
    total = total_re.search(text)
    return ParseResult(
        rows=rows,
        printed_total_cents=amount_to_cents(total.group(1)) if total else None,
        issuer=issuer,
        period_start=p_start.isoformat(),
        period_end=p_end.isoformat(),
    )


def _parse_long_date(s: str) -> date | None:
    """'April 16, 2026' -> date."""
    m = re.match(r"([A-Za-z]+) (\d{1,2}), (\d{4})", s.strip())
    if not m:
        return None
    month = _MONTHS.get(m.group(1)[:3].lower())
    if month is None:
        return None
    return date(int(m.group(3)), month, int(m.group(2)))


def _simplii_period(text: str, period_re: re.Pattern) -> tuple[date, date]:
    """Simplii prints 'April 23to May 22, 2026' - end carries the year, start borrows it."""
    m = period_re.search(text)
    if not m:
        raise UnknownIssuerError("Simplii Financial: could not read statement period.")
    end = _parse_long_date(m.group(2))
    sm = re.match(r"([A-Za-z]+) (\d{1,2})", m.group(1).strip())
    start_month = _MONTHS.get(sm.group(1)[:3].lower())
    # Start year = end year, unless the period wraps December -> January.
    start_year = end.year - 1 if start_month > end.month else end.year
    start = date(start_year, start_month, int(sm.group(2)))
    return start, end


def _rogers_period(text: str, period_re: re.Pattern) -> tuple[date, date]:
    """Rogers prints 'Apr19,2026-May18,2026' (no spaces)."""
    m = period_re.search(text)
    if not m:
        raise UnknownIssuerError("Rogers Bank: could not read statement period.")

    def parse(tok: str) -> date:
        mm = re.match(r"([A-Za-z]+)(\d{1,2}),(\d{4})", tok)
        return date(int(mm.group(3)), _MONTHS[mm.group(1)[:3].lower()], int(mm.group(2)))

    return parse(m.group(1)), parse(m.group(2))


class GenericStatementParser:
    """Issuer-agnostic heuristic parser (tier 2).

    Exploits the shape shared by almost every statement: dated line items whose
    first trailing amount is the transaction amount, plus one labeled spending
    total. It has no per-bank knowledge, so it is only trustworthy when its rows
    reconcile against the printed total -- the caller is expected to check that
    and escalate to an LLM otherwise. Descriptors may be messier than a dedicated
    parser (embedded spend-category labels, etc.), but the amounts are what
    reconcile, and reconciliation is the correctness gate.
    """

    # A date at the start of a line: "Apr 23", "Apr30", "04/23", "2026-04-23".
    _DATE = r"(?:[A-Z][a-z]{2}\.?\s?\d{1,2}|\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?|\d{4}-\d{2}-\d{2})"
    _MONEY = r"-?\(?\$?\d{1,3}(?:,\d{3})*\.\d{2}\)?"
    _LEAD = re.compile(rf"^\s*({_DATE})(?:\s+{_DATE})?\s+(.*)$")
    _MONEY_RE = re.compile(_MONEY)
    _FULL_DATE = re.compile(r"([A-Za-z]{3,9})\.?\s?(\d{1,2}),\s*(\d{4})")

    # Tried in order; first match wins. Inter-word whitespace is optional so
    # glued text like "Newpurchases &debits" still matches.
    _TOTAL_PATTERNS = [
        r"New\s*purchases?\s*&\s*debits?\s*\$?([\d,]+\.\d{2})",
        r"Total\s*charges\s*\+?\s*\$?([\d,]+\.\d{2})",
        r"Total\s*purchases\s*\$?([\d,]+\.\d{2})",
        r"New\s*purchases\s*\$?([\d,]+\.\d{2})",
        r"Total\s*this\s*period\s*\$?([\d,]+\.\d{2})",
        r"New\s*balance\s*=?\s*\$?([\d,]+\.\d{2})",
    ]

    def parse(self, pdf_bytes: bytes, *, debug: bool = False) -> ParseResult:
        return self.parse_text(layout_text(pdf_bytes), debug=debug)

    def parse_text(self, text: str, *, debug: bool = False) -> ParseResult:
        full_dates = [
            date(int(y), _MONTHS[mon[:3].lower()], int(d))
            for mon, d, y in self._FULL_DATE.findall(text)
            if mon[:3].lower() in _MONTHS
        ]
        ref_end = max(full_dates) if full_dates else None
        ref_start = min(full_dates) if full_dates else None

        rows: list[TransactionRow] = []
        for line in text.splitlines():
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

        return ParseResult(
            rows=rows,
            printed_total_cents=self._find_total(text),
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
        m = re.match(r"(\d{1,2})[/-](\d{1,2})", tok)  # assume MM/DD
        if m:
            return int(m.group(1)), int(m.group(2))
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

    def _find_total(self, text: str) -> int | None:
        for pat in self._TOTAL_PATTERNS:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                return amount_to_cents(m.group(1))
        return None


# Issuer-specific parsers disabled for now -- running generic-only to see how the
# heuristic parser holds up on its own. Re-enable by uncommenting.
_ISSUERS: list[IssuerParser] = [
    # CanadianTireParser(),
    # SimpliiParser(),
    # RogersBankParser(),
]


class DeterministicStatementParser(StatementParser):
    """Detects the issuer and dispatches to its parser. Raises on unknown."""

    def parse(self, pdf_bytes: bytes, *, debug: bool = False) -> ParseResult:
        text = layout_text(pdf_bytes)
        for issuer in _ISSUERS:
            if issuer.matches(text):
                if debug:
                    print(f"[debug] deterministic parser matched: {issuer.name}")
                return issuer.parse(text)
        raise UnknownIssuerError(
            "No deterministic parser matched this statement. "
            "Add an IssuerParser for it, or use an LLM backend."
        )


class AutoStatementParser(StatementParser):
    """Reconcile-gated escalation across three tiers:

      1. issuer-specific parser (exact, for known cards)
      2. generic heuristic parser (no per-bank code, catches many unknowns)
      3. LLM fallback (last resort)

    A tier's output is accepted only if it reconciles (rows sum to the printed
    total). Otherwise the next tier is tried. This is what makes the generic tier
    safe: a wrong guess is detected, not trusted. The LLM result is returned even
    if it doesn't reconcile -- it's the last resort, and the caller still sees the
    reconciliation verdict.
    """

    def __init__(self, fallback: StatementParser | None = None, *, use_generic: bool = True):
        self._fallback = fallback
        self._use_generic = use_generic
        self._generic = GenericStatementParser()

    def parse(self, pdf_bytes: bytes, *, debug: bool = False) -> ParseResult:
        # Deferred import avoids a module-load cycle (reconcile imports schema only).
        from .reconcile import reconcile

        text = layout_text(pdf_bytes)

        for issuer in _ISSUERS:
            if issuer.matches(text):
                try:
                    res = issuer.parse(text)
                except UnknownIssuerError as e:
                    if debug:
                        print(f"[debug] tier 1 ({issuer.name}) failed to parse: {e}; escalating.")
                    break
                if reconcile(res).passed:
                    if debug:
                        print(f"[debug] tier 1 ({issuer.name}) reconciled.")
                    return res
                if debug:
                    print(f"[debug] tier 1 ({issuer.name}) did NOT reconcile; escalating.")
                break

        if self._use_generic:
            res = self._generic.parse_text(text, debug=debug)
            if reconcile(res).passed:
                if debug:
                    print("[debug] tier 2 (generic) reconciled.")
                return res
            if debug:
                print("[debug] tier 2 (generic) did NOT reconcile; escalating.")

        # LLM fallback disabled for now -- running generic-only. Re-enable by
        # uncommenting this block.
        # if self._fallback is not None:
        #     if debug:
        #         print("[debug] tier 3 (LLM fallback).")
        #     return self._fallback.parse(pdf_bytes, debug=debug)

        raise UnknownIssuerError(
            "No deterministic tier produced a reconciling parse (LLM fallback is "
            "currently disabled)."
        )
