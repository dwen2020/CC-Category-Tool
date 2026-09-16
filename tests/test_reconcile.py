"""reconcile.py must compare purchase-only spend to the printed total.

Regression coverage for a real bug found in production data: refunds and
payments were being netted into the comparison, so any statement with a
refund failed reconciliation even though the parser had extracted every
transaction correctly (see git history / ARCHITECTURE.md for the incident).
"""

from cc_tool.reconcile import TOLERANCE_CENTS, reconcile
from cc_tool.schema import ParseResult, TransactionRow


def _row(amount_cents: int, ttype: str, descriptor: str = "MERCHANT") -> TransactionRow:
    return TransactionRow(
        date="2026-01-15",
        descriptor=descriptor,
        amount_cents=amount_cents,
        transaction_type=ttype,
    )


def test_passes_when_purchases_match_printed_total():
    result = ParseResult(
        rows=[_row(1000, "purchase"), _row(2000, "purchase")],
        printed_total_cents=3000,
    )
    recon = reconcile(result)
    assert recon.passed
    assert recon.rows_sum_cents == 3000
    assert recon.delta_cents == 0


def test_refunds_are_excluded_from_the_comparison():
    """The printed 'Purchases' total never nets out refunds -- neither should we.

    Regression test: previously summed everything except `payment` rows, which
    dragged the total below the printed purchases figure on any statement with
    a refund and incorrectly flagged it for review.
    """
    result = ParseResult(
        rows=[
            _row(10000, "purchase"),
            _row(-3000, "refund"),  # would have wrongly reduced rows_sum before the fix
        ],
        printed_total_cents=10000,
    )
    recon = reconcile(result)
    assert recon.passed
    assert recon.rows_sum_cents == 10000


def test_payments_are_excluded_from_the_comparison():
    result = ParseResult(
        rows=[
            _row(5000, "purchase"),
            _row(-20000, "payment"),
        ],
        printed_total_cents=5000,
    )
    recon = reconcile(result)
    assert recon.passed
    assert recon.rows_sum_cents == 5000


def test_within_tolerance_still_passes():
    result = ParseResult(
        rows=[_row(1000 + TOLERANCE_CENTS, "purchase")],
        printed_total_cents=1000,
    )
    assert reconcile(result).passed


def test_beyond_tolerance_fails_with_delta():
    result = ParseResult(
        rows=[_row(1050, "purchase")],
        printed_total_cents=1000,
    )
    recon = reconcile(result)
    assert not recon.passed
    assert recon.delta_cents == 50


def test_missing_printed_total_fails_for_review_not_silently():
    result = ParseResult(rows=[_row(1000, "purchase")], printed_total_cents=None)
    recon = reconcile(result)
    assert not recon.passed
    assert recon.printed_total_cents is None
    assert "No printed total" in recon.reason
