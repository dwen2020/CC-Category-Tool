"""Reconciliation: verify spending transactions sum to the statement's printed
purchases total. Payments are excluded — they represent prior-month balance
repayment, not current spending."""

from dataclasses import dataclass
from typing import Optional

from .schema import ParseResult


TOLERANCE_CENTS = 1


@dataclass
class ReconciliationResult:
    passed: bool
    rows_sum_cents: int
    printed_total_cents: Optional[int]
    delta_cents: Optional[int]
    reason: str


def reconcile(parse: ParseResult) -> ReconciliationResult:
    spending_rows = [r for r in parse.rows if r.transaction_type != "payment"]
    rows_sum = sum(r.amount_cents for r in spending_rows)

    if parse.printed_total_cents is None:
        return ReconciliationResult(
            passed=False,
            rows_sum_cents=rows_sum,
            printed_total_cents=None,
            delta_cents=None,
            reason="No printed total available. Manual review required.",
        )

    delta = rows_sum - parse.printed_total_cents
    if abs(delta) <= TOLERANCE_CENTS:
        return ReconciliationResult(
            passed=True,
            rows_sum_cents=rows_sum,
            printed_total_cents=parse.printed_total_cents,
            delta_cents=delta,
            reason="Rows sum matches printed total within tolerance.",
        )
    return ReconciliationResult(
        passed=False,
        rows_sum_cents=rows_sum,
        printed_total_cents=parse.printed_total_cents,
        delta_cents=delta,
        reason=(
            f"Rows sum ({rows_sum} cents) differs from printed total "
            f"({parse.printed_total_cents} cents) by {delta} cents."
        ),
    )
