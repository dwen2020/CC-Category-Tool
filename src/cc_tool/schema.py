"""Pydantic models for parser output. Field descriptions are sent to the model
as part of the structured-output schema, so phrasing matters."""

from typing import Literal, Optional

from pydantic import BaseModel, Field


class TransactionRow(BaseModel):
    date: str = Field(description="Transaction date in ISO format YYYY-MM-DD.")
    descriptor: str = Field(
        description="Merchant descriptor exactly as printed on the statement, "
        "including any store numbers, city, or state suffixes."
    )
    amount_cents: int = Field(
        description="Signed integer cents. Charges, purchases, fees, and interest are POSITIVE. "
        "Credits, refunds, and payments received are NEGATIVE. "
        "Example: $12.34 charge = 1234, $50.00 refund = -5000."
    )
    transaction_type: Literal["purchase", "payment", "refund", "fee", "interest"] = Field(
        description=(
            "Type of transaction. Use 'purchase' for merchant charges. "
            "Use 'payment' for balance payments received from the cardholder. "
            "Use 'refund' for merchant credits or chargebacks. "
            "Use 'fee' for bank-assessed fees (annual fee, foreign transaction fee, etc.). "
            "Use 'interest' for interest charges."
        )
    )


class ParseResult(BaseModel):
    rows: list[TransactionRow] = Field(
        description="Every line-item transaction on the statement, in the order printed."
    )
    printed_total_cents: Optional[int] = Field(
        default=None,
        description="Total purchases/charges as printed on the statement (often labeled "
        "'Total Purchases', 'New Charges', or 'Total this period'). "
        "This should be the spending total only — do NOT include payments received. "
        "Integer cents. Null if the statement does not show one.",
    )
    issuer: Optional[str] = Field(
        default=None,
        description="Bank or card issuer name as printed on the statement.",
    )
    period_start: Optional[str] = Field(
        default=None,
        description="Start of statement period in ISO format YYYY-MM-DD.",
    )
    period_end: Optional[str] = Field(
        default=None,
        description="End of statement period in ISO format YYYY-MM-DD.",
    )
