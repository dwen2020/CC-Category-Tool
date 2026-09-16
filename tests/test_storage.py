"""Storage's source-precedence rule (seed < model < user) is the thing that
makes a human correction permanent. If a model guess could ever clobber a
user's correction, every override in the app would be silently temporary --
these tests exist to make that regression loud instead of quiet.
"""

import pytest

from cc_tool.schema import ParseResult, TransactionRow
from cc_tool.storage import Storage


@pytest.fixture
def storage(tmp_path):
    st = Storage(tmp_path / "test.db")
    yield st
    st.close()


def _purchase_result(descriptor: str, cents: int, category=None, confidence=None) -> ParseResult:
    return ParseResult(
        rows=[
            TransactionRow(
                date="2026-01-15",
                descriptor=descriptor,
                amount_cents=cents,
                transaction_type="purchase",
                category=category,
                confidence=confidence,
            )
        ],
        printed_total_cents=cents,
    )


# --- source precedence -------------------------------------------------------


def test_user_source_beats_model_source(storage):
    cache = storage.merchant_cache()
    cache.set("starbucks", "Dining", confidence=0.9, source="model")
    cache.set("starbucks", "Shopping", confidence=1.0, source="user")
    assert cache.get("starbucks") == "Shopping"


def test_model_source_cannot_overwrite_user_source(storage):
    """The core invariant: once a human corrects a merchant, no later model
    guess -- however confident -- may silently replace it."""
    cache = storage.merchant_cache()
    cache.set("starbucks", "Shopping", confidence=1.0, source="user")
    cache.set("starbucks", "Dining", confidence=0.99, source="model")
    assert cache.get("starbucks") == "Shopping"


def test_model_source_can_overwrite_seed_source(storage):
    cache = storage.merchant_cache()
    cache.set("starbucks", "Other", confidence=None, source="seed")
    cache.set("starbucks", "Dining", confidence=0.8, source="model")
    assert cache.get("starbucks") == "Dining"


def test_same_source_updates_normally(storage):
    cache = storage.merchant_cache()
    cache.set("starbucks", "Dining", confidence=0.5, source="model")
    cache.set("starbucks", "Entertainment", confidence=0.9, source="model")
    assert cache.get("starbucks") == "Entertainment"


# --- add_statement / reporting -----------------------------------------------


def test_add_statement_persists_rows_and_counts(storage):
    result = _purchase_result("TIM HORTONS", 1234, category="Dining", confidence=0.95)
    storage.add_statement(file_hash="abc123", file_name="a.pdf", result=result, reconciled=True)
    assert storage.statement_count() == 1
    assert storage.transaction_count() == 1


def test_statement_exists_is_hash_based(storage):
    result = _purchase_result("TIM HORTONS", 1234)
    storage.add_statement(file_hash="abc123", file_name="a.pdf", result=result, reconciled=True)
    assert storage.statement_exists("abc123")
    assert not storage.statement_exists("does-not-exist")


def test_category_totals_excludes_non_purchase_rows(storage):
    result = ParseResult(
        rows=[
            TransactionRow(
                date="2026-01-15", descriptor="STORE", amount_cents=1000,
                transaction_type="purchase", category="Shopping",
            ),
            TransactionRow(
                date="2026-01-16", descriptor="PAYMENT", amount_cents=-5000,
                transaction_type="payment",
            ),
        ],
        printed_total_cents=1000,
    )
    storage.add_statement(file_hash="h1", file_name="a.pdf", result=result, reconciled=True)
    totals = storage.category_totals_by_month()
    assert len(totals) == 1
    assert totals[0]["category"] == "Shopping"
    assert totals[0]["total_cents"] == 1000


def test_apply_category_updates_existing_transactions_and_cache(storage):
    result = _purchase_result("UNKNOWN MERCHANT", 500)
    storage.add_statement(file_hash="h1", file_name="a.pdf", result=result, reconciled=True)

    n = storage.apply_category("UNKNOWN MERCHANT", "Shopping")
    assert n == 1
    merchants = storage.all_merchants()
    assert merchants[0]["category"] == "Shopping"
    assert merchants[0]["source"] == "user"


def test_apply_category_rejects_invalid_category(storage):
    with pytest.raises(ValueError):
        storage.apply_category("some merchant", "NotARealCategory")


def test_delete_statement_cascades_transactions(storage):
    result = _purchase_result("STORE", 1000)
    statement_id = storage.add_statement(
        file_hash="h1", file_name="a.pdf", result=result, reconciled=True
    )
    assert storage.transaction_count() == 1
    assert storage.delete_statement(statement_id) is True
    assert storage.transaction_count() == 0
    assert storage.delete_statement(statement_id) is False
