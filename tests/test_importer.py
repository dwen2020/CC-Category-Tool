"""Import orchestration: hash dedup (never double-count the same statement)
and non-fatal degradation (a broken categorizer or parser must not lose
already-extracted data, since categorization commonly fails when the model
isn't installed -- see ARCHITECTURE.md's model-artifact setup step)."""

import pytest

from cc_tool.importer import file_hash, import_pdf
from cc_tool.schema import ParseResult, TransactionRow
from cc_tool.storage import Storage


class _StubParser:
    """Always returns the same two-row, reconciling result."""

    def parse(self, data: bytes):
        return ParseResult(
            rows=[
                TransactionRow(
                    date="2026-01-15", descriptor="STORE", amount_cents=1000,
                    transaction_type="purchase",
                ),
            ],
            printed_total_cents=1000,
        )


class _BrokenParser:
    def parse(self, data: bytes):
        raise ValueError("simulated parse failure")


class _BrokenCategorizer:
    def categorize(self, result, debug=False):
        raise RuntimeError("simulated categorizer failure (e.g. model not found)")


@pytest.fixture
def storage(tmp_path):
    st = Storage(tmp_path / "test.db")
    yield st
    st.close()


def test_file_hash_is_deterministic():
    assert file_hash(b"same bytes") == file_hash(b"same bytes")
    assert file_hash(b"one") != file_hash(b"other")


def test_reimporting_same_bytes_is_skipped(tmp_path, storage):
    pdf_path = tmp_path / "statement.pdf"
    pdf_path.write_bytes(b"fake pdf bytes")

    first = import_pdf(pdf_path, storage=storage, parser=_StubParser())
    second = import_pdf(pdf_path, storage=storage, parser=_StubParser())

    assert first.status == "imported"
    assert second.status == "skipped"
    assert storage.statement_count() == 1  # not double-counted


def test_parse_failure_is_reported_not_raised(tmp_path, storage):
    pdf_path = tmp_path / "statement.pdf"
    pdf_path.write_bytes(b"fake pdf bytes")

    outcome = import_pdf(pdf_path, storage=storage, parser=_BrokenParser())

    assert outcome.status == "error"
    assert storage.statement_count() == 0


def test_categorizer_failure_still_imports_uncategorized(tmp_path, storage):
    """Categorization commonly fails on a fresh install (model not yet copied
    into place); the statement's transactions must still be stored so spending
    totals and later re-categorization work."""
    pdf_path = tmp_path / "statement.pdf"
    pdf_path.write_bytes(b"fake pdf bytes")

    outcome = import_pdf(
        pdf_path, storage=storage, parser=_StubParser(), categorizer=_BrokenCategorizer()
    )

    assert outcome.status == "imported"
    assert storage.transaction_count() == 1
    assert "categorization skipped" in outcome.detail


def test_unreconciled_statement_is_still_stored_but_flagged():
    class _MismatchedParser:
        def parse(self, data: bytes):
            return ParseResult(
                rows=[
                    TransactionRow(
                        date="2026-01-15", descriptor="STORE", amount_cents=1000,
                        transaction_type="purchase",
                    ),
                ],
                printed_total_cents=9999,  # doesn't match rows_sum
            )

    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        pdf_path = Path(d) / "statement.pdf"
        pdf_path.write_bytes(b"fake pdf bytes")
        storage = Storage(Path(d) / "test.db")
        try:
            outcome = import_pdf(pdf_path, storage=storage, parser=_MismatchedParser())
            assert outcome.status == "review"
            assert storage.transaction_count() == 1  # stored despite not reconciling
        finally:
            storage.close()
