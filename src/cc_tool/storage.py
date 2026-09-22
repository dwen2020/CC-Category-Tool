"""Local SQLite persistence: statements, transactions, and the learned
merchant->category mapping.

This is the "Local Storage" component from ARCHITECTURE.md -- the thing that
turns one-shot parsing into an accumulating record you can report on over time.
Single file, stdlib sqlite3, no server, no ORM: it matches the "runs entirely on
the user's machine" constraint and adds no dependency.

Key invariants from the design:
  - Each PDF is imported at most once, keyed by its content hash (statements.file_hash).
  - The learned cache is authoritative for user assignments; a user category always
    wins over a model-derived one and is never silently overwritten.
"""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from .schema import ParseResult, TransactionRow


def default_db_path() -> Path:
    """DB location. Honors CC_TOOL_DB, else ~/.cc_tool/cc_tool.db."""
    env = os.environ.get("CC_TOOL_DB")
    if env:
        return Path(env)
    return Path.home() / ".cc_tool" / "cc_tool.db"


_SCHEMA = """
CREATE TABLE IF NOT EXISTS statements (
    id                  INTEGER PRIMARY KEY,
    file_hash           TEXT UNIQUE NOT NULL,
    file_name           TEXT,
    issuer              TEXT,
    period_start        TEXT,
    period_end          TEXT,
    printed_total_cents INTEGER,
    reconciled          INTEGER NOT NULL DEFAULT 0,
    imported_at         TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS transactions (
    id                  INTEGER PRIMARY KEY,
    statement_id        INTEGER NOT NULL REFERENCES statements(id) ON DELETE CASCADE,
    date                TEXT NOT NULL,
    descriptor          TEXT NOT NULL,
    normalized_merchant TEXT NOT NULL,
    amount_cents        INTEGER NOT NULL,
    transaction_type    TEXT NOT NULL,
    category            TEXT,
    category_source     TEXT,
    confidence          REAL
);

CREATE INDEX IF NOT EXISTS idx_tx_statement ON transactions(statement_id);
CREATE INDEX IF NOT EXISTS idx_tx_date      ON transactions(date);
CREATE INDEX IF NOT EXISTS idx_tx_category  ON transactions(category);

-- Learned merchant -> category. Holds both user assignments (authoritative) and
-- model-derived guesses; `source` distinguishes them and user always wins.
CREATE TABLE IF NOT EXISTS learned_categories (
    normalized_merchant TEXT PRIMARY KEY,
    category            TEXT NOT NULL,
    source              TEXT NOT NULL,
    confidence          REAL,
    updated_at          TEXT NOT NULL
);

-- Generic small key/value store. Currently just tracks the retrain-nudge
-- baseline (see retrain_status()): how many user corrections existed as of
-- the last time you retrained the model, so the dashboard can show how many
-- new ones have piled up since.
CREATE TABLE IF NOT EXISTS model_state (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""

# Source precedence: a lower-priority source must never overwrite a higher one.
_SOURCE_RANK = {"seed": 0, "model": 1, "user": 2}

# Reference point for how much to trust a model categorization: below this
# confidence, the model is meaningfully more likely to be wrong (see
# tests/calibrate_confidence.py). Not used to gate any UI feature -- every
# merchant is always browsable/correctable via all_merchants(), regardless of
# confidence -- this is purely a diagnostic constant for calibration checks.
DEFAULT_REVIEW_THRESHOLD = 0.4

# Nudge point for the "corrections since last retrain" indicator: below this,
# a retrain wouldn't move a model fine-tuned on ~90K rows; see project memory
# on retrain volume. Not a hard gate -- retraining stays a deliberate,
# validated action, this just flags when it's *worth considering*.
DEFAULT_RETRAIN_THRESHOLD = 150

_MIGRATION_COLUMNS = {
    "transactions": [("confidence", "REAL")],
    "learned_categories": [("confidence", "REAL")],
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Storage:
    def __init__(self, path: Path | None = None):
        self._path = path or default_db_path()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        # WAL lets the watcher thread write while web requests read concurrently.
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.executescript(_SCHEMA)
        self._migrate()
        self._conn.commit()

    def _migrate(self) -> None:
        """Add columns introduced after a table already existed on disk, so an
        existing local DB file isn't broken by a newer schema version."""
        for table, columns in _MIGRATION_COLUMNS.items():
            existing = {
                row["name"] for row in self._conn.execute(f"PRAGMA table_info({table})")
            }
            for name, coltype in columns:
                if name not in existing:
                    self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {coltype}")

    def close(self) -> None:
        self._conn.close()

    # --- statements -----------------------------------------------------------

    def statement_exists(self, file_hash: str) -> bool:
        cur = self._conn.execute(
            "SELECT 1 FROM statements WHERE file_hash = ?", (file_hash,)
        )
        return cur.fetchone() is not None

    def add_statement(
        self,
        *,
        file_hash: str,
        file_name: str,
        result: ParseResult,
        reconciled: bool,
    ) -> int:
        cur = self._conn.execute(
            """INSERT INTO statements
               (file_hash, file_name, issuer, period_start, period_end,
                printed_total_cents, reconciled, imported_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                file_hash,
                file_name,
                result.issuer,
                result.period_start,
                result.period_end,
                result.printed_total_cents,
                1 if reconciled else 0,
                _now(),
            ),
        )
        statement_id = cur.lastrowid
        rows = []
        for row in result.rows:
            key = _norm(row.descriptor)
            source = None
            if row.category is not None:
                found = self._conn.execute(
                    "SELECT source FROM learned_categories WHERE normalized_merchant = ?",
                    (key,),
                ).fetchone()
                source = found["source"] if found else None
            rows.append(
                (
                    statement_id,
                    row.date,
                    row.descriptor,
                    key,
                    row.amount_cents,
                    row.transaction_type,
                    row.category,
                    source,
                    row.confidence,
                )
            )
        self._conn.executemany(
            """INSERT INTO transactions
               (statement_id, date, descriptor, normalized_merchant,
                amount_cents, transaction_type, category, category_source, confidence)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )
        self._conn.commit()
        return statement_id

    def list_statements(self) -> list[dict]:
        """Every imported statement, most recent first, with its transaction
        count -- lets a human see (and, via delete_statement, correct) a bad
        import instead of that state being invisible outside the DB file."""
        rows = self._conn.execute(
            """SELECT s.id, s.file_name, s.issuer, s.period_start, s.period_end,
                      s.printed_total_cents, s.reconciled, s.imported_at,
                      COUNT(t.id) AS n_transactions
               FROM statements s
               LEFT JOIN transactions t ON t.statement_id = s.id
               GROUP BY s.id
               ORDER BY s.imported_at DESC"""
        )
        return [dict(r) for r in rows]

    def delete_statement(self, statement_id: int) -> bool:
        """Delete a statement and its transactions (foreign_keys=ON cascades
        the delete). Returns False if no such statement exists. Does not
        touch learned_categories -- a merchant's learned category outlives
        any one statement that happened to first teach it."""
        cur = self._conn.execute("DELETE FROM statements WHERE id = ?", (statement_id,))
        self._conn.commit()
        return cur.rowcount > 0

    # --- reporting ------------------------------------------------------------

    def category_totals_by_month(
        self, *, start: str | None = None, end: str | None = None
    ) -> list[dict]:
        """Purchase spending per (month, category). Payments/refunds excluded."""
        q = [
            "SELECT substr(date, 1, 7) AS month,",
            "       COALESCE(category, 'Uncategorized') AS category,",
            "       SUM(amount_cents) AS total_cents,",
            "       COUNT(*) AS n",
            "FROM transactions",
            "WHERE transaction_type = 'purchase'",
        ]
        params: list[str] = []
        if start:
            q.append("AND date >= ?")
            params.append(start)
        if end:
            q.append("AND date <= ?")
            params.append(end)
        q.append("GROUP BY month, category ORDER BY month, total_cents DESC")
        return [dict(r) for r in self._conn.execute(" ".join(q), params)]

    def merchants_for_month_category(self, *, month: str, category: str) -> list[dict]:
        """Drill-down for one (month, category) cell of category_totals_by_month:
        every merchant contributing to it, biggest spend first. `category` may
        be the literal string 'Uncategorized' to match NULL, mirroring how
        category_totals_by_month reports uncategorized spend."""
        want_null = category == "Uncategorized"
        rows = self._conn.execute(
            f"""SELECT normalized_merchant AS merchant,
                       MAX(descriptor)     AS sample,
                       COUNT(*)            AS n,
                       SUM(amount_cents)   AS total_cents
                FROM transactions
                WHERE transaction_type = 'purchase'
                  AND substr(date, 1, 7) = ?
                  AND {"category IS NULL" if want_null else "category = ?"}
                GROUP BY normalized_merchant
                ORDER BY total_cents DESC""",
            (month,) if want_null else (month, category),
        )
        return [dict(r) for r in rows]

    def all_merchants(self) -> list[dict]:
        """Every purchase merchant ever seen, regardless of confidence -- the
        full browsable list, biggest spend first. Nothing is filtered out;
        this is the sole review/correction surface -- a human scans and
        corrects anything, not just what the model flagged as unsure."""
        rows = self._conn.execute(
            """SELECT t.normalized_merchant   AS merchant,
                      MAX(t.descriptor)       AS sample,
                      COUNT(*)                AS n,
                      SUM(t.amount_cents)     AS total_cents,
                      MAX(t.category)         AS category,
                      lc.confidence           AS confidence,
                      lc.source               AS source
               FROM transactions t
               LEFT JOIN learned_categories lc
                      ON lc.normalized_merchant = t.normalized_merchant
               WHERE t.transaction_type = 'purchase'
               GROUP BY t.normalized_merchant
               ORDER BY total_cents DESC"""
        )
        return [dict(r) for r in rows]

    # --- retrain nudge ----------------------------------------------------

    def _model_state_get(self, key: str) -> str | None:
        row = self._conn.execute(
            "SELECT value FROM model_state WHERE key = ?", (key,)
        ).fetchone()
        return row["value"] if row else None

    def user_correction_count(self) -> int:
        """How many merchants a human has ever pinned (source='user'), ever --
        the running total the retrain-nudge baseline is measured against."""
        return self._conn.execute(
            "SELECT COUNT(*) FROM learned_categories WHERE source = 'user'"
        ).fetchone()[0]

    def retrain_status(self) -> dict:
        """Corrections accumulated since the last time you marked the model
        retrained. `marked_at` is None if you've never marked one -- in that
        case the baseline is 0, so this just reports the all-time total."""
        baseline = int(self._model_state_get("retrain_baseline") or 0)
        marked_at = self._model_state_get("retrain_marked_at")
        total = self.user_correction_count()
        return {
            "corrections_since": max(0, total - baseline),
            "marked_at": marked_at,
            "threshold": DEFAULT_RETRAIN_THRESHOLD,
        }

    def mark_retrained(self) -> dict:
        """Reset the baseline to the current correction count -- call this
        after actually retraining and validating a new model checkpoint."""
        now = _now()
        baseline = self.user_correction_count()
        self._conn.execute(
            """INSERT INTO model_state (key, value) VALUES ('retrain_baseline', ?)
               ON CONFLICT(key) DO UPDATE SET value = excluded.value""",
            (str(baseline),),
        )
        self._conn.execute(
            """INSERT INTO model_state (key, value) VALUES ('retrain_marked_at', ?)
               ON CONFLICT(key) DO UPDATE SET value = excluded.value""",
            (now,),
        )
        self._conn.commit()
        return self.retrain_status()

    def apply_category(self, normalized_merchant: str, category: str) -> int:
        """User assignment: pin the merchant in the learned cache (source=user,
        wins forever) and update every existing transaction for it. Returns the
        number of transactions updated."""
        from .categories import CATEGORY_SET, CATEGORIES

        if category not in CATEGORY_SET:
            raise ValueError(
                f"'{category}' is not a valid category. Choose from: {', '.join(CATEGORIES)}"
            )
        self.merchant_cache().set(normalized_merchant, category, confidence=1.0, source="user")
        cur = self._conn.execute(
            """UPDATE transactions SET category = ?, category_source = 'user', confidence = 1.0
               WHERE normalized_merchant = ?""",
            (category, normalized_merchant),
        )
        self._conn.commit()
        return cur.rowcount

    def apply_category_like(self, text: str, category: str) -> tuple[list[str], int]:
        """CLI-friendly override: match any stored merchant whose normalized text
        CONTAINS `text` (so 'PUBLIC MOBILE' matches 'PUBLIC MOBILE SELF-SER ...'),
        pin each as source=user, and update their transactions. Returns
        (matched merchants, transactions updated)."""
        from .categories import CATEGORY_SET, CATEGORIES

        if category not in CATEGORY_SET:
            raise ValueError(
                f"'{category}' is not a valid category. Choose from: {', '.join(CATEGORIES)}"
            )
        key = _norm(text)
        matches = [
            r["normalized_merchant"]
            for r in self._conn.execute(
                "SELECT DISTINCT normalized_merchant FROM transactions "
                "WHERE normalized_merchant LIKE ?",
                (f"%{key}%",),
            )
        ]
        if not matches:
            # Nothing stored yet -- still record the exact key so a future import
            # of the same descriptor resolves automatically.
            self.merchant_cache().set(key, category, confidence=1.0, source="user")
            self._conn.commit()
            return [], 0
        n = 0
        for m in matches:
            self.merchant_cache().set(m, category, confidence=1.0, source="user")
            n += self._conn.execute(
                "UPDATE transactions SET category = ?, category_source = 'user', confidence = 1.0 "
                "WHERE normalized_merchant = ?",
                (category, m),
            ).rowcount
        self._conn.commit()
        return matches, n

    def statement_count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM statements").fetchone()[0]

    def transaction_count(self) -> int:
        return self._conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]

    # --- learned cache access -------------------------------------------------

    def merchant_cache(self) -> "DbMerchantCache":
        return DbMerchantCache(self._conn)


def _norm(descriptor: str) -> str:
    # Local import keeps storage importable without the normalize module at type-check.
    from .normalize import normalize_merchant

    return normalize_merchant(descriptor)


class DbMerchantCache:
    """SQLite-backed learned cache, duck-compatible with the categorizer's cache
    interface (get / set / set_override / save). Enforces source precedence so a
    model guess never clobbers a user (or seed) assignment."""

    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn

    def get(self, merchant_key: str) -> str | None:
        row = self._conn.execute(
            "SELECT category FROM learned_categories WHERE normalized_merchant = ?",
            (merchant_key,),
        ).fetchone()
        return row["category"] if row else None

    def get_confidence(self, merchant_key: str) -> float | None:
        row = self._conn.execute(
            "SELECT confidence FROM learned_categories WHERE normalized_merchant = ?",
            (merchant_key,),
        ).fetchone()
        return row["confidence"] if row else None

    def set(
        self, merchant_key: str, category: str, *, confidence: float | None = None, source: str
    ) -> None:
        existing = self._conn.execute(
            "SELECT source FROM learned_categories WHERE normalized_merchant = ?",
            (merchant_key,),
        ).fetchone()
        if existing and _SOURCE_RANK.get(source, 0) < _SOURCE_RANK.get(existing["source"], 0):
            return  # never downgrade a higher-priority assignment
        self._conn.execute(
            """INSERT INTO learned_categories (normalized_merchant, category, source, confidence, updated_at)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(normalized_merchant) DO UPDATE SET
                   category = excluded.category,
                   source = excluded.source,
                   confidence = excluded.confidence,
                   updated_at = excluded.updated_at""",
            (merchant_key, category, source, confidence, _now()),
        )

    def set_override(self, raw_merchant: str, category: str) -> None:
        from .categories import CATEGORY_SET, CATEGORIES

        if category not in CATEGORY_SET:
            raise ValueError(
                f"'{category}' is not a valid category. Choose from: {', '.join(CATEGORIES)}"
            )
        self.set(_norm(raw_merchant), category, confidence=1.0, source="user")
        self._conn.commit()

    def save(self) -> None:
        self._conn.commit()
