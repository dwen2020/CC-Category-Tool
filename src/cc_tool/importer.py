"""Statement import orchestration.

Ties the existing pieces together into the ingest path from ARCHITECTURE.md: hash the
file, skip if already imported, parse, reconcile, categorize, persist. Idempotent
by file hash, so dropping the same PDF twice never double-counts spending.
"""

from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass
from pathlib import Path

from .reconcile import reconcile
from .storage import Storage


def file_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# Serializes the whole check-parse-categorize-persist sequence below. Needed
# because the webapp's upload handler saves into the watched drop folder (so
# uploads and dropped files share one path), which means a manual upload also
# fires the folder watcher's own import almost immediately -- without this
# lock, both race to import the same file: neither sees the other's
# not-yet-committed statement row, both do the (expensive) parse + categorize
# work, and the loser's INSERT fails the file_hash UNIQUE constraint.
_IMPORT_LOCK = threading.Lock()


@dataclass
class ImportOutcome:
    file_name: str
    status: str  # imported | skipped | review | error
    n_transactions: int = 0
    reconciled: bool = False
    detail: str = ""


def import_pdf(
    path: Path, *, storage: Storage, parser, categorizer=None, on_stage=None
) -> ImportOutcome:
    """`on_stage`, if given, is called as `on_stage(stage, detail="")` at each
    step (parsing / categorizing / skipped / error / done) -- purely for
    progress reporting (see webapp.py's activity feed), never for control flow.
    """
    def stage(name: str, detail: str = "") -> None:
        if on_stage is not None:
            on_stage(name, detail)

    path = Path(path)
    with _IMPORT_LOCK:
        data = path.read_bytes()
        h = file_hash(data)

        if storage.statement_exists(h):
            stage("skipped", "already imported")
            return ImportOutcome(path.name, "skipped", detail="already imported")

        stage("parsing")
        try:
            result = parser.parse(data)
        except Exception as e:  # parsing is best-effort; report, don't crash the batch
            stage("error", f"parse failed: {e}")
            return ImportOutcome(path.name, "error", detail=f"parse failed: {e}")

        recon = reconcile(result)

        # Categorization is non-fatal: if the model backend is down, still import the
        # statement (uncategorized) so spending totals and later recategorization work.
        if categorizer is not None:
            n_purchases = sum(1 for r in result.rows if r.transaction_type == "purchase")
            stage("categorizing", f"{n_purchases} purchase row(s)")
            try:
                categorizer.categorize(result)
            except Exception as e:
                result_note = f" (categorization skipped: {e})"
            else:
                result_note = ""
        else:
            result_note = ""

        storage.add_statement(
            file_hash=h, file_name=path.name, result=result, reconciled=recon.passed
        )
        # Un-reconciled parses are stored but flagged for review rather than trusted;
        # the safety property (ARCHITECTURE.md) is "never silently accept".
        status = "imported" if recon.passed else "review"
        detail = (recon.reason + result_note).strip()
        stage("done", f"{status}: {detail}")
        return ImportOutcome(
            path.name,
            status,
            n_transactions=len(result.rows),
            reconciled=recon.passed,
            detail=detail,
        )


def import_folder(folder: Path, *, storage: Storage, parser, categorizer=None, on_stage=None) -> list[ImportOutcome]:
    return [
        import_pdf(p, storage=storage, parser=parser, categorizer=categorizer, on_stage=on_stage)
        for p in sorted(Path(folder).glob("*.pdf"))
    ]
