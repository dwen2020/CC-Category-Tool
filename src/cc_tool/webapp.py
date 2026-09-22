"""Local web app: dashboard, a watched drop folder, and the merchant browser.

This is the "Web UI + Backend" from ARCHITECTURE.md. Everything runs on localhost against
the local SQLite DB. Drop a PDF into the watched folder (or use the upload button)
and it is hashed, parsed, reconciled, categorized, and stored automatically; the
dashboard shows category totals over time; the "All merchants" browser lets you
pin a category for any merchant at all, not just ones the model flagged as unsure.

Threading note: the folder watcher runs in its own thread and each web request is
short-lived, so every operation opens its own Storage (its own SQLite connection)
against the shared WAL-mode DB file. Connections are never shared across threads.
"""

import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .categories import CATEGORIES
from .categorizer import Categorizer, DistilBertCategorizer
from .deterministic import GenericStatementParser
from .importer import ImportOutcome, import_pdf
from .storage import Storage

_STATIC = Path(__file__).parent / "static"

# In-memory, most-recent-first log of import stage transitions (detected /
# parsing / categorizing / done / skipped / error), so the dashboard can show
# the drop-folder trigger and the model's progress as separate visible steps
# instead of one opaque wait. Deliberately not persisted -- it's a live status
# feed, not history; a server restart clearing it is fine.
_ACTIVITY_LOCK = threading.Lock()
_ACTIVITY: list[dict] = []
_ACTIVITY_MAX = 50


def _record_activity(file_name: str, stage: str, detail: str = "") -> None:
    with _ACTIVITY_LOCK:
        _ACTIVITY.append(
            {
                "file": file_name,
                "stage": stage,
                "detail": detail,
                "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
        )
        del _ACTIVITY[:-_ACTIVITY_MAX]


def _make_categorizer(storage: Storage) -> Categorizer:
    return Categorizer(DistilBertCategorizer(), storage.merchant_cache())


def create_app(
    *,
    db_path: Path,
    drop_folder: Path,
):
    from fastapi import FastAPI, File, HTTPException, UploadFile
    from fastapi.responses import FileResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles

    drop_folder = Path(drop_folder)
    drop_folder.mkdir(parents=True, exist_ok=True)
    parser = GenericStatementParser()

    def do_import(path: Path) -> ImportOutcome:
        """Import one PDF with a fresh Storage (thread-safe for the watcher)."""
        storage = Storage(db_path)
        try:
            cat = None
            try:
                cat = _make_categorizer(storage)
            except Exception as e:
                cat = None  # model unavailable -> import uncategorized
                _record_activity(path.name, "warning", f"categorization unavailable: {e}")
            return import_pdf(
                path,
                storage=storage,
                parser=parser,
                categorizer=cat,
                on_stage=lambda stage, detail="": _record_activity(path.name, stage, detail),
            )
        finally:
            storage.close()

    app = FastAPI(title="CC Category Tool")

    # --- watched drop folder --------------------------------------------------

    observer = {"obj": None}

    def _import_and_log(p: Path) -> None:
        try:
            outcome = do_import(p)
            print(f"[watch] {outcome.file_name}: {outcome.status} ({outcome.detail})")
        except Exception as e:
            _record_activity(p.name, "error", str(e))
            print(f"[watch] error importing {p.name}: {e}")

    @app.on_event("startup")
    def _start_watcher() -> None:
        from watchdog.events import FileSystemEventHandler
        from watchdog.observers import Observer

        class Handler(FileSystemEventHandler):
            def on_created(self, event):
                if event.is_directory:
                    return
                p = Path(event.src_path)
                if p.suffix.lower() != ".pdf":
                    return
                _record_activity(p.name, "detected", "dropped in watched folder")
                _wait_until_stable(p)
                _import_and_log(p)

        obs = Observer()
        obs.schedule(Handler(), str(drop_folder), recursive=False)
        obs.daemon = True
        obs.start()
        observer["obj"] = obs
        print(f"[watch] watching drop folder: {drop_folder}")

        # on_created only fires for files created AFTER the watcher starts --
        # anything already sitting in the folder from before this run would
        # otherwise be silently ignored forever. Import that backlog now, off
        # the startup path so a large backlog doesn't delay the dashboard
        # becoming responsive.
        def _import_backlog() -> None:
            for p in sorted(drop_folder.glob("*.pdf")):
                _record_activity(p.name, "detected", "already in drop folder at startup")
                _import_and_log(p)

        threading.Thread(target=_import_backlog, daemon=True).start()

    @app.on_event("shutdown")
    def _stop_watcher() -> None:
        if observer["obj"]:
            observer["obj"].stop()
            observer["obj"].join(timeout=2)

    # --- API ------------------------------------------------------------------

    @app.get("/api/summary")
    def summary():
        st = Storage(db_path)
        try:
            return {
                "statements": st.statement_count(),
                "transactions": st.transaction_count(),
                "drop_folder": str(drop_folder),
                "categories": CATEGORIES,
            }
        finally:
            st.close()

    @app.get("/api/totals")
    def totals(start: Optional[str] = None, end: Optional[str] = None):
        st = Storage(db_path)
        try:
            return {"rows": st.category_totals_by_month(start=start, end=end)}
        finally:
            st.close()

    @app.get("/api/totals/drilldown")
    def totals_drilldown(month: str, category: str):
        st = Storage(db_path)
        try:
            return {"rows": st.merchants_for_month_category(month=month, category=category)}
        finally:
            st.close()

    @app.get("/api/merchants")
    def merchants():
        st = Storage(db_path)
        try:
            return {"rows": st.all_merchants()}
        finally:
            st.close()

    @app.get("/api/retrain_status")
    def retrain_status():
        st = Storage(db_path)
        try:
            return st.retrain_status()
        finally:
            st.close()

    @app.post("/api/retrain_status/mark")
    def mark_retrained():
        st = Storage(db_path)
        try:
            return st.mark_retrained()
        finally:
            st.close()

    @app.get("/api/statements")
    def statements():
        st = Storage(db_path)
        try:
            return {"rows": st.list_statements()}
        finally:
            st.close()

    @app.delete("/api/statements/{statement_id}")
    def delete_statement(statement_id: int):
        st = Storage(db_path)
        try:
            if not st.delete_statement(statement_id):
                raise HTTPException(404, "Statement not found")
            return {"deleted": statement_id}
        finally:
            st.close()

    @app.get("/api/activity")
    def activity():
        with _ACTIVITY_LOCK:
            return {"rows": list(reversed(_ACTIVITY))}

    @app.post("/api/categorize")
    def categorize(payload: dict):
        merchant = (payload or {}).get("merchant")
        category = (payload or {}).get("category")
        if not merchant or not category:
            raise HTTPException(400, "merchant and category are required")
        st = Storage(db_path)
        try:
            n = st.apply_category(merchant, category)
            return {"updated": n}
        except ValueError as e:
            raise HTTPException(400, str(e))
        finally:
            st.close()

    @app.post("/api/upload")
    async def upload(file: UploadFile = File(...)):
        # Save into the drop folder so uploads and dropped files share one path;
        # import synchronously here so the caller gets the outcome. This also
        # fires the watcher's own on_created for the same file -- import_pdf's
        # lock (see importer.py) makes that safe, and the watcher will record
        # its own "detected" for the same reason: it really was detected twice.
        dest = drop_folder / file.filename
        dest.write_bytes(await file.read())
        _record_activity(file.filename, "detected", "uploaded")
        outcome = do_import(dest)
        return JSONResponse(outcome.__dict__)

    @app.get("/")
    def index():
        return FileResponse(_STATIC / "index.html")

    app.mount("/static", StaticFiles(directory=_STATIC), name="static")
    return app


def _wait_until_stable(path: Path, tries: int = 20, interval: float = 0.15) -> None:
    """Wait for a freshly-dropped file to finish copying (size stops changing)."""
    last = -1
    for _ in range(tries):
        try:
            size = path.stat().st_size
        except OSError:
            size = -1
        if size == last and size >= 0:
            return
        last = size
        time.sleep(interval)


def serve(
    *,
    db_path: Path,
    drop_folder: Path,
    host: str = "127.0.0.1",
    port: int = 8765,
) -> None:
    import uvicorn

    app = create_app(db_path=db_path, drop_folder=drop_folder)
    print(f"Dashboard: http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="warning")
