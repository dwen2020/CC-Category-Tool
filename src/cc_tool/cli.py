"""Command-line entry. Parses a PDF and prints extracted transactions + the
reconciliation verdict."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

from .categorizer import Categorizer, DistilBertCategorizer
from .deterministic import GenericStatementParser
from .normalize import normalize_merchant
from .reconcile import reconcile


def main(argv: list[str] | None = None) -> int:
    load_dotenv()

    p = argparse.ArgumentParser(
        prog="cc-tool",
        description="Parse a credit-card statement PDF.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    parse_cmd = sub.add_parser(
        "parse", help="Parse a PDF and print extracted transactions."
    )
    parse_cmd.add_argument("pdf", type=Path, help="Path to the statement PDF.")
    parse_cmd.add_argument(
        "--categorize",
        action="store_true",
        help="Assign a spending category to each purchase (local DistilBERT model, cached).",
    )
    parse_cmd.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON instead of a human-readable table.",
    )
    parse_cmd.add_argument(
        "--debug",
        action="store_true",
        help="Print parser-internal diagnostics (candidate row count, reference period).",
    )

    imp = sub.add_parser("import", help="Import statement PDF(s) into the local store.")
    imp.add_argument("path", type=Path, help="A PDF file, or a folder of PDFs.")

    rep = sub.add_parser("report", help="Show spending by category over time.")
    rep.add_argument("--start", default=None, help="ISO date lower bound (YYYY-MM-DD).")
    rep.add_argument("--end", default=None, help="ISO date upper bound (YYYY-MM-DD).")

    srv = sub.add_parser("serve", help="Launch the local web dashboard + watched drop folder.")
    srv.add_argument("--drop", type=Path, default=None, help="Watched drop folder (default: ~/.cc_tool/inbox).")
    srv.add_argument("--host", default="127.0.0.1")
    srv.add_argument("--port", type=int, default=8765)

    setp = sub.add_parser("set", help="Pin a merchant to a category (user override).")
    setp.add_argument("merchant", help="Merchant text (will be normalized).")
    setp.add_argument("category", help="Category to assign.")

    args = p.parse_args(argv)

    if args.cmd == "import":
        return _cmd_import(args)
    if args.cmd == "report":
        return _cmd_report(args)
    if args.cmd == "serve":
        return _cmd_serve(args)
    if args.cmd == "set":
        return _cmd_set(args)
    return _cmd_parse(args)


def _cmd_parse(args: argparse.Namespace) -> int:
    if not args.pdf.exists():
        print(f"File not found: {args.pdf}", file=sys.stderr)
        return 2

    pdf_bytes = args.pdf.read_bytes()

    parser = GenericStatementParser()
    result = parser.parse(pdf_bytes, debug=args.debug)
    recon = reconcile(result)

    if args.categorize:
        Categorizer(DistilBertCategorizer()).categorize(result, debug=args.debug)

    if args.json:
        out = {
            "parse": result.model_dump(),
            "reconciliation": {
                "passed": recon.passed,
                "rows_sum_cents": recon.rows_sum_cents,
                "printed_total_cents": recon.printed_total_cents,
                "delta_cents": recon.delta_cents,
                "reason": recon.reason,
            },
        }
        print(json.dumps(out, indent=2))
        return 0 if recon.passed else 1

    print(f"Issuer:         {result.issuer or '(unknown)'}")
    print(f"Period:         {result.period_start or '?'} -> {result.period_end or '?'}")
    print(f"Transactions:   {len(result.rows)}")
    print(f"Rows sum:       ${recon.rows_sum_cents / 100:,.2f}")
    if recon.printed_total_cents is not None:
        print(f"Printed total:  ${recon.printed_total_cents / 100:,.2f}")
    print(f"Reconciled:     {'YES' if recon.passed else 'NO'}")
    print(f"  -> {recon.reason}")
    print()
    if args.categorize:
        print(
            f"{'Date':<12} {'Merchant (normalized)':<32} {'Amount':>10}  {'Category':<15} Descriptor"
        )
        print(f"{'-' * 12} {'-' * 32} {'-' * 10}  {'-' * 15} {'-' * 30}")
        for row in result.rows:
            norm = normalize_merchant(row.descriptor)
            amount_str = f"${row.amount_cents / 100:,.2f}"
            cat = row.category or ("-" if row.transaction_type == "purchase" else row.transaction_type)
            print(
                f"{row.date:<12} {norm[:32]:<32} {amount_str:>10}  {cat[:15]:<15} {row.descriptor[:30]}"
            )
        _print_category_totals(result)
    else:
        print(
            f"{'Date':<12} {'Merchant (normalized)':<35} {'Amount':>10}  Descriptor"
        )
        print(f"{'-' * 12} {'-' * 35} {'-' * 10}  {'-' * 40}")
        for row in result.rows:
            norm = normalize_merchant(row.descriptor)
            amount_str = f"${row.amount_cents / 100:,.2f}"
            print(
                f"{row.date:<12} {norm[:35]:<35} {amount_str:>10}  {row.descriptor[:40]}"
            )

    return 0 if recon.passed else 1


def _print_category_totals(result) -> None:
    """Per-category spending totals (purchase rows only)."""
    totals: dict[str, int] = {}
    for row in result.rows:
        if row.category:
            totals[row.category] = totals.get(row.category, 0) + row.amount_cents
    if not totals:
        return
    print()
    print("Spending by category:")
    for cat, cents in sorted(totals.items(), key=lambda kv: kv[1], reverse=True):
        print(f"  {cat:<18} ${cents / 100:,.2f}")


def _build_categorizer(storage) -> Categorizer:
    return Categorizer(DistilBertCategorizer(), storage.merchant_cache())


def _cmd_import(args: argparse.Namespace) -> int:
    from .storage import Storage
    from .importer import import_pdf, import_folder

    if not args.path.exists():
        print(f"Path not found: {args.path}", file=sys.stderr)
        return 2

    storage = Storage()
    parser = GenericStatementParser()
    try:
        categorizer = _build_categorizer(storage)
    except Exception as e:
        print(f"(categorization disabled: {e})", file=sys.stderr)
        categorizer = None

    if args.path.is_dir():
        outcomes = import_folder(args.path, storage=storage, parser=parser, categorizer=categorizer)
    else:
        outcomes = [import_pdf(args.path, storage=storage, parser=parser, categorizer=categorizer)]

    new = sum(1 for o in outcomes if o.status in ("imported", "review"))
    for o in outcomes:
        print(f"  {o.file_name:20} {o.status:9} {o.detail}")
    print(f"\n{len(outcomes)} file(s), {new} new. "
          f"{storage.statement_count()} statements / {storage.transaction_count()} transactions total.")
    storage.close()
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    from .storage import Storage

    storage = Storage()
    rows = storage.category_totals_by_month(start=args.start, end=args.end)
    if not rows:
        print("No transactions yet. Import a statement first (cc-tool import <path>).")
        storage.close()
        return 0

    months = sorted({r["month"] for r in rows})
    cats = sorted({r["category"] for r in rows})
    cell = {(r["month"], r["category"]): r["total_cents"] for r in rows}

    w = 16
    print(f"{'Category':<16}" + "".join(f"{m:>12}" for m in months) + f"{'Total':>12}")
    print("-" * (16 + 12 * (len(months) + 1)))
    for cat in cats:
        line = f"{cat:<16}"
        total = 0
        for m in months:
            v = cell.get((m, cat))
            total += v or 0
            line += f"{('$%.2f' % (v/100)) if v else '-':>12}"
        line += f"{'$%.2f' % (total/100):>12}"
        print(line)
    storage.close()
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    from .storage import default_db_path
    from .webapp import serve

    drop = args.drop or (Path.home() / ".cc_tool" / "inbox")
    serve(
        db_path=default_db_path(),
        drop_folder=drop,
        host=args.host,
        port=args.port,
    )
    return 0


def _cmd_set(args: argparse.Namespace) -> int:
    from .storage import Storage

    storage = Storage()
    try:
        matches, n = storage.apply_category_like(args.merchant, args.category)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        storage.close()
        return 2
    if matches:
        print(f"Set {args.category} for {len(matches)} merchant(s), {n} transaction(s):")
        for m in matches:
            print(f"  {m}")
    else:
        print(f"No stored merchant matched '{args.merchant}'. Recorded the mapping "
              f"for future imports.")
    storage.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
