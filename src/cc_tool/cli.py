"""Command-line entry. Parses a PDF and prints extracted transactions + the
reconciliation verdict."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

from .normalize import normalize_merchant
from .parser import OllamaStatementParser
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
        "--model",
        default=None,
        help="Override OLLAMA_MODEL for this run.",
    )
    parse_cmd.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON instead of a human-readable table.",
    )

    args = p.parse_args(argv)

    if not args.pdf.exists():
        print(f"File not found: {args.pdf}", file=sys.stderr)
        return 2

    pdf_bytes = args.pdf.read_bytes()
    parser = OllamaStatementParser(model=args.model)
    result = parser.parse(pdf_bytes)
    recon = reconcile(result)

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


if __name__ == "__main__":
    sys.exit(main())
