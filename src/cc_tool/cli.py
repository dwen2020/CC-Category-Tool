"""Command-line entry. Parses a PDF and prints extracted transactions + the
reconciliation verdict."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

from .categorizer import Categorizer, FuelixCategorizer, OllamaCategorizer
from .deterministic import (
    AutoStatementParser,
    DeterministicStatementParser,
    GenericStatementParser,
)
from .normalize import normalize_merchant
from .parser import FuelixStatementParser, OllamaStatementParser
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
        help="Override FUELIX_MODEL (or OLLAMA_MODEL) for this run.",
    )
    parse_cmd.add_argument(
        "--backend",
        choices=["deterministic", "generic", "auto", "fuelix", "ollama"],
        default="generic",
        help=(
            "Parser backend. 'deterministic' uses per-issuer regex parsers "
            "(default). 'generic' uses the issuer-agnostic heuristic parser only. "
            "'auto' escalates issuer-specific -> generic -> LLM, accepting the "
            "first tier that reconciles. 'fuelix'/'ollama' force the LLM path."
        ),
    )
    parse_cmd.add_argument(
        "--fallback",
        choices=["ollama", "fuelix"],
        default="ollama",
        help="LLM backend used for --backend auto fallback (default: ollama).",
    )
    parse_cmd.add_argument(
        "--categorize",
        action="store_true",
        help="Assign a spending category to each purchase (LLM-backed, cached).",
    )
    parse_cmd.add_argument(
        "--cat-backend",
        choices=["fuelix", "ollama"],
        default="fuelix",
        help="LLM backend for categorization (default: fuelix).",
    )
    parse_cmd.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON instead of a human-readable table.",
    )
    parse_cmd.add_argument(
        "--debug",
        action="store_true",
        help="Print token counts, finish reason, and raw model response.",
    )

    args = p.parse_args(argv)

    if not args.pdf.exists():
        print(f"File not found: {args.pdf}", file=sys.stderr)
        return 2

    pdf_bytes = args.pdf.read_bytes()

    def make_llm(backend: str) -> object:
        if backend == "ollama":
            return OllamaStatementParser(model=args.model)
        return FuelixStatementParser(model=args.model)

    if args.backend == "deterministic":
        parser = DeterministicStatementParser()
    elif args.backend == "generic":
        parser = GenericStatementParser()
    elif args.backend == "auto":
        parser = AutoStatementParser(fallback=make_llm(args.fallback))
    else:
        parser = make_llm(args.backend)
    result = parser.parse(pdf_bytes, debug=args.debug)
    recon = reconcile(result)

    if args.categorize:
        cat_llm = (
            OllamaCategorizer(model=args.model)
            if args.cat_backend == "ollama"
            else FuelixCategorizer(model=args.model)
        )
        Categorizer(cat_llm).categorize(result, debug=args.debug)

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


if __name__ == "__main__":
    sys.exit(main())
