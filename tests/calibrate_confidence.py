"""Checks whether the DistilBertCategorizer's confidence score actually
separates correct from incorrect predictions, using the 100 hand-labeled rows
in merchants_gold.csv. Run after placing the trained model
(see ARCHITECTURE.md / CC_TOOL_MODEL_PATH):

    python tests/calibrate_confidence.py [threshold]

Prints overall accuracy, accuracy above/below the threshold, and how many gold
rows would land in the review queue at that threshold. If accuracy above the
threshold isn't clearly higher than accuracy below it, the threshold (or the
model) needs another look before trusting auto-acceptance.
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cc_tool.categorizer import DistilBertCategorizer  # noqa: E402
from cc_tool.storage import DEFAULT_REVIEW_THRESHOLD  # noqa: E402


def main() -> int:
    threshold = float(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_REVIEW_THRESHOLD

    gold_path = Path(__file__).parent / "merchants_gold.csv"
    with gold_path.open(encoding="utf-8") as f:
        gold = list(csv.DictReader(f))

    model = DistilBertCategorizer()
    predictions = model.categorize_merchants([row["descriptor"] for row in gold])

    above = below = 0
    above_correct = below_correct = 0
    for row in gold:
        pred_cat, conf = predictions[row["descriptor"]]
        correct = pred_cat == row["category"]
        if conf >= threshold:
            above += 1
            above_correct += correct
        else:
            below += 1
            below_correct += correct

    total = len(gold)
    total_correct = above_correct + below_correct
    print(f"Threshold: {threshold}")
    print(f"Overall accuracy:            {total_correct}/{total} ({100*total_correct/total:.1f}%)")
    print(
        f"Auto-accepted (conf >= {threshold}): {above}/{total} rows, "
        f"accuracy {100*above_correct/above:.1f}%" if above else "Auto-accepted: 0 rows"
    )
    print(
        f"Sent to review (conf < {threshold}):  {below}/{total} rows, "
        f"accuracy {100*below_correct/below:.1f}%" if below else "Sent to review: 0 rows"
    )
    if above and below and (above_correct / above) <= (below_correct / below):
        print(
            "\nWARNING: auto-accepted predictions are not more accurate than "
            "reviewed ones -- the threshold isn't doing its job. Consider "
            "raising it, or re-examine the model."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
