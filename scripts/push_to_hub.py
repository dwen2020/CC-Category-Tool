"""Push the locally-trained model and its training data to the public Hugging
Face Hub repos that categorizer.HF_MODEL_REPO_ID points at, so a fresh install
of cc-tool can download the model instead of requiring the original trainer's
machine. Run this once after training, and again whenever the model is
retrained.

Usage:
    pip install -e ".[dev]"
    huggingface-cli login   # paste a token from huggingface.co/settings/tokens
    python scripts/push_to_hub.py
"""

from __future__ import annotations

from pathlib import Path

from huggingface_hub import HfApi

from cc_tool.categorizer import HF_MODEL_REPO_ID, default_model_path

DATASET_REPO_ID = "Dluvhugging/cc-tool-merchant-training-data"
DATASET_FILE = Path(__file__).resolve().parent.parent / "data" / "cc_merchants_overture.csv"


def main() -> None:
    api = HfApi()

    model_path = default_model_path()
    if not model_path.exists():
        raise SystemExit(f"No local model at {model_path} -- nothing to push.")

    print(f"Creating/using model repo {HF_MODEL_REPO_ID} ...")
    api.create_repo(HF_MODEL_REPO_ID, repo_type="model", exist_ok=True, private=False)
    print(f"Uploading {model_path} ...")
    api.upload_folder(repo_id=HF_MODEL_REPO_ID, repo_type="model", folder_path=str(model_path))

    if not DATASET_FILE.exists():
        raise SystemExit(f"No dataset file at {DATASET_FILE} -- nothing to push.")

    print(f"Creating/using dataset repo {DATASET_REPO_ID} ...")
    api.create_repo(DATASET_REPO_ID, repo_type="dataset", exist_ok=True, private=False)
    print(f"Uploading {DATASET_FILE} ...")
    api.upload_file(
        repo_id=DATASET_REPO_ID,
        repo_type="dataset",
        path_or_fileobj=str(DATASET_FILE),
        path_in_repo=DATASET_FILE.name,
    )

    print("Done.")
    print(f"Model:   https://huggingface.co/{HF_MODEL_REPO_ID}")
    print(f"Dataset: https://huggingface.co/datasets/{DATASET_REPO_ID}")


if __name__ == "__main__":
    main()
