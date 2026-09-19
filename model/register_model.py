from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import HfApi, create_repo

load_dotenv()

MODEL_PATH = Path(__file__).resolve().parent / "model.pkl"
MODEL_CARD_PATH = Path(__file__).resolve().parent / "model_card.md"
HF_REPO_ID = os.environ.get("HF_REPO_ID")  # e.g., "your-username/smhi-elpris-se2"
HF_TOKEN = os.environ.get("HF_TOKEN")


def main() -> None:
    if not HF_REPO_ID:
        raise SystemExit("Error: HF_REPO_ID environment variable is not set.")

    print(f"Connecting to Hugging Face Hub for repo: {HF_REPO_ID}...")
    api = HfApi(token=HF_TOKEN)
    
    # Ensure the repo exists
    create_repo(repo_id=HF_REPO_ID, repo_type="model", exist_ok=True, token=HF_TOKEN)

    if MODEL_PATH.exists():
        api.upload_file(
            path_or_fileobj=str(MODEL_PATH),
            path_in_repo="model.pkl",
            repo_id=HF_REPO_ID,
            repo_type="model",
        )
        print(f"✅ Uploaded {MODEL_PATH.name} to {HF_REPO_ID}")
    else:
        print(f"⚠️ Warning: {MODEL_PATH} not found. Run training first.")

    if MODEL_CARD_PATH.exists():
        api.upload_file(
            path_or_fileobj=str(MODEL_CARD_PATH),
            path_in_repo="README.md",
            repo_id=HF_REPO_ID,
            repo_type="model",
        )
        print(f"✅ Uploaded {MODEL_CARD_PATH.name} as README.md to {HF_REPO_ID}")


if __name__ == "__main__":
    main()