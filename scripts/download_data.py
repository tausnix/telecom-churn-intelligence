"""Download the public IBM Telco Customer Churn dataset."""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import RAW_DATA_FILE, RAW_DIR  # noqa: E402

URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if RAW_DATA_FILE.exists():
        print(f"Dataset already present: {RAW_DATA_FILE}")
        return
    print(f"Downloading dataset -> {RAW_DATA_FILE}")
    urllib.request.urlretrieve(URL, RAW_DATA_FILE)
    print("Done.")


if __name__ == "__main__":
    main()
