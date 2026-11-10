"""Quick health check: load every model and score the sample customer."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import load_assets, score_customer  # noqa: E402


def main() -> None:
    metadata, pipelines, thresholds = load_assets()
    sample_file = PROJECT_ROOT / "examples" / "sample_customer.json"
    with open(sample_file, "r", encoding="utf-8") as handle:
        record = json.load(handle)

    print(f"Best model: {metadata['best_model']}")
    for name in [entry["name"] for entry in metadata["models"]]:
        result = score_customer(record, pipelines[name], metadata, thresholds[name])
        probability = result["churn_probability"]
        assert 0.0 <= probability <= 1.0, "Probability outside [0, 1]"
        print(
            f"  {name:<24} p={probability:.4f}  "
            f"threshold={result['threshold']:.3f}  risk={result['risk_band']}"
        )
    print("All models loaded and scored successfully.")


if __name__ == "__main__":
    main()
