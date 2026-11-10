"""Inference helpers for scoring individual customers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib

from src.config import BASE_DIR, METADATA_FILE, MODEL_FILE
from src.preprocessing import prepare_customer_record
from src.utils import load_json


def load_assets(model_name: str | None = None) -> tuple[dict, dict, dict]:
    """Load metadata, all available pipelines and their thresholds."""
    if not METADATA_FILE.exists():
        raise FileNotFoundError(
            "Model artefacts not found. Run `python run_pipeline.py` first."
        )
    metadata = load_json(METADATA_FILE)
    pipelines: dict[str, object] = {}
    thresholds: dict[str, float] = {}
    for entry in metadata["models"]:
        path = BASE_DIR / entry["path"]
        pipelines[entry["name"]] = joblib.load(path)
        thresholds[entry["name"]] = float(entry["threshold"])

    if model_name is not None:
        if model_name not in pipelines:
            raise ValueError(
                f"Unknown model '{model_name}'. Available: {sorted(pipelines)}"
            )
        metadata = {**metadata, "best_model": model_name}

    # Backwards-compatible alias for the winner.
    pipelines.setdefault("best", joblib.load(MODEL_FILE))
    thresholds.setdefault("best", float(metadata["threshold"]))
    return metadata, pipelines, thresholds


def score_customer(
    record: dict,
    pipeline,
    metadata: dict,
    threshold: float | None = None,
) -> dict:
    """Score a single raw customer record and return probability + label."""
    X = prepare_customer_record(record, metadata.get("input_features"))
    probability = float(pipeline.predict_proba(X)[0, 1])
    cut_off = float(metadata["threshold"] if threshold is None else threshold)
    prediction = int(probability >= cut_off)
    return {
        "churn_probability": probability,
        "prediction": prediction,
        "label": "Churn" if prediction else "Retain",
        "threshold": cut_off,
        "risk_band": (
            "High" if probability >= 0.60 else "Moderate" if probability >= 0.30 else "Low"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Score a telecom customer for churn risk.")
    parser.add_argument(
        "--json",
        type=Path,
        required=True,
        help="Path to a JSON file containing one customer record.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name (default: best model from training).",
    )
    args = parser.parse_args()

    with open(args.json, "r", encoding="utf-8") as handle:
        record = json.load(handle)

    metadata, pipelines, thresholds = load_assets(args.model)
    name = args.model or metadata["best_model"]
    result = score_customer(record, pipelines[name], metadata, thresholds[name])
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
