"""Shared helpers for plotting, serialisation and reporting."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import COLORS, FIGURES_DIR, ensure_directories


def setup_plot_style() -> None:
    """Apply a consistent, presentation-ready visual style."""
    sns.set_theme(
        style="whitegrid",
        context="notebook",
        font="DejaVu Sans",
        font_scale=1.05,
        rc={
            "axes.edgecolor": "#CBD5E1",
            "axes.linewidth": 0.8,
            "axes.titleweight": "bold",
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "axes.labelcolor": "#334155",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "grid.color": COLORS["grid"],
            "grid.linewidth": 0.7,
            "legend.frameon": False,
        },
    )


def save_figure(fig: plt.Figure, filename: str, tight: bool = True) -> Path:
    """Save a figure to the reports/figures directory and close it."""
    ensure_directories()
    path = FIGURES_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight" if tight else None, facecolor="white")
    plt.close(fig)
    return path


def slugify(value: str) -> str:
    """Turn a model name into a filesystem-safe slug."""
    value = value.lower().strip()
    return re.sub(r"[^a-z0-9]+", "_", value).strip("_")


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serialisable")


def save_json(payload: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=_json_default)
    return path


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def pct(value: float, digits: int = 1) -> str:
    """Format a fraction as a percentage string."""
    return f"{value * 100:.{digits}f}%"
