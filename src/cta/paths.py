"""Filesystem locations for camera-taught-arm data and models."""
import os
from pathlib import Path

DEFAULT_DATA_DIR = r"C:\Users\trayd\dev\cta-data"


def data_dir() -> Path:
    """Data root from CTA_DATA_DIR (default cta-data under dev); created if missing."""
    p = Path(os.environ.get("CTA_DATA_DIR", DEFAULT_DATA_DIR))
    p.mkdir(parents=True, exist_ok=True)
    return p


def models_dir() -> Path:
    """Models from CTA_MODELS_DIR if set (tests point data_dir at a temp dir but keep real models)."""
    override = os.environ.get("CTA_MODELS_DIR")
    p = Path(override) if override else data_dir() / "models"
    p.mkdir(parents=True, exist_ok=True)
    return p
