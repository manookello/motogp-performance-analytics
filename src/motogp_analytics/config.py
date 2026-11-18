"""Project paths and constants."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "reports" / "figures"
MODELS = ROOT / "models"

FIRST_SEASON = 2013  # first season of the Q1/Q2 qualifying format
LAST_SEASON = 2026
TEST_SEASONS = [2022, 2023, 2024, 2025, 2026]

# Session types we keep. Q1/Q2 share type "Q" and are told apart by "number".
SESSION_TYPES = {"Q", "SPR", "RAC"}
