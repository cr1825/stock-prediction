"""Central configuration for the pipeline."""

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class Config:
    # --- Data ---
    ticker: str = "AAPL"
    start: str = "2013-01-01"
    end: Optional[str] = None  # None -> up to the latest available day
    synthetic: bool = False  # use generated data (offline smoke test only)

    # --- Sequence / split ---
    lookback: int = 60  # trading days fed to the LSTM per sample
    train_frac: float = 0.70
    val_frac: float = 0.15  # remainder (0.15) is the untouched test set

    # --- Model ---
    lstm_units: Tuple[int, int] = (64, 32)
    dropout: float = 0.2
    learning_rate: float = 1e-3

    # --- Training ---
    epochs: int = 50
    batch_size: int = 32
    seed: int = 42
    verbose: int = 1

    # --- Paths ---
    data_dir: str = "data"
    model_dir: str = "models"
    output_dir: str = "outputs"
