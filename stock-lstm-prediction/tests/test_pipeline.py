"""End-to-end smoke test on synthetic data (needs TensorFlow, no internet)."""

import json
from pathlib import Path

import pytest

pytest.importorskip("tensorflow")

from src.config import Config  # noqa: E402
from src.pipeline import forecast_with_saved_model, run_pipeline  # noqa: E402


def test_full_pipeline_runs_and_saves_artifacts(tmp_path):
    cfg = Config(synthetic=True, lookback=20, epochs=2, batch_size=64, verbose=0,
                 data_dir=str(tmp_path / "data"), model_dir=str(tmp_path / "models"),
                 output_dir=str(tmp_path / "outputs"))
    res = run_pipeline(cfg)

    m = res["metrics"]
    assert m["n_test_samples"] > 50
    assert m["model"]["rmse"] > 0 and 0 <= m["model"]["directional_accuracy"] <= 100
    assert res["forecast"]["direction"] in ("UP", "DOWN")
    assert res["forecast"]["predicted_close"] > 0

    for key in ("model", "artifacts", "metrics", "predictions", "overview",
                "predictions_plot", "training", "scatter", "backtest"):
        assert Path(res["paths"][key]).exists(), key
    json.loads(Path(res["paths"]["metrics"]).read_text())


def test_missing_model_gives_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        forecast_with_saved_model("NOPE", model_dir=str(tmp_path))
