"""Tests de la couche analyse/visualisation (analysis.py)."""
import pandas as pd

from analysis import correlation_table, plot_correlations, save_results


def _sample_df():
    # fusion suit parfaitement le marché ; nlp est l'inverse exact.
    return pd.DataFrame({
        "nlp": [1.0, 0.0, -1.0],
        "audio": [0.5, 0.0, -0.5],
        "vision": [0.2, 0.1, -0.2],
        "fusion": [0.8, 0.0, -0.8],
        "market": [1.6, 0.0, -1.6],
    })


def test_correlation_table_values():
    table = correlation_table(_sample_df())
    # fusion est parfaitement corrélé au marché, nlp parfaitement anti-corrélé.
    assert abs(table.loc["fusion", "corr_with_market"] - 1.0) < 1e-9
    assert abs(table.loc["nlp", "corr_with_market"] - 1.0) < 1e-9
    assert set(table.index) == {"nlp", "audio", "vision", "fusion"}


def test_plot_correlations_creates_png(tmp_path):
    out = tmp_path / "correlations.png"
    result = plot_correlations(_sample_df(), out)
    assert result == out
    assert out.exists()
    assert out.stat().st_size > 0  # une vraie image a été écrite


def test_correlation_table_ignores_missing_channel():
    df = _sample_df().drop(columns=["vision"])  # canal vision indisponible
    table = correlation_table(df)
    assert "vision" not in table.index
    assert "fusion" in table.index


def test_save_results_writes_all_artifacts(tmp_path):
    out = save_results(_sample_df(), results_dir=tmp_path)
    assert (out / "scores.csv").exists()
    assert (out / "correlations.csv").exists()
    assert (out / "correlations.png").exists()
    assert (out / "correlations.png").stat().st_size > 0
