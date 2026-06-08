import importlib.util
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _load_market():
    spec = importlib.util.spec_from_file_location("market", ROOT / "data" / "market.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def test_compute_market_signal_pct_change():
    market = _load_market()
    df = pd.DataFrame({"Close": [100.0, 101.0, 103.0]})
    # signal = (last - first) / first * 100
    assert abs(market.compute_market_signal(df) - 3.0) < 1e-9


def test_compute_market_signal_empty_returns_none():
    market = _load_market()
    df = pd.DataFrame({"Close": []})
    assert market.compute_market_signal(df) is None
