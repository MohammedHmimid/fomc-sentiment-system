def compute_market_signal(prices):
    """Percentage change in Close across the window. `prices` is a DataFrame with a 'Close' column.
    Returns float percent, or None if insufficient data."""
    closes = list(prices["Close"]) if "Close" in prices else []
    if len(closes) < 2:
        return None
    first, last = float(closes[0]), float(closes[-1])
    if first == 0:
        return None
    return (last - first) / first * 100.0


def fetch_market_window(ticker="^GSPC", start=None, end=None):
    """Download daily prices for [start, end] inclusive. Returns a DataFrame."""
    import yfinance as yf
    return yf.download(ticker, start=start, end=end, progress=False)
