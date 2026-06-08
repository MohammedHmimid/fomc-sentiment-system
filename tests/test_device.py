from shared.device import get_device


def test_get_device_returns_known_value(monkeypatch):
    import shared.device as d
    monkeypatch.setattr(d, "_cuda_available", lambda: True)
    assert get_device() == "cuda"
    monkeypatch.setattr(d, "_cuda_available", lambda: False)
    assert get_device() == "cpu"
