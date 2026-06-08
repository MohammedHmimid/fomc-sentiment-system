def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def get_device() -> str:
    return "cuda" if _cuda_available() else "cpu"
