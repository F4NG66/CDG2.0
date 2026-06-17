from ..config import BackendConfig
from .dlm_runner import DLMRunner
from .dummy_runner import DummyRunner


def build_runner(cfg: BackendConfig, sae_root: str = "", device: str = "cuda",
                 dummy: bool = False):
    """dummy=True -> CPU random stub (test pipeline); otherwise load real model + SAE."""
    if dummy:
        return DummyRunner(cfg)
    if not sae_root:
        raise ValueError("no sae_root provided")
    return DLMRunner(cfg, sae_root=sae_root, device=device)


__all__ = ["build_runner", "DLMRunner", "DummyRunner"]
