"""CDG: diffusion-LM 攻击 × SAE feature × steering defense 的实验脚手架（阶段1：generation+记录）。"""
from .config import get_backend_config, BackendConfig, RecordConfig, DecodeConfig

__all__ = ["get_backend_config", "BackendConfig", "RecordConfig", "DecodeConfig"]
