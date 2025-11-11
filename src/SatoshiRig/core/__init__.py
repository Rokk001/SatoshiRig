"""Core mining logic module."""
from .miner import Miner
from .state import MinerState

__all__ = ["Miner", "MinerState"]

# GPU compute imports are optional
try:
    from .gpu_compute import create_gpu_miner, CUDAMiner, OpenCLMiner
    __all__.extend(["create_gpu_miner", "CUDAMiner", "OpenCLMiner"])
except ImportError:
    pass

