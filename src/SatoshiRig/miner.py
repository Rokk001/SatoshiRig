"""Compatibility facade for historical imports.

DEPRECATED: This module is kept for backward compatibility only.
Prefer using: from SatoshiRig.core.miner import Miner
"""

from .core.miner import Miner  # noqa: F401

__all__ = ["Miner"]
