"""
NovaMind — Level-of-Detail compute routing

Thread-safe global precision routing based on `contextvars`.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar


_LOD_TIER: ContextVar[str] = ContextVar("novamind_lod_tier", default="high_fi")


def get_lod_tier() -> str:
    return _LOD_TIER.get()


def is_low_fi() -> bool:
    return get_lod_tier() == "low_fi"


def is_high_fi() -> bool:
    return get_lod_tier() == "high_fi"


@contextmanager
def lod_compute(tier: str = "low_fi"):
    if tier not in {"low_fi", "high_fi"}:
        raise ValueError("tier must be 'low_fi' or 'high_fi'")

    token = _LOD_TIER.set(tier)
    try:
        yield
    finally:
        _LOD_TIER.reset(token)
