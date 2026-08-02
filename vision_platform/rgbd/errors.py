"""Stable error types for the in-memory RGB-D kernel."""

from __future__ import annotations


class RgbdContractError(ValueError):
    """A caller supplied an invalid RGB-D contract value."""

    def __init__(self, code: str, message: str) -> None:
        self.code = str(code)
        super().__init__(f"{self.code}: {message}")


__all__ = ["RgbdContractError"]
