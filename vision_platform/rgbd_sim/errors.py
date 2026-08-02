"""Errors for the isolated CoppeliaSim RGB-D adapter."""

from __future__ import annotations


class RgbdSimContractError(ValueError):
    """A simulator adapter contract value is invalid."""

    def __init__(self, code: str, message: str) -> None:
        self.code = str(code)
        super().__init__(f"{self.code}: {message}")


__all__ = ["RgbdSimContractError"]
