from __future__ import annotations

from abc import ABC, abstractmethod

from vision_platform.models import Frame


class CameraBackend(ABC):
    @property
    @abstractmethod
    def is_open(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def open(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def read(self, timeout_s: float = 1.0) -> Frame:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError

    def __enter__(self) -> "CameraBackend":
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
