from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from vision_platform.errors import CameraUnavailableError


@dataclass(frozen=True)
class CoppeliaConnection:
    sim: Any
    client: Any | None
    owns_client: bool


class CoppeliaClientResolver:
    """Lazily resolve a shared or owned ZeroMQ Remote API connection."""

    def __init__(
        self,
        *,
        sim: Any | None = None,
        client: Any | None = None,
        host: str = "127.0.0.1",
        port: int = 23000,
        client_factory: Callable[..., Any] | None = None,
    ) -> None:
        self._injected_sim = sim
        self._injected_client = client
        self.host = str(host)
        self.port = int(port)
        self._client_factory = client_factory
        self._connection: CoppeliaConnection | None = None

    def resolve(self) -> CoppeliaConnection:
        if self._connection is not None:
            return self._connection
        if self._injected_sim is not None:
            self._connection = CoppeliaConnection(
                sim=self._injected_sim,
                client=self._injected_client,
                owns_client=False,
            )
            return self._connection

        client = self._injected_client
        owns_client = client is None
        try:
            if client is None:
                factory = self._client_factory
                if factory is None:
                    from coppeliasim_zmqremoteapi_client import RemoteAPIClient

                    factory = RemoteAPIClient
                client = factory(host=self.host, port=self.port)
            if hasattr(client, "require"):
                sim = client.require("sim")
            else:
                sim = client.getObject("sim")
        except Exception as error:
            raise CameraUnavailableError(
                f"Could not connect to CoppeliaSim at {self.host}:{self.port}: {error}",
                host=self.host,
                port=self.port,
            ) from error
        self._connection = CoppeliaConnection(
            sim=sim,
            client=client,
            owns_client=owns_client,
        )
        return self._connection

    def close_owned(self) -> None:
        connection = self._connection
        if connection is None or not connection.owns_client:
            return
        close = getattr(connection.client, "close", None)
        if callable(close):
            close()
        self._connection = None
