"""Owned CoppeliaSim process lifecycle for the opt-in D1-01 acceptance."""

from __future__ import annotations

import os
import socket
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


def _listener_pid(port: int) -> int | None:
    """Return the Windows listener PID when available, without touching it."""

    command = (
        "(Get-NetTCPConnection -LocalPort "
        f"{int(port)} -State Listen -ErrorAction SilentlyContinue | "
        "Select-Object -First 1 -ExpandProperty OwningProcess)"
    )
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", command],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    value = result.stdout.strip()
    try:
        return int(value) if value else None
    except ValueError:
        return None


def _port_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.25):
            return True
    except OSError:
        return False


@dataclass
class OwnedCoppeliaSim:
    process: subprocess.Popen
    executable: Path
    scene: Path
    host: str
    port: int

    @classmethod
    def start(
        cls,
        scene: Path,
        *,
        host: str = "127.0.0.1",
        port: int = 23009,
        timeout_s: float = 30.0,
        coppelia_root: str | os.PathLike[str] | None = None,
    ) -> "OwnedCoppeliaSim":
        if port != 23009:
            raise RuntimeError("D1-01 CoppeliaSim process must use port 23009")
        if _listener_pid(port) is not None or _port_open(host, port):
            raise RuntimeError("dedicated D1-01 port 23009 is already occupied; refusing to clean it")
        root = Path(coppelia_root or os.environ.get("COPPELIASIM_ROOT", r"E:\CoppeliaSim")).resolve()
        executable = root / "coppeliaSim.exe"
        if not executable.is_file():
            raise RuntimeError(f"CoppeliaSim executable was not found: {executable}")
        scene = Path(scene).resolve()
        if not scene.is_file():
            raise RuntimeError(f"D1-01 scene was not found: {scene}")
        process = subprocess.Popen(
            [str(executable), f"-GzmqRemoteApi.rpcPort={port}", str(scene)],
            cwd=str(root),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
        )
        owned = cls(process=process, executable=executable, scene=scene, host=host, port=port)
        deadline = time.monotonic() + timeout_s
        try:
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"CoppeliaSim exited before port {port} became ready")
                listener_pid = _listener_pid(port)
                if listener_pid is not None and listener_pid != process.pid:
                    raise RuntimeError(
                        f"D1-01 port {port} is owned by unexpected PID {listener_pid}"
                    )
                if listener_pid == process.pid:
                    # Complete a real Remote API handshake before yielding the process.
                    from coppeliasim_zmqremoteapi_client import RemoteAPIClient

                    client = None
                    try:
                        client = RemoteAPIClient(host=host, port=port)
                        sim = client.require("sim")
                        int(sim.getObject("/RgbdLab/CameraRig/RgbdSensor"))
                        return owned
                    except Exception:
                        if client is not None:
                            close = getattr(client, "close", None)
                            if callable(close):
                                close()
                    finally:
                        if client is not None:
                            close = getattr(client, "close", None)
                            if callable(close):
                                close()
                time.sleep(0.2)
            raise RuntimeError(f"CoppeliaSim port {port} did not become ready within {timeout_s:.1f}s")
        except BaseException:
            owned.stop()
            raise

    def stop(self, timeout_s: float = 10.0) -> None:
        if self.process.poll() is not None:
            return
        listener = _listener_pid(self.port)
        if listener is not None and listener != self.process.pid:
            raise RuntimeError(
                f"D1-01 cleanup refused replacement listener PID {listener}; owned PID is {self.process.pid}"
            )
        self.process.terminate()
        try:
            self.process.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=timeout_s)
        if _listener_pid(self.port) == self.process.pid:
            raise RuntimeError("owned CoppeliaSim process still owns port 23009 after cleanup")

    def __enter__(self) -> "OwnedCoppeliaSim":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.stop()


__all__ = ["OwnedCoppeliaSim"]
