from __future__ import annotations

import argparse
import ntpath
import sys
import time
from math import isfinite
from pathlib import Path


def scene_paths_match(expected: str | Path, reported: str | Path) -> bool:
    return ntpath.normcase(ntpath.normpath(str(expected))) == ntpath.normcase(
        ntpath.normpath(str(reported))
    )


def require_expected_scene(sim, scene_path: str | Path) -> None:
    reported = sim.getStringParam(sim.stringparam_scene_path_and_name)
    if not scene_paths_match(scene_path, reported):
        raise RuntimeError(
            "CoppeliaSim loaded an unexpected scene: "
            f"expected={scene_path}, reported={reported}"
        )
    expected_name = ntpath.normcase(
        ntpath.basename(ntpath.normpath(str(scene_path)))
    )
    scene_root = (
        "/VisionQualityLab"
        if expected_name == "bl23_vision_quality_lab.ttt"
        else "/VisionCodeRoutingLab"
        if expected_name == "bl23_vision_code_routing_lab.ttt"
        else "/VisionOcrSortingLab"
        if expected_name == "bl23_vision_ocr_sorting_lab.ttt"
        else "/VisionLab"
    )
    for sentinel in (scene_root, "/BLX_base_link"):
        sim.getObject(sentinel)


def close_remote_client(client) -> None:
    socket_handle = getattr(client, "socket", None)
    context = getattr(client, "context", None)
    socket_error: BaseException | None = None
    context_error: BaseException | None = None
    try:
        if socket_handle is not None:
            close_socket = getattr(socket_handle, "close", None)
            if callable(close_socket):
                close_socket(linger=0)
    except BaseException as error:
        socket_error = error
    try:
        if context is not None:
            terminate_context = getattr(context, "term", None)
            if callable(terminate_context):
                terminate_context()
    except BaseException as error:
        context_error = error
    if socket_error is not None:
        if context_error is not None:
            raise socket_error from context_error
        raise socket_error
    if context_error is not None:
        raise context_error


def wait_for_scene(
    client_factory,
    *,
    host: str,
    port: int,
    scene_path: str | Path,
    timeout_s: float = 30.0,
    retry_interval_s: float = 0.1,
    clock=time.monotonic,
    sleep=time.sleep,
):
    if not isfinite(timeout_s) or timeout_s < 0:
        raise ValueError(
            "timeout_s must be finite and greater than or equal to zero"
        )
    if not isfinite(retry_interval_s) or retry_interval_s <= 0:
        raise ValueError(
            "retry_interval_s must be finite and greater than zero"
        )
    deadline = clock() + timeout_s
    last_error: Exception | None = None
    while True:
        client = None
        try:
            client = client_factory(host=host, port=port)
            sim = client.require("sim")
            sim.getSimulationState()
            require_expected_scene(sim, scene_path)
            return client, sim
        except Exception as error:
            last_error = error
            if client is not None:
                close_remote_client(client)

        remaining_s = deadline - clock()
        if remaining_s <= 0:
            raise TimeoutError(
                "CoppeliaSim readiness timed out at "
                f"{host}:{port} within {timeout_s:.3f}s; "
                f"last RPC error: {last_error}"
            ) from last_error
        sleep(min(retry_interval_s, remaining_s))


def main(
    argv=None,
    *,
    client_factory=None,
    out=sys.stdout,
    err=sys.stderr,
) -> int:
    parser = argparse.ArgumentParser(
        description="Wait for the expected CoppeliaSim scene Remote API."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=23000)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args(argv)

    if client_factory is None:
        from coppeliasim_zmqremoteapi_client import RemoteAPIClient

        client_factory = RemoteAPIClient

    try:
        client, _ = wait_for_scene(
            client_factory,
            host=args.host,
            port=args.port,
            scene_path=args.scene,
            timeout_s=args.timeout,
        )
        close_remote_client(client)
        print(
            f"READY {args.host}:{args.port} scene={args.scene}",
            file=out,
        )
        return 0
    except Exception as error:
        print(f"NOT_READY {error}", file=err)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
