from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

from simulation.training_scenes.scene_contract import validate_scene_contract
from vision_platform.cameras.coppeliasim import CoppeliaSimCamera
from vision_platform.coppeliasim_readiness import close_remote_client
from vision_platform.experiments.catalog import ExperimentCatalog
from vision_platform.vision_quality import load_profile_catalog
from vision_platform.vision_quality.controller import controller_for_experiment


ROOT = Path(__file__).resolve().parents[2]


def _vision_frame_observation(image_bgr):
    assert isinstance(image_bgr, np.ndarray)
    assert image_bgr.dtype == np.uint8
    assert image_bgr.ndim == 3 and image_bgr.shape[2] == 3
    height, width = image_bgr.shape[:2]
    total = height * width
    assert total > 0
    pixels = image_bgr.astype(np.int16)
    blue, green, red = (pixels[:, :, index] for index in range(3))
    color_counts = {
        "red": int(((red > green + 20) & (red > blue + 20) & (red > 30)).sum()),
        "green": int(
            ((green > red + 20) & (green > blue + 20) & (green > 30)).sum()
        ),
        "blue": int(((blue > red + 20) & (blue > green + 20) & (blue > 30)).sum()),
    }
    minimum_color_pixels = max(16, int(total * 0.0005))
    assert all(
        count >= minimum_color_pixels for count in color_counts.values()
    ), color_counts

    luma = image_bgr.astype(np.float32).mean(axis=2)
    assert float(luma.std()) >= 5.0
    neutral = pixels.max(axis=2) - pixels.min(axis=2) <= 15
    dark = neutral & (luma <= 70.0)
    bright = neutral & (luma >= 170.0)
    transitions = int(
        (
            (dark[:, :-1] & bright[:, 1:])
            | (bright[:, :-1] & dark[:, 1:])
        ).sum()
        + (
            (dark[:-1, :] & bright[1:, :])
            | (bright[:-1, :] & dark[1:, :])
        ).sum()
    )
    assert transitions >= max(8, int(total * 0.00005)), transitions
    return {
        "mean_luma": float(luma.mean()),
        "std_luma": float(luma.std()),
        "color_counts": color_counts,
        "neutral_transitions": transitions,
    }


def _resolution_target_bounds(image_shape, *, profile, spec):
    height, width = image_shape[:2]
    assert (width, height) == tuple(profile.resolution)
    orientation = tuple(float(value) for value in spec["camera"]["orientation_deg"])
    assert orientation == (180.0, 0.0, 0.0)
    camera_x, camera_y, _ = (
        float(value) for value in spec["camera"]["rig_position_m"]
    )
    target = spec["samples"]["ResolutionTarget"]
    center_x, center_y, center_z = (
        float(value) for value in target["position_m"]
    )
    size_x, size_y, size_z = (
        float(value) for value in target["size_m"]
    )
    depth = float(profile.camera_rig_z_m) - (center_z + size_z / 2.0)
    half_view = depth * math.tan(
        math.radians(float(profile.perspective_angle_deg)) / 2.0
    )
    assert half_view > 0
    # For the fixed 180-degree-down Coppelia sensor, the normalized horizontal
    # centre follows -world-X and the normalized vertical centre follows
    # +world-Y after CoppeliaSimCamera's bottom-up frame normalization.
    centre_u = 0.5 - (center_x - camera_x) / (2.0 * half_view)
    centre_v = 0.5 + (center_y - camera_y) / (2.0 * half_view)
    half_width = size_x / (4.0 * half_view)
    half_height = size_y / (4.0 * half_view)
    x0 = max(0, math.floor((centre_u - half_width) * width))
    x1 = min(width, math.ceil((centre_u + half_width) * width))
    y0 = max(0, math.floor((centre_v - half_height) * height))
    y1 = min(height, math.ceil((centre_v + half_height) * height))
    assert x1 - x0 >= 12 and y1 > y0
    return x0, y0, x1, y1


def _resolution_target_observation(image_bgr, *, profile, spec):
    x0, y0, x1, y1 = _resolution_target_bounds(
        image_bgr.shape,
        profile=profile,
        spec=spec,
    )
    vertical_inset = max(1, int((y1 - y0) * 0.15))
    roi = image_bgr[y0 + vertical_inset : y1 - vertical_inset, x0:x1]
    assert roi.size > 0
    column_luma = roi.astype(np.float32).mean(axis=(0, 2))
    low, high = np.percentile(column_luma, [10, 90])
    assert float(high - low) >= 40.0, (float(low), float(high))
    bright = column_luma > (float(low) + float(high)) / 2.0
    transitions = int(np.count_nonzero(bright[1:] != bright[:-1]))
    assert transitions >= 8, transitions
    return {
        "bounds": [x0, y0, x1, y1],
        "stripe_contrast": float(high - low),
        "stripe_transitions": transitions,
    }


def _cleanup_online_resources(*, controller, camera, client, primary):
    failures = []
    for label, resource, method in (
        ("controller.reset", controller, "reset"),
        ("camera.close", camera, "close"),
    ):
        if resource is None:
            continue
        try:
            getattr(resource, method)()
        except BaseException as exc:
            failures.append((label, exc))
    if client is not None:
        try:
            close_remote_client(client)
        except BaseException as exc:
            failures.append(("close_remote_client", exc))
    if not failures:
        return
    if primary is not None:
        for label, exc in failures:
            primary.add_note(
                f"cleanup {label} failed: {type(exc).__name__}: {exc}"
            )
        return
    raise BaseExceptionGroup(
        "vision quality online cleanup failed",
        [exc for _label, exc in failures],
    )


def test_frame_observation_rejects_empty_background_and_accepts_four_targets():
    with pytest.raises(AssertionError):
        _vision_frame_observation(np.full((128, 128, 3), 20, dtype=np.uint8))

    image = np.full((128, 128, 3), 190, dtype=np.uint8)
    image[12:32, 12:32] = [20, 20, 210]
    image[12:32, 48:68] = [20, 210, 20]
    image[12:32, 84:104] = [210, 20, 20]
    for index in range(12):
        value = 20 if index % 2 == 0 else 240
        image[70:100, 16 + index * 6 : 22 + index * 6] = value

    observed = _vision_frame_observation(image)

    assert observed["std_luma"] > 5
    assert observed["neutral_transitions"] >= 8


def test_resolution_target_observation_rejects_plain_roi_and_accepts_stripes():
    spec = json.loads(
        (
            ROOT / "simulation/vision_quality_lab/scene_spec.json"
        ).read_text(encoding="utf-8")
    )
    profile = SimpleNamespace(
        resolution=(128, 128),
        perspective_angle_deg=60,
        camera_rig_z_m=0.7,
    )
    plain = np.full((128, 128, 3), 190, dtype=np.uint8)

    with pytest.raises(AssertionError):
        _resolution_target_observation(plain, profile=profile, spec=spec)

    striped = plain.copy()
    x0, y0, x1, y1 = _resolution_target_bounds(
        striped.shape,
        profile=profile,
        spec=spec,
    )
    stripe_edges = np.linspace(x0, x1, 13, dtype=int)
    for index, (left, right) in enumerate(
        zip(stripe_edges[:-1], stripe_edges[1:])
    ):
        striped[y0:y1, left:right] = 20 if index % 2 == 0 else 240

    observed = _resolution_target_observation(
        striped,
        profile=profile,
        spec=spec,
    )

    assert observed["stripe_transitions"] >= 8


def test_resolution_target_bounds_follow_fixed_sensor_world_axis_mapping():
    spec = json.loads(
        (
            ROOT / "simulation/vision_quality_lab/scene_spec.json"
        ).read_text(encoding="utf-8")
    )
    profile = SimpleNamespace(
        resolution=(512, 512),
        perspective_angle_deg=60,
        camera_rig_z_m=0.7,
    )

    bounds = _resolution_target_bounds(
        (512, 512, 3),
        profile=profile,
        spec=spec,
    )

    assert bounds == (189, 282, 243, 323)


def test_resolution_target_observation_rejects_horizontal_roi_interference():
    spec = json.loads(
        (
            ROOT / "simulation/vision_quality_lab/scene_spec.json"
        ).read_text(encoding="utf-8")
    )
    profile = SimpleNamespace(
        resolution=(128, 128),
        perspective_angle_deg=60,
        camera_rig_z_m=0.7,
    )
    image = np.full((128, 128, 3), 190, dtype=np.uint8)
    x0, y0, x1, y1 = _resolution_target_bounds(
        image.shape,
        profile=profile,
        spec=spec,
    )
    stripe_edges = np.linspace(y0, y1, 13, dtype=int)
    for index, (top, bottom) in enumerate(
        zip(stripe_edges[:-1], stripe_edges[1:])
    ):
        image[top:bottom, x0:x1] = 20 if index % 2 == 0 else 240

    with pytest.raises(AssertionError):
        _resolution_target_observation(
            image,
            profile=profile,
            spec=spec,
        )


def test_cleanup_preserves_primary_failure_and_runs_every_step(monkeypatch):
    events = []

    class FailingController:
        def reset(self):
            events.append("reset")
            raise RuntimeError("reset failed")

    class FailingCamera:
        def close(self):
            events.append("camera.close")
            raise ValueError("camera close failed")

    def failing_remote_close(_client):
        events.append("remote.close")
        raise OSError("remote close failed")

    monkeypatch.setitem(
        _cleanup_online_resources.__globals__,
        "close_remote_client",
        failing_remote_close,
    )
    primary = AssertionError("body failed")

    _cleanup_online_resources(
        controller=FailingController(),
        camera=FailingCamera(),
        client=object(),
        primary=primary,
    )

    assert events == ["reset", "camera.close", "remote.close"]
    assert len(primary.__notes__) == 3


def test_cleanup_raises_group_after_all_steps_without_primary(monkeypatch):
    events = []

    class FailingController:
        def reset(self):
            events.append("reset")
            raise RuntimeError("reset failed")

    class FailingCamera:
        def close(self):
            events.append("camera.close")
            raise ValueError("camera close failed")

    def failing_remote_close(_client):
        events.append("remote.close")
        raise OSError("remote close failed")

    monkeypatch.setitem(
        _cleanup_online_resources.__globals__,
        "close_remote_client",
        failing_remote_close,
    )

    with pytest.raises(ExceptionGroup) as raised:
        _cleanup_online_resources(
            controller=FailingController(),
            camera=FailingCamera(),
            client=object(),
            primary=None,
        )

    assert events == ["reset", "camera.close", "remote.close"]
    assert len(raised.value.exceptions) == 3


def test_cleanup_groups_base_exceptions_without_losing_later_steps(monkeypatch):
    events = []

    class InterruptedController:
        def reset(self):
            events.append("reset")
            raise KeyboardInterrupt("interrupted reset")

    class ClosingCamera:
        def close(self):
            events.append("camera.close")

    monkeypatch.setitem(
        _cleanup_online_resources.__globals__,
        "close_remote_client",
        lambda _client: events.append("remote.close"),
    )

    with pytest.raises(BaseExceptionGroup) as raised:
        _cleanup_online_resources(
            controller=InterruptedController(),
            camera=ClosingCamera(),
            client=object(),
            primary=None,
        )

    assert events == ["reset", "camera.close", "remote.close"]
    assert isinstance(raised.value.exceptions[0], KeyboardInterrupt)


@pytest.mark.coppeliasim
def test_vision_quality_scene_required_paths_profiles_and_frames(request):
    host = (
        request.config.getoption("--coppelia-host")
        or os.environ.get("COPPELIA_HOST")
        or "127.0.0.1"
    )
    configured_port = request.config.getoption("--coppelia-port")
    port = (
        configured_port
        if configured_port is not None
        else int(os.environ.get("COPPELIA_PORT", "23005"))
    )
    definition = ExperimentCatalog.load(
        ROOT / "config/experiments/catalog.json",
        project_root=ROOT,
    ).require("V1-01")
    client = None
    camera = None
    controller = None
    primary = None
    try:
        client = RemoteAPIClient(host=host, port=port)
        sim = client.require("sim")
        loaded = Path(
            sim.getStringParam(sim.stringparam_scene_path_and_name)
        ).resolve()
        assert loaded == definition.scene.resolve()
        manifest = json.loads(
            definition.scene_manifest.read_text(encoding="utf-8")
        )
        report = validate_scene_contract(
            definition.scene.parent / "scene_spec.json",
            definition.scene_manifest,
            project_root=ROOT,
        )
        assert report["status"] == "PASS"
        for path in manifest["required_paths"]:
            sim.getObject(path)
        for index in range(1, 13):
            sim.getObject(
                f"/VisionQualityLab/Samples/ResolutionTarget/Stripe{index:02d}"
            )

        profile_catalog = load_profile_catalog(
            definition.scene.parent / "profiles.json"
        )
        scene_spec = json.loads(
            (definition.scene.parent / "scene_spec.json").read_text(
                encoding="utf-8"
            )
        )
        camera = CoppeliaSimCamera(
            sensor_path=profile_catalog.sensor_path,
            sim=sim,
            client=client,
        )
        application = SimpleNamespace(
            sim=sim,
            camera=camera,
            config=SimpleNamespace(camera_backend="sim"),
        )
        controller = controller_for_experiment(
            application,
            definition,
            manifest,
        )
        assert controller is not None
        camera.open()
        observed = {}
        for profile_id in ("standard", "wide_dim", "detail_bright"):
            state = controller.apply(profile_id)
            frame = camera.read(timeout_s=5.0)
            observed[profile_id] = (
                frame.width,
                frame.height,
                hashlib.sha256(frame.image_bgr.tobytes()).hexdigest(),
                _vision_frame_observation(frame.image_bgr),
                _resolution_target_observation(
                    frame.image_bgr,
                    profile=state,
                    spec=scene_spec,
                ),
            )
            assert (frame.width, frame.height) == state.resolution
        assert len({item[2] for item in observed.values()}) == 3
        assert (
            observed["wide_dim"][3]["mean_luma"]
            < observed["standard"][3]["mean_luma"]
            < observed["detail_bright"][3]["mean_luma"]
        )
    except BaseException as exc:
        primary = exc
        raise
    finally:
        _cleanup_online_resources(
            controller=controller,
            camera=camera,
            client=client,
            primary=primary,
        )
