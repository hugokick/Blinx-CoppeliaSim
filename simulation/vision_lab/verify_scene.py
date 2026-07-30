from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

from vision_platform.cameras.coppeliasim import CoppeliaSimCamera
from vision_platform.coppelia_scene import stage_scene_for_coppeliasim


PROJECT_ROOT = Path(__file__).resolve().parents[2]
VISION_DIR = Path(__file__).resolve().parent
DEFAULT_REPORT = VISION_DIR / "runtime_verification.json"
DEFAULT_EVIDENCE_DIR = VISION_DIR / "evidence"
DEFAULT_FRAME = DEFAULT_EVIDENCE_DIR / "scene-camera.png"
DEFAULT_ROBOT_VIEWS_DIR = DEFAULT_EVIDENCE_DIR / "robot-views"


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _report_path(path: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def _wait_for_state(
    sim: Any,
    *,
    predicate: Any,
    timeout_s: float = 5.0,
) -> int:
    deadline = time.monotonic() + timeout_s
    state = int(sim.getSimulationState())
    while not predicate(state) and time.monotonic() < deadline:
        time.sleep(0.05)
        state = int(sim.getSimulationState())
    return state


def _pose_change(before: list[float], after: list[float]) -> float:
    position_delta = math.dist(before[:3], after[:3])
    direct_quaternion_delta = math.dist(before[3:], after[3:])
    negated_quaternion_delta = math.dist(
        before[3:],
        [-value for value in after[3:]],
    )
    return max(
        position_delta,
        min(direct_quaternion_delta, negated_quaternion_delta),
    )


def _verify_joint_motion(
    sim: Any,
    manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    visual_paths = {
        item["segment"]: item["path"] for item in manifest["robot_visuals"]
    }
    rows: list[dict[str, Any]] = []
    for index in range(1, 7):
        joint_path = f"/BLX_joint{index}"
        visual_path = visual_paths[f"link{index}"]
        joint = int(sim.getObject(joint_path))
        visual = int(sim.getObject(visual_path))
        original_joint_position = float(sim.getJointPosition(joint))
        before = [
            float(value) for value in sim.getObjectPose(visual, sim.handle_world)
        ]
        sim.setJointPosition(joint, original_joint_position + math.radians(10.0))
        after = [
            float(value) for value in sim.getObjectPose(visual, sim.handle_world)
        ]
        sim.setJointPosition(joint, original_joint_position)
        change = _pose_change(before, after)
        rows.append(
            {
                "joint_path": joint_path,
                "visual_path": visual_path,
                "test_rotation_deg": 10.0,
                "pose_change": change,
                "follows_joint": change > 1e-5,
            }
        )
    return rows


def _capture_camera(
    *,
    sim: Any,
    client: Any,
    output: Path,
) -> dict[str, Any]:
    camera = CoppeliaSimCamera(sim=sim, client=client)
    camera.open()
    frame = camera.read(timeout_s=5.0)
    output.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", frame.image_bgr)
    if not ok:
        raise RuntimeError("OpenCV could not encode the CoppeliaSim frame")
    encoded.tofile(str(output))
    image = np.asarray(frame.image_bgr, dtype=np.uint8)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    chromatic = (saturation >= 90) & (value >= 70)
    masks = {
        "red": chromatic & ((hue <= 10) | (hue >= 170)),
        "green": chromatic & (hue >= 35) & (hue <= 85),
        "blue": chromatic & (hue >= 90) & (hue <= 135),
        "yellow": chromatic & (hue >= 18) & (hue <= 35),
    }
    teaching_color_pixels = {
        name: int(mask.sum()) for name, mask in masks.items()
    }
    visible_teaching_colors = [
        name
        for name, pixel_count in teaching_color_pixels.items()
        if pixel_count >= 100
    ]
    stats = {
        "width": int(frame.width),
        "height": int(frame.height),
        "sequence_id": int(frame.sequence_id),
        "source": frame.source,
        "path": _report_path(output),
        "sha256": _sha256(output),
        "size_bytes": output.stat().st_size,
        "mean_bgr": [
            float(value) for value in image.reshape(-1, 3).mean(axis=0)
        ],
        "gray_stddev": float(gray.std()),
        "non_blank": bool(gray.std() >= 5.0),
        "teaching_color_pixels": teaching_color_pixels,
        "visible_teaching_colors": visible_teaching_colors,
    }
    camera.close()
    return stats


def _normalized(vector: Any) -> np.ndarray:
    value = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(value))
    if norm <= 1e-12:
        raise ValueError("Cannot normalize a zero-length vector")
    return value / norm


def _quaternion_from_rotation_matrix(matrix: np.ndarray) -> list[float]:
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        return [
            float((matrix[2, 1] - matrix[1, 2]) / scale),
            float((matrix[0, 2] - matrix[2, 0]) / scale),
            float((matrix[1, 0] - matrix[0, 1]) / scale),
            float(0.25 * scale),
        ]
    diagonal_index = int(np.argmax(np.diag(matrix)))
    if diagonal_index == 0:
        scale = math.sqrt(
            1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]
        ) * 2.0
        return [
            float(0.25 * scale),
            float((matrix[0, 1] + matrix[1, 0]) / scale),
            float((matrix[0, 2] + matrix[2, 0]) / scale),
            float((matrix[2, 1] - matrix[1, 2]) / scale),
        ]
    if diagonal_index == 1:
        scale = math.sqrt(
            1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]
        ) * 2.0
        return [
            float((matrix[0, 1] + matrix[1, 0]) / scale),
            float(0.25 * scale),
            float((matrix[1, 2] + matrix[2, 1]) / scale),
            float((matrix[0, 2] - matrix[2, 0]) / scale),
        ]
    scale = math.sqrt(
        1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]
    ) * 2.0
    return [
        float((matrix[0, 2] + matrix[2, 0]) / scale),
        float((matrix[1, 2] + matrix[2, 1]) / scale),
        float(0.25 * scale),
        float((matrix[1, 0] - matrix[0, 1]) / scale),
    ]


def _look_at_pose(
    position: list[float],
    target: list[float],
) -> list[float]:
    forward = _normalized(np.asarray(target) - np.asarray(position))
    right = _normalized(np.cross([0.0, 0.0, 1.0], forward))
    up = np.cross(forward, right)
    rotation = np.column_stack([right, up, forward])
    return [
        *[float(value) for value in position],
        *_quaternion_from_rotation_matrix(rotation),
    ]


def _capture_robot_views(
    *,
    sim: Any,
    output_dir: Path,
) -> dict[str, dict[str, Any]]:
    views = {
        "iso": {
            "position_m": [0.48, -0.48, 0.38],
            "target_m": [0.02, 0.05, 0.10],
        },
        "front": {
            "position_m": [0.02, -0.70, 0.30],
            "target_m": [0.02, 0.05, 0.10],
        },
        "side": {
            "position_m": [0.85, 0.05, 0.36],
            "target_m": [0.02, 0.05, 0.12],
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: dict[str, dict[str, Any]] = {}
    for name, specification in views.items():
        sensor = int(
            sim.createVisionSensor(
                2 | 4 | 64 | 128,
                [800, 600, 0, 0],
                [
                    0.05,
                    2.0,
                    math.radians(48.0),
                    0.02,
                    0.0,
                    0.0,
                    0.93,
                    0.94,
                    0.96,
                    0.0,
                    0.0,
                ],
            )
        )
        try:
            sim.setObjectPose(
                sensor,
                _look_at_pose(
                    specification["position_m"],
                    specification["target_m"],
                ),
                sim.handle_world,
            )
            time.sleep(0.1)
            raw, resolution = sim.getVisionSensorImg(sensor)
            width, height = int(resolution[0]), int(resolution[1])
            image_bytes = bytes(raw or b"")
            expected_bytes = width * height * 3
            if len(image_bytes) != expected_bytes:
                raise RuntimeError(
                    f"Robot view {name} returned {len(image_bytes)} "
                    f"bytes; expected {expected_bytes}"
                )
            rgb_bottom_up = np.frombuffer(
                image_bytes,
                dtype=np.uint8,
            ).reshape(height, width, 3)
            image_bgr = np.flipud(rgb_bottom_up)[:, :, ::-1].copy()
            output = output_dir / f"robot-{name}.png"
            ok, encoded = cv2.imencode(".png", image_bgr)
            if not ok:
                raise RuntimeError(
                    f"OpenCV could not encode robot view: {name}"
                )
            encoded.tofile(str(output))
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
            blue_robot_pixels = int(
                (
                    (hsv[:, :, 0] >= 90)
                    & (hsv[:, :, 0] <= 135)
                    & (hsv[:, :, 1] >= 70)
                    & (hsv[:, :, 2] >= 50)
                ).sum()
            )
            rows[name] = {
                "path": _report_path(output),
                "sha256": _sha256(output),
                "size_bytes": output.stat().st_size,
                "width": width,
                "height": height,
                "gray_stddev": float(gray.std()),
                "non_blank": bool(gray.std() >= 5.0),
                "blue_robot_pixels": blue_robot_pixels,
                **specification,
            }
        finally:
            sim.removeObject(sensor)
    return rows


def _inspect_assembly_geometry(
    sim: Any,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    visual_by_segment = {
        item["segment"]: item for item in manifest["robot_visuals"]
    }
    link6 = int(sim.getObject(visual_by_segment["link6"]["path"]))
    link6_bbox, _bbox_pose = sim.getShapeBB(link6)
    visibility = {
        segment: int(
            sim.getObjectInt32Param(
                sim.getObject(visual_by_segment[segment]["path"]),
                sim.objintparam_visibility_layer,
            )
        )
        for segment in (
            "base",
            "link1",
            "link2",
            "link3",
            "link4",
            "link5",
            "link6",
        )
    }
    bl23_visual_handles: list[int] = []
    for segment in (
        "base",
        "link1",
        "link2",
        "link3",
        "link4",
        "link5",
        "link6",
        "tool",
    ):
        handle = int(
            sim.getObject(
                f"/BL23_visual_{segment}",
                {"noError": True},
            )
        )
        if handle >= 0:
            bl23_visual_handles.append(handle)
    tcp_handle = int(sim.getObject(visual_by_segment["tool"]["tcp_path"]))
    suction_components: list[dict[str, Any]] = []
    for path in visual_by_segment["tool"]["component_paths"]:
        handle = int(sim.getObject(path))
        parent = int(sim.getObjectParent(handle))
        suction_components.append(
            {
                "path": path,
                "handle": handle,
                "parent_handle": parent,
                "parent_is_tcp": parent == tcp_handle,
            }
        )
    return {
        "link6_bbox_size_m": [
            float(value) for value in link6_bbox
        ],
        "link6_bbox_max_m": max(float(value) for value in link6_bbox),
        "body_visibility_layers": visibility,
        "bl23_visual_handles": bl23_visual_handles,
        "tcp_handle": tcp_handle,
        "suction_components": suction_components,
    }


def _inspect_teaching_ready_pose(
    sim: Any,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    expected = [
        float(value)
        for value in manifest["teaching_ready_pose"]["joint_angles_deg"]
    ]
    actual = [
        math.degrees(
            float(
                sim.getJointPosition(
                    sim.getObject(f"/BLX_joint{index}")
                )
            )
        )
        for index in range(1, 7)
    ]
    joint_errors = [
        abs(actual_value - expected_value)
        for actual_value, expected_value in zip(actual, expected)
    ]
    return {
        "expected_joint_angles_deg": expected,
        "actual_joint_angles_deg": actual,
        "joint_errors_deg": joint_errors,
        "max_joint_error_deg": max(joint_errors),
    }


def verify_scene(
    *,
    report_path: Path,
    frame_path: Path,
    robot_views_dir: Path,
    host: str,
    port: int,
) -> dict[str, Any]:
    manifest = _load_json(VISION_DIR / "scene_manifest.json")
    scene_path = (VISION_DIR / manifest["scene"]["path"]).resolve()
    if _sha256(scene_path) != manifest["scene"]["sha256"]:
        raise RuntimeError("Scene hash does not match scene_manifest.json")

    client = RemoteAPIClient(host=host, port=port)
    sim = client.require("sim")
    if sim.getSimulationState() != sim.simulation_stopped:
        sim.stopSimulation()
        state = _wait_for_state(
            sim,
            predicate=lambda value: value == sim.simulation_stopped,
        )
        if state != sim.simulation_stopped:
            raise RuntimeError("CoppeliaSim did not stop before verification")
    staged_scene = stage_scene_for_coppeliasim(scene_path)
    sim.loadScene(staged_scene.as_posix())

    required_paths_found: list[str] = []
    for path in manifest["required_paths"]:
        sim.getObject(path)
        required_paths_found.append(path)

    base_pose = [
        float(value)
        for value in sim.getObjectPose(
            sim.getObject("/BLX_base_link"),
            sim.handle_world,
        )
    ]
    base_flipped = not (
        math.dist(base_pose[:3], [0.0, 0.0, 0.0]) < 1e-8
        and min(
            math.dist(base_pose[3:], [0.0, 0.0, 0.0, 1.0]),
            math.dist(base_pose[3:], [0.0, 0.0, 0.0, -1.0]),
        )
        < 1e-8
    )
    joint_motion = _verify_joint_motion(sim, manifest)
    assembly_geometry = _inspect_assembly_geometry(sim, manifest)
    teaching_ready_pose = _inspect_teaching_ready_pose(sim, manifest)

    sim.startSimulation()
    started_state = _wait_for_state(
        sim,
        predicate=lambda value: value != sim.simulation_stopped,
    )
    started = started_state != sim.simulation_stopped
    camera_frame: dict[str, Any]
    try:
        time.sleep(0.2)
        camera_frame = _capture_camera(
            sim=sim,
            client=client,
            output=frame_path,
        )
        robot_views = _capture_robot_views(
            sim=sim,
            output_dir=robot_views_dir,
        )
    finally:
        sim.stopSimulation()
    stopped_state = _wait_for_state(
        sim,
        predicate=lambda value: value == sim.simulation_stopped,
    )
    stopped = stopped_state == sim.simulation_stopped

    gates = {
        "scene_hash_matches": True,
        "all_required_paths_found": set(manifest["required_paths"])
        <= set(required_paths_found),
        "base_not_flipped": not base_flipped,
        "all_six_visuals_follow_joints": len(joint_motion) == 6
        and all(item["follows_joint"] for item in joint_motion),
        "simulation_started": started,
        "simulation_stopped": stopped,
        "camera_frame_non_blank": camera_frame["non_blank"],
        "camera_resolution_matches": [
            camera_frame["width"],
            camera_frame["height"],
        ]
        == manifest["camera"]["resolution"],
        "all_teaching_colors_visible": set(
            camera_frame["visible_teaching_colors"]
        )
        == {"red", "green", "blue", "yellow"},
        "link6_scale_corrected": (
            assembly_geometry["link6_bbox_max_m"] < 0.05
        ),
        "all_reference_body_visuals_visible": all(
            value != 0
            for value in assembly_geometry[
                "body_visibility_layers"
            ].values()
        ),
        "no_bl23_runtime_visuals": (
            assembly_geometry["bl23_visual_handles"] == []
        ),
        "custom_suction_parented_to_tcp": (
            len(assembly_geometry["suction_components"]) == 3
            and all(
                item["parent_is_tcp"]
                for item in assembly_geometry["suction_components"]
            )
        ),
        "teaching_ready_pose_persisted": (
            teaching_ready_pose["max_joint_error_deg"] < 0.01
        ),
        "three_robot_views_non_blank": (
            set(robot_views) == {"iso", "front", "side"}
            and all(item["non_blank"] for item in robot_views.values())
        ),
        "reference_body_visible_in_three_views": all(
            item["blue_robot_pixels"] >= 5000
            for item in robot_views.values()
        ),
    }
    report = {
        "schema_version": 2,
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "verifier": "simulation.vision_lab.verify_scene",
        "scene_sha256": manifest["scene"]["sha256"],
        "source_step_sha256": manifest["source_step_sha256"],
        "position_only_contract": (
            manifest["contract"]["coordinate_mode"] == "position_only"
            and manifest["contract"]["orientation_ignored"] is True
        ),
        "base_pose_world": base_pose,
        "base_flipped": base_flipped,
        "required_paths_found": required_paths_found,
        "joint_motion": joint_motion,
        "assembly_geometry": assembly_geometry,
        "teaching_ready_pose": teaching_ready_pose,
        "simulation_cycle": {
            "started": started,
            "started_state": started_state,
            "stopped": stopped,
            "stopped_state": stopped_state,
        },
        "camera_frame": camera_frame,
        "robot_views": robot_views,
        "gates": gates,
        "verdict": "PASS" if all(gates.values()) else "FAIL",
    }
    _write_json(report_path, report)
    if report["verdict"] != "PASS":
        failed = [name for name, passed in gates.items() if not passed]
        raise RuntimeError(f"Scene verification failed: {failed}")
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run live CoppeliaSim gates for the BL23 teaching scene",
    )
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--frame", type=Path, default=DEFAULT_FRAME)
    parser.add_argument(
        "--robot-views-dir",
        type=Path,
        default=DEFAULT_ROBOT_VIEWS_DIR,
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=23000)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = verify_scene(
        report_path=args.report.expanduser().resolve(),
        frame_path=args.frame.expanduser().resolve(),
        robot_views_dir=args.robot_views_dir.expanduser().resolve(),
        host=args.host,
        port=args.port,
    )
    print(
        json.dumps(
            {
                "status": report["verdict"],
                "scene_sha256": report["scene_sha256"],
                "joint_gates": len(report["joint_motion"]),
                "camera_frame": report["camera_frame"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
