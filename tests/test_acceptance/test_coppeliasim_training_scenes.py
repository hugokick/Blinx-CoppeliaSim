from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from simulation.training_scenes import verify_scene as scene_verifier
from vision_platform.experiments.scene_setup import LOGISTICS_GROUPS


ROOT = Path(__file__).resolve().parents[2]
FORMAL_SCENES = {
    "simulation/robot_basics": "BL23_robot_basics.ttt",
    "simulation/logistics_lab": "BL23_logistics_lab.ttt",
}


class _FakeSocket:
    def __init__(self) -> None:
        self.closed_with = None

    def close(self, *, linger: int) -> None:
        self.closed_with = linger


class _FakeContext:
    def __init__(self) -> None:
        self.terminated = False

    def term(self) -> None:
        self.terminated = True


class _FakeSim:
    simulation_stopped = 0
    handle_world = -1

    def __init__(self, *, reset_error: Exception | None = None) -> None:
        self.state = self.simulation_stopped
        self.reset_error = reset_error
        self.loaded_scenes: list[str] = []
        self.handles: dict[str, int] = {}
        self.paths_by_handle: dict[int, str] = {}
        self.positions: dict[str, list[float]] = {}
        self.explicit_handling: list[tuple[int, int]] = []
        self.stop_calls = 0

    def getSimulationState(self) -> int:
        return self.state

    def stopSimulation(self) -> None:
        self.stop_calls += 1
        self.state = self.simulation_stopped

    def startSimulation(self) -> None:
        self.state = 1

    def loadScene(self, path: str) -> None:
        self.loaded_scenes.append(path)
        self.state = self.simulation_stopped

    def getObject(self, path: str) -> int:
        if path not in self.handles:
            handle = len(self.handles) + 1
            self.handles[path] = handle
            self.paths_by_handle[handle] = path
            self.positions.setdefault(path, [0.0, 0.0, 0.0])
        return self.handles[path]

    def setObjectPosition(
        self,
        handle: int,
        position: list[float],
        relative_to: int,
    ) -> None:
        assert relative_to == self.handle_world
        self.positions[self.paths_by_handle[handle]] = list(position)

    def getObjectPosition(self, handle: int, relative_to: int):
        assert relative_to == self.handle_world
        return list(self.positions[self.paths_by_handle[handle]])

    def setExplicitHandling(self, handle: int, enabled: int) -> None:
        self.explicit_handling.append((handle, enabled))
        if enabled == 0 and self.reset_error is not None:
            raise self.reset_error

    def handleVisionSensor(self, handle: int) -> None:
        assert handle in self.paths_by_handle


class _FakeClient:
    def __init__(self, sim: _FakeSim) -> None:
        self.sim = sim
        self.socket = _FakeSocket()
        self.context = _FakeContext()

    def require(self, name: str):
        assert name == "sim"
        return self.sim


def _pattern(seed: int) -> np.ndarray:
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    image[20 + seed : 80 + seed, 30:160, seed % 3] = 100 + seed
    return image


def _install_fake_runtime(
    monkeypatch,
    *,
    images: list[np.ndarray] | None = None,
    read_error: Exception | None = None,
    close_error: Exception | None = None,
    reset_error: Exception | None = None,
):
    sim = _FakeSim(reset_error=reset_error)
    client = _FakeClient(sim)
    selected_images = list(images or [_pattern(1)])
    next_image_index = 0

    def client_factory(*, host: str, port: int):
        client.host = host
        client.port = port
        return client

    class FakeCamera:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs
            self.opened = False

        def open(self) -> None:
            self.opened = True

        def read(self, timeout_s: float):
            nonlocal next_image_index
            assert timeout_s == 5.0
            if read_error is not None:
                raise read_error
            image = selected_images[
                min(next_image_index, len(selected_images) - 1)
            ]
            next_image_index += 1
            return SimpleNamespace(
                image_bgr=image,
                width=image.shape[1],
                height=image.shape[0],
            )

        def close(self) -> None:
            self.opened = False
            if close_error is not None:
                raise close_error

    monkeypatch.setattr(scene_verifier, "RemoteAPIClient", client_factory)
    monkeypatch.setattr(scene_verifier, "CoppeliaSimCamera", FakeCamera)
    monkeypatch.setattr(scene_verifier.time, "sleep", lambda _: None)
    return sim, client


def _endpoint(request) -> tuple[str, int]:
    host = (
        request.config.getoption("--coppelia-host")
        or os.environ.get("COPPELIA_HOST")
        or "127.0.0.1"
    )
    configured_port = request.config.getoption("--coppelia-port")
    port = (
        configured_port
        if configured_port is not None
        else int(os.environ.get("COPPELIA_PORT", "23000"))
    )
    return str(host), int(port)


def test_blank_frame_invalidates_stale_pass_and_exact_evidence(
    tmp_path,
    monkeypatch,
):
    report_path = tmp_path / "robot-basics.json"
    report_path.write_text('{"status":"PASS"}\n', encoding="utf-8")
    evidence_dir = tmp_path / "robot-basics-frames"
    evidence_dir.mkdir()
    stale_frame = evidence_dir / "overview.png"
    stale_frame.write_bytes(b"stale-pass-evidence")
    unrelated = evidence_dir / "instructor-note.txt"
    unrelated.write_text("keep", encoding="utf-8")
    sim, client = _install_fake_runtime(
        monkeypatch,
        images=[np.zeros((480, 640, 3), dtype=np.uint8)],
    )

    with pytest.raises(RuntimeError, match="blank"):
        scene_verifier.verify_training_scene(
            ROOT / "simulation/robot_basics",
            host="192.0.2.10",
            port=23111,
            output=report_path,
        )

    assert not report_path.exists()
    assert not stale_frame.exists()
    assert unrelated.read_text(encoding="utf-8") == "keep"
    assert sim.state == sim.simulation_stopped
    assert sim.explicit_handling[-1][1] == 0
    assert client.socket.closed_with == 0
    assert client.context.terminated is True


def test_uniform_non_gray_frame_is_still_blank():
    image = np.empty((480, 640, 3), dtype=np.uint8)
    image[:] = [10, 20, 30]
    frame = SimpleNamespace(image_bgr=image, width=640, height=480)

    with pytest.raises(RuntimeError, match="spatial"):
        scene_verifier._frame_summary(
            frame,
            expected_resolution=(640, 480),
        )


@pytest.mark.parametrize(
    ("directory", "scene_id", "expected_names"),
    (
        (
            "simulation/robot_basics",
            "robot-basics",
            ("overview.png",),
        ),
        (
            "simulation/logistics_lab",
            "logistics-lab",
            ("group-1.png", "group-2.png", "group-3.png"),
        ),
    ),
    ids=("robot-basics", "logistics-lab"),
)
def test_invalid_endpoint_also_invalidates_stale_pass_and_expected_frames(
    tmp_path,
    directory,
    scene_id,
    expected_names,
):
    report_path = tmp_path / "stale.json"
    report_path.write_text('{"status":"PASS"}\n', encoding="utf-8")
    evidence_dir = tmp_path / f"{scene_id}-frames"
    evidence_dir.mkdir()
    expected_frames = tuple(evidence_dir / name for name in expected_names)
    for frame in expected_frames:
        frame.write_bytes(b"stale-pass-evidence")
    unrelated = evidence_dir / "instructor-note.txt"
    unrelated.write_text("keep", encoding="utf-8")

    with pytest.raises(ValueError, match="port"):
        scene_verifier.verify_training_scene(
            ROOT / directory,
            host="127.0.0.1",
            port=0,
            output=report_path,
        )

    assert not report_path.exists()
    assert all(not frame.exists() for frame in expected_frames)
    assert unrelated.read_text(encoding="utf-8") == "keep"


def test_invalid_endpoint_with_explicit_spec_invalidates_expected_frame(
    tmp_path,
):
    report_path = tmp_path / "stale.json"
    report_path.write_text('{"status":"PASS"}\n', encoding="utf-8")
    evidence_dir = tmp_path / "robot-basics-frames"
    evidence_dir.mkdir()
    expected_frame = evidence_dir / "overview.png"
    expected_frame.write_bytes(b"stale-pass-evidence")

    with pytest.raises(ValueError, match="port"):
        scene_verifier.verify_training_scene(
            scene=ROOT
            / "simulation/robot_basics/BL23_robot_basics.ttt",
            spec=ROOT / "simulation/robot_basics/scene_spec.json",
            host="127.0.0.1",
            port=0,
            output=report_path,
        )

    assert not report_path.exists()
    assert not expected_frame.exists()


@pytest.mark.parametrize(
    "selector",
    (
        "directory-with-wrong-scene",
        "spec-with-wrong-scene",
        "spec-with-wrong-manifest",
    ),
)
def test_conflicting_explicit_input_still_invalidates_formal_expected_frame(
    tmp_path,
    selector,
):
    report_path = tmp_path / "stale.json"
    report_path.write_text('{"status":"PASS"}\n', encoding="utf-8")
    evidence_dir = tmp_path / "robot-basics-frames"
    evidence_dir.mkdir()
    expected_frame = evidence_dir / "overview.png"
    expected_frame.write_bytes(b"stale-pass-evidence")
    unrelated = evidence_dir / "instructor-note.txt"
    unrelated.write_text("keep", encoding="utf-8")

    wrong_scene = ROOT / "simulation/logistics_lab/BL23_logistics_lab.ttt"
    kwargs = {
        "scene_directory": ROOT / "simulation/robot_basics",
        "scene": wrong_scene,
    }
    expected_error = "explicit scene does not match"
    if selector.startswith("spec-"):
        kwargs = {
            "spec": ROOT / "simulation/robot_basics/scene_spec.json",
            "scene": (
                wrong_scene
                if selector == "spec-with-wrong-scene"
                else ROOT / "simulation/robot_basics/BL23_robot_basics.ttt"
            ),
        }
    if selector == "spec-with-wrong-manifest":
        kwargs["manifest"] = (
            ROOT / "simulation/logistics_lab/scene_manifest.json"
        )
        expected_error = "scene_id"

    with pytest.raises(ValueError, match=expected_error):
        scene_verifier.verify_training_scene(
            host="127.0.0.1",
            port=23000,
            output=report_path,
            **kwargs,
        )

    assert not report_path.exists()
    assert not expected_frame.exists()
    assert unrelated.read_text(encoding="utf-8") == "keep"


def test_invalid_endpoint_does_not_guess_evidence_for_unknown_scene(tmp_path):
    report_path = tmp_path / "stale.json"
    report_path.write_text('{"status":"PASS"}\n', encoding="utf-8")
    evidence_dir = tmp_path / "robot-basics-frames"
    evidence_dir.mkdir()
    unrelated_formal_frame = evidence_dir / "overview.png"
    unrelated_formal_frame.write_bytes(b"belongs-to-another-run")

    with pytest.raises(ValueError, match="port"):
        scene_verifier.verify_training_scene(
            tmp_path / "unknown-scene",
            host="127.0.0.1",
            port=0,
            output=report_path,
        )

    assert not report_path.exists()
    assert unrelated_formal_frame.read_bytes() == b"belongs-to-another-run"


def test_capture_error_remains_primary_when_camera_cleanup_fails(
    tmp_path,
    monkeypatch,
):
    report_path = tmp_path / "report.json"
    report_path.write_text('{"status":"PASS"}\n', encoding="utf-8")
    _, client = _install_fake_runtime(
        monkeypatch,
        read_error=RuntimeError("capture exploded"),
        close_error=RuntimeError("camera close exploded"),
        reset_error=RuntimeError("explicit reset exploded"),
    )

    with pytest.raises(RuntimeError, match="capture exploded") as caught:
        scene_verifier.verify_training_scene(
            ROOT / "simulation/robot_basics",
            host="127.0.0.1",
            port=23000,
            output=report_path,
        )

    assert caught.value.__cause__ is not None
    assert "camera close exploded" in str(caught.value.__cause__)
    assert "explicit reset exploded" in str(caught.value.__cause__)
    assert not report_path.exists()
    assert client.socket.closed_with == 0
    assert client.context.terminated is True


def test_unprintable_cleanup_error_cannot_replace_primary_error(
    tmp_path,
    monkeypatch,
):
    class UnprintableCleanupError(RuntimeError):
        def __str__(self) -> str:
            raise RuntimeError("cleanup formatting exploded")

    _install_fake_runtime(
        monkeypatch,
        read_error=RuntimeError("capture remains primary"),
        close_error=UnprintableCleanupError(),
    )

    with pytest.raises(RuntimeError, match="capture remains primary") as caught:
        scene_verifier.verify_training_scene(
            ROOT / "simulation/robot_basics",
            host="127.0.0.1",
            port=23000,
            output=tmp_path / "report.json",
        )

    assert caught.value.__cause__ is not None
    assert "UnprintableCleanupError" in str(caught.value.__cause__)
    assert "unprintable" in str(caught.value.__cause__).lower()


def test_new_cleanup_failure_preserves_existing_primary_cause_and_traceback():
    class UnprintableOriginalCause(RuntimeError):
        def __str__(self) -> str:
            raise RuntimeError("original cause formatting exploded")

    original_cause = UnprintableOriginalCause()

    def raise_primary() -> None:
        try:
            raise original_cause
        except UnprintableOriginalCause as error:
            raise LookupError("primary failure") from error

    try:
        raise_primary()
    except LookupError as error:
        primary = error
        primary_traceback = error.__traceback__
    else:
        raise AssertionError("primary fixture did not raise")

    cleanup_failure = RuntimeError("final cleanup exploded")
    with pytest.raises(LookupError, match="primary failure") as caught:
        scene_verifier._raise_with_cleanup(
            (primary, primary_traceback),
            [("final stop", cleanup_failure)],
        )

    assert caught.value is primary
    traceback_names = []
    traceback = caught.value.__traceback__
    while traceback is not None:
        traceback_names.append(traceback.tb_frame.f_code.co_name)
        traceback = traceback.tb_next
    assert "raise_primary" in traceback_names
    cleanup_cause = caught.value.__cause__
    assert isinstance(cleanup_cause, scene_verifier.SceneCleanupError)
    assert "final stop" in str(cleanup_cause)
    assert "final cleanup exploded" in str(cleanup_cause)
    assert cleanup_cause.__cause__ is original_cause


def test_primary_error_links_final_stop_and_client_close_failures(
    tmp_path,
    monkeypatch,
):
    sim, _ = _install_fake_runtime(
        monkeypatch,
        images=[np.zeros((480, 640, 3), dtype=np.uint8)],
    )

    def stop_failure():
        raise RuntimeError("final stop exploded")

    sim.stopSimulation = stop_failure
    monkeypatch.setattr(
        scene_verifier,
        "close_remote_client",
        lambda client: (_ for _ in ()).throw(
            RuntimeError("client close exploded")
        ),
    )

    with pytest.raises(RuntimeError, match="blank") as caught:
        scene_verifier.verify_training_scene(
            ROOT / "simulation/robot_basics",
            host="127.0.0.1",
            port=23000,
            output=tmp_path / "report.json",
        )

    assert caught.value.__cause__ is not None
    assert "final stop exploded" in str(caught.value.__cause__)
    assert "client close exploded" in str(caught.value.__cause__)


def test_logistics_uses_shared_group_activation_and_atomic_report(
    tmp_path,
    monkeypatch,
):
    sim, client = _install_fake_runtime(
        monkeypatch,
        images=[_pattern(1), _pattern(2), _pattern(3)],
    )
    activated: list[str | None] = []

    def activate(simulator, *, active_path):
        activated.append(active_path)
        parked = dict(zip(LOGISTICS_GROUPS, (-2.0, -3.0, -4.0)))
        for group in LOGISTICS_GROUPS:
            handle = simulator.getObject(group)
            simulator.setObjectPosition(
                handle,
                [0.0, 0.0, 0.0 if group == active_path else parked[group]],
                simulator.handle_world,
            )

    monkeypatch.setattr(
        scene_verifier,
        "activate_scene_group",
        activate,
        raising=False,
    )
    replace_calls: list[tuple[Path, Path]] = []
    original_replace = Path.replace

    def recording_replace(source: Path, target: Path):
        replace_calls.append((Path(source), Path(target)))
        return original_replace(source, target)

    monkeypatch.setattr(Path, "replace", recording_replace)
    report_path = tmp_path / "logistics.json"
    report = scene_verifier.verify_training_scene(
        ROOT / "simulation/logistics_lab",
        host="198.51.100.8",
        port=23007,
        output=report_path,
    )

    assert activated == list(LOGISTICS_GROUPS)
    assert any(
        source.parent == report_path.parent and target == report_path
        for source, target in replace_calls
    )
    assert json.loads(report_path.read_text(encoding="utf-8")) == report
    assert len({item["sha256"] for item in report["captures"]}) == 3
    assert client.socket.closed_with == 0
    assert client.context.terminated is True
    assert sim.state == sim.simulation_stopped


def test_training_scene_parser_accepts_explicit_formal_paths_and_endpoint():
    parser_factory = getattr(scene_verifier, "_parser", None)
    assert callable(parser_factory)
    args = parser_factory().parse_args(
        [
            "--scene",
            "simulation/robot_basics/BL23_robot_basics.ttt",
            "--spec",
            "simulation/robot_basics/scene_spec.json",
            "--report",
            "artifacts/robot-basics.json",
            "--host",
            "203.0.113.7",
            "--port",
            "23009",
        ]
    )

    assert args.scene == Path(
        "simulation/robot_basics/BL23_robot_basics.ttt"
    )
    assert args.spec == Path("simulation/robot_basics/scene_spec.json")
    assert args.report == Path("artifacts/robot-basics.json")
    assert args.host == "203.0.113.7"
    assert args.port == 23009


def test_acceptance_script_has_exact_seven_item_first_batch_gate():
    source = (ROOT / "tools/vision_lab/run_acceptance.ps1").read_text(
        encoding="utf-8"
    )

    required = (
        'Invoke-CheckedPython -Name "v2_2_first_batch_online"',
        '"tests/test_acceptance/test_coppeliasim_training_scenes.py"',
        '"tests/test_acceptance/test_coppeliasim_r1_experiments.py"',
        '"--coppelia-host", $HostAddress',
        '"--coppelia-port", [string]$Port',
        "v2-2-first-batch.xml",
        "Assert-JUnitNoSkips",
        "-ExpectedTests 7",
        "failures",
        "errors",
        'v2_2_first_batch = $V22FirstBatchStatus',
        'r1_experiments = @(',
        '"R1-01"',
        '"R1-02"',
        '"R1-05"',
        '"R1-06"',
        '"R1-07"',
    )
    for item in required:
        assert item in source

    gate_start = source.index(
        'Invoke-CheckedPython -Name "v2_2_first_batch_online"'
    )
    gate_assertion = source.index("Assert-JUnitNoSkips", gate_start)
    gate_pass = source.index("$V22FirstBatchGatePassed = $true", gate_start)
    gate_block = source[gate_start:gate_assertion]
    assert '"--coppelia-host", $HostAddress' in gate_block
    assert '"--coppelia-port", [string]$Port' in gate_block
    assert gate_start < gate_assertion < gate_pass
    assert "-ExpectedTests 7" in source[gate_assertion:gate_pass]
    assert source.rindex("Stop-ExactOwnedProcess") < source.index(
        "$V22FirstBatchStatus = if"
    )


@pytest.mark.coppeliasim
@pytest.mark.parametrize(
    "directory",
    FORMAL_SCENES,
    ids=("robot-basics", "logistics-lab"),
)
def test_training_scene_loads_paths_captures_and_reloads(
    directory,
    tmp_path,
    request,
):
    host, port = _endpoint(request)
    report_path = tmp_path / f"{Path(directory).name}.json"
    report = scene_verifier.verify_training_scene(
        ROOT / directory,
        host=host,
        port=port,
        output=report_path,
    )
    spec = json.loads(
        (ROOT / directory / "scene_spec.json").read_text(encoding="utf-8")
    )

    assert report["status"] == "PASS"
    assert report["hardware_status"] == "PENDING_HARDWARE"
    assert report["endpoint"] == {"host": host, "port": port}
    assert report["reload_verified"] is True
    assert Path(report["scene"]).resolve() == (
        ROOT / directory / FORMAL_SCENES[directory]
    ).resolve()
    assert Path(report["spec"]).resolve() == (
        ROOT / directory / "scene_spec.json"
    ).resolve()
    assert set(report["required_paths"]) == set(spec["required_paths"])
    assert set(report["reload_required_paths"]) == set(spec["required_paths"])
    assert len(report["required_paths"]) == report["contract"][
        "required_path_count"
    ]
    assert len(report["captures"]) == (
        3 if directory.endswith("logistics_lab") else 1
    )
    assert report_path.is_file()
    assert json.loads(report_path.read_text(encoding="utf-8")) == report
    for capture in report["captures"]:
        image = (report_path.parent / capture["path"]).resolve()
        assert report_path.parent.resolve() in image.parents
        assert image.is_file()
        assert image.stat().st_size > 0
        assert capture["size"] == spec["camera"]["resolution"] == [640, 480]
        assert len(capture["sha256"]) == 64
        assert capture["sha256"] == hashlib.sha256(image.read_bytes()).hexdigest()
        assert capture["frame_summary"]["non_blank"] is True
        assert capture["frame_summary"]["dynamic_range"] > 0

    if directory.endswith("logistics_lab"):
        assert [item["group"] for item in report["captures"]] == list(
            LOGISTICS_GROUPS
        )
        assert len({item["sha256"] for item in report["captures"]}) == 3
        for capture in report["captures"]:
            positions = capture["group_positions_m"]
            assert set(positions) == set(LOGISTICS_GROUPS)
            assert positions[capture["group"]] == pytest.approx([0.0, 0.0, 0.0])
            assert all(
                position[0:2] == pytest.approx([0.0, 0.0])
                and position[2] <= -2.0
                for group, position in positions.items()
                if group != capture["group"]
            )
