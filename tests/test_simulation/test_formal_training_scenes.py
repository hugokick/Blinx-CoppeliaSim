from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

import simulation.training_scenes.build_scene as scene_builder
from simulation.training_scenes.scene_contract import validate_scene_contract


ROOT = Path(__file__).resolve().parents[2]
SCENES = (
    ROOT / "simulation" / "robot_basics",
    ROOT / "simulation" / "logistics_lab",
)


def test_formal_training_scene_contracts_pass():
    reports = [
        validate_scene_contract(
            directory / "scene_spec.json",
            directory / "scene_manifest.json",
            project_root=ROOT,
        )
        for directory in SCENES
    ]

    assert [item["scene_id"] for item in reports] == [
        "robot-basics",
        "logistics-lab",
    ]
    assert all(item["status"] == "PASS" for item in reports)


def test_training_scenes_have_distinct_outputs_and_roots():
    specs = [
        json.loads(
            (directory / "scene_spec.json").read_text(encoding="utf-8")
        )
        for directory in SCENES
    ]

    assert specs[0]["output"] != specs[1]["output"]
    assert specs[0]["root_path"] == "/RobotBasics"
    assert specs[1]["root_path"] == "/LogisticsLab"
    assert all(
        item["template"] == "simulation/vision_lab/BL23_vision_lab.ttt"
        for item in specs
    )
    assert all(item["output"] != item["template"] for item in specs)
    assert all(item["remove_paths"] == ["/VisionLab"] for item in specs)


def test_logistics_scene_declares_three_isolated_task_groups():
    spec = json.loads(
        (SCENES[1] / "scene_spec.json").read_text(encoding="utf-8")
    )

    assert tuple(spec["tasks"]) == ("Stack", "Digits", "Classes")
    assert len(spec["tasks"]["Stack"]["objects"]) == 6
    assert [item["digit"] for item in spec["tasks"]["Digits"]["objects"]] == [
        1,
        2,
        3,
    ]
    assert len(spec["tasks"]["Classes"]["objects"]) == 4


def test_logistics_formal_contract_keeps_group_paths_and_non_overlapping_classes():
    spec = json.loads(
        (SCENES[1] / "scene_spec.json").read_text(encoding="utf-8")
    )

    for name in ("Stack", "Digits", "Classes"):
        assert f"/LogisticsLab/Tasks/{name}" in spec["required_paths"]

    objects = {
        item["alias"]: item for item in spec["tasks"]["Classes"]["objects"]
    }
    green = objects["green_cylinder"]
    yellow = objects["yellow_cylinder"]
    assert yellow["position_mm"][1] == -25
    x_overlap = abs(
        green["position_mm"][0] - yellow["position_mm"][0]
    ) < (green["size_mm"][0] + yellow["size_mm"][0]) / 2
    y_separated = abs(
        green["position_mm"][1] - yellow["position_mm"][1]
    ) >= (green["size_mm"][1] + yellow["size_mm"][1]) / 2
    assert x_overlap is True
    assert y_separated is True


def test_digit_groups_plate_last_as_stable_compound_reference(monkeypatch):
    names_by_handle = {}

    def fake_shape(_sim, *, name, **_kwargs):
        handle = len(names_by_handle) + 1
        names_by_handle[handle] = name
        return handle

    class FakeDigitSim:
        def __init__(self):
            self.grouped = None
            self.aliases = []
            self.parents = []

        def groupShapes(self, parts, merge):
            assert merge is False
            self.grouped = list(parts)
            return 99

        def setObjectAlias(self, handle, alias):
            self.aliases.append((handle, alias))

        def setObjectParent(self, handle, parent, keep_in_place):
            self.parents.append((handle, parent, keep_in_place))

    monkeypatch.setattr(scene_builder, "_shape", fake_shape)
    sim = FakeDigitSim()

    compound = scene_builder._digit(
        sim,
        {
            "alias": "digit_2",
            "digit": 2,
            "position_mm": [75, -55, 20],
        },
        parent=42,
    )

    assert [names_by_handle[handle] for handle in sim.grouped] == [
        "digit_2_a",
        "digit_2_b",
        "digit_2_g",
        "digit_2_e",
        "digit_2_d",
        "digit_2_plate",
    ]
    assert compound == 99
    assert sim.aliases == [(99, "digit_2")]
    assert sim.parents == [(99, 42, True)]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _builder_project(tmp_path, monkeypatch, *, scene_id="robot-basics"):
    root = tmp_path / "project"
    root.mkdir()
    template = root / "simulation/vision_lab/BL23_vision_lab.ttt"
    template.parent.mkdir(parents=True)
    template.write_bytes(b"protected-template")

    protected = {
        "robot_backends/models/BLX_openr6.ttt": b"robot-model",
        "robot_backends/models/openr6_arm_coppeliasim.urdf": b"urdf\n",
        "robot_backends/models/meshes_blx/base_link.STL": b"mesh",
    }
    for relative, content in protected.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    visual = root / "simulation/vision_lab/assets/robot/base_visual.STL"
    visual.parent.mkdir(parents=True)
    visual.write_bytes(b"visual")

    source_manifest = root / "simulation/vision_lab/source_manifest.json"
    _write_json(
        source_manifest,
        {
            "schema_version": 1,
            "protected_formal_assets": [
                {
                    "path": relative,
                    "sha256": _sha256(root / relative),
                }
                for relative in protected
            ],
        },
    )

    directory_name = (
        "robot_basics" if scene_id == "robot-basics" else "logistics_lab"
    )
    source_spec = SCENES[0 if scene_id == "robot-basics" else 1] / (
        "scene_spec.json"
    )
    spec_path = root / f"simulation/{directory_name}/scene_spec.json"
    spec_path.parent.mkdir(parents=True)
    spec_path.write_bytes(source_spec.read_bytes())
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    output = root / spec["output"]
    output.write_bytes(b"old-scene")
    manifest = spec_path.parent / "scene_manifest.json"
    manifest.write_text('{"old": true}\n', encoding="utf-8")

    monkeypatch.setattr(scene_builder, "PROJECT_ROOT", root.resolve())
    monkeypatch.setattr(
        scene_builder,
        "SOURCE_MANIFEST",
        source_manifest.resolve(),
    )
    return root, spec_path, output, manifest, template


class _ForbiddenClient:
    calls = 0

    def __init__(self, **_kwargs):
        type(self).calls += 1
        raise AssertionError("CoppeliaSim connection happened before preflight")


class _FakeSim:
    simulation_stopped = 0

    def __init__(self, *, save_error=None, mutate_protected=None):
        self.saved_paths = []
        self.loaded_paths = []
        self.save_error = save_error
        self.mutate_protected = mutate_protected

    def getSimulationState(self):
        return self.simulation_stopped

    def loadScene(self, path):
        self.loaded_paths.append(Path(path))

    def getObject(self, _path):
        return 1

    def removeObjects(self, _handles, _options):
        return None

    def saveScene(self, path):
        staged = Path(path)
        self.saved_paths.append(staged)
        staged.write_bytes(b"new-scene")
        if self.mutate_protected is not None:
            self.mutate_protected.write_bytes(b"tampered")
        if self.save_error is not None:
            raise self.save_error


def _install_fake_sim(monkeypatch, sim, *, close_error=None):
    clients = []

    class FakeClient:
        def __init__(self, **_kwargs):
            clients.append(self)

        def require(self, name):
            assert name == "sim"
            return sim

        def close(self):
            if close_error is not None:
                raise close_error

    monkeypatch.setattr(scene_builder, "RemoteAPIClient", FakeClient)
    monkeypatch.setattr(
        scene_builder,
        "stage_scene_for_coppeliasim",
        lambda path: Path(path),
    )
    monkeypatch.setattr(scene_builder, "_dummy", lambda *_args, **_kwargs: 1)
    monkeypatch.setattr(
        scene_builder, "_build_workspace", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(scene_builder, "_camera", lambda *_args, **_kwargs: 1)
    monkeypatch.setattr(
        scene_builder, "_build_basics", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        scene_builder, "_build_logistics", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        scene_builder,
        "_attach_logistics_camera_scope",
        lambda *_args, **_kwargs: 1,
    )
    return clients


def _write_consistent_old_manifest(spec_path, output, manifest):
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    root = spec_path.parents[2]
    template = root / spec["template"]
    payload = {
        "schema_version": 1,
        "scene_id": spec["scene_id"],
        "template": {
            "path": spec["template"],
            "sha256": _sha256(template),
        },
        "scene": {
            "path": spec["output"],
            "sha256": _sha256(output),
            "size_bytes": output.stat().st_size,
        },
        "required_paths": spec["required_paths"],
        "protected_assets_unchanged": True,
    }
    _write_json(manifest, payload)
    return manifest.read_bytes()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda spec: spec.__setitem__("schema_version", True), "integer 1"),
        (lambda spec: spec.__setitem__("scene_id", "logistics-lab"), "scene_id"),
        (lambda spec: spec.__setitem__("root_path", "/Wrong"), "root_path"),
        (lambda spec: spec.__setitem__("remove_paths", []), "remove_paths"),
        (
            lambda spec: spec.__setitem__(
                "template", "../vision_lab/BL23_vision_lab.ttt"
            ),
            "template",
        ),
        (
            lambda spec: spec.__setitem__(
                "output", "simulation/vision_lab/BL23_vision_lab.ttt"
            ),
            "output",
        ),
        (
            lambda spec: spec["camera"].__setitem__(
                "resolution", [True, 480]
            ),
            "camera resolution",
        ),
        (
            lambda spec: spec["workspace"].__setitem__(
                "size_mm", [True, 200, 10]
            ),
            "workspace size_mm",
        ),
    ],
)
def test_builder_rejects_invalid_formal_specs_before_connecting_or_writing(
    tmp_path,
    monkeypatch,
    mutation,
    message,
):
    _, spec_path, output, manifest, _ = _builder_project(tmp_path, monkeypatch)
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    mutation(spec)
    _write_json(spec_path, spec)
    old_scene = output.read_bytes()
    old_manifest = manifest.read_bytes()
    _ForbiddenClient.calls = 0
    monkeypatch.setattr(scene_builder, "RemoteAPIClient", _ForbiddenClient)

    with pytest.raises(ValueError, match=message):
        scene_builder.build_scene(
            spec_path=spec_path,
            host="127.0.0.1",
            port=23000,
        )

    assert _ForbiddenClient.calls == 0
    assert output.read_bytes() == old_scene
    assert manifest.read_bytes() == old_manifest


def test_builder_rejects_unapproved_spec_path_before_connecting(tmp_path, monkeypatch):
    root, spec_path, output, manifest, _ = _builder_project(
        tmp_path, monkeypatch
    )
    unapproved = root / "simulation/other/scene_spec.json"
    unapproved.parent.mkdir()
    unapproved.write_bytes(spec_path.read_bytes())
    old_scene = output.read_bytes()
    old_manifest = manifest.read_bytes()
    _ForbiddenClient.calls = 0
    monkeypatch.setattr(scene_builder, "RemoteAPIClient", _ForbiddenClient)

    with pytest.raises(ValueError, match="approved formal spec"):
        scene_builder.build_scene(
            spec_path=unapproved,
            host="127.0.0.1",
            port=23000,
        )

    assert _ForbiddenClient.calls == 0
    assert output.read_bytes() == old_scene
    assert manifest.read_bytes() == old_manifest


def test_builder_rejects_hardlinked_formal_output_before_connecting(
    tmp_path, monkeypatch
):
    _, spec_path, output, manifest, template = _builder_project(
        tmp_path, monkeypatch
    )
    output.unlink()
    os.link(template, output)
    old_manifest = manifest.read_bytes()
    _ForbiddenClient.calls = 0
    monkeypatch.setattr(scene_builder, "RemoteAPIClient", _ForbiddenClient)

    with pytest.raises(ValueError, match="alias|hardlink"):
        scene_builder.build_scene(
            spec_path=spec_path,
            host="127.0.0.1",
            port=23000,
        )

    assert _ForbiddenClient.calls == 0
    assert output.samefile(template)
    assert manifest.read_bytes() == old_manifest


def test_builder_rejects_symlinked_path_component_before_connecting(
    tmp_path, monkeypatch
):
    root, spec_path, output, manifest, _ = _builder_project(
        tmp_path, monkeypatch
    )
    original = Path.is_symlink

    def report_alias(path):
        return path == root / "simulation/robot_basics" or original(path)

    monkeypatch.setattr(Path, "is_symlink", report_alias)
    _ForbiddenClient.calls = 0
    monkeypatch.setattr(scene_builder, "RemoteAPIClient", _ForbiddenClient)

    with pytest.raises(ValueError, match="symlink|junction|alias"):
        scene_builder.build_scene(
            spec_path=spec_path,
            host="127.0.0.1",
            port=23000,
        )

    assert _ForbiddenClient.calls == 0
    assert output.read_bytes() == b"old-scene"
    assert manifest.read_text(encoding="utf-8") == '{"old": true}\n'


def test_builder_saves_unique_staged_scene_and_publishes_manifest_last(
    tmp_path, monkeypatch
):
    _, spec_path, output, manifest, template = _builder_project(
        tmp_path, monkeypatch
    )
    sim = _FakeSim()
    _install_fake_sim(monkeypatch, sim)
    replace_calls = []
    real_replace = os.replace

    def recording_replace(source, destination):
        replace_calls.append((Path(source), Path(destination)))
        real_replace(source, destination)

    monkeypatch.setattr(scene_builder.os, "replace", recording_replace)

    result = scene_builder.build_scene(
        spec_path=spec_path,
        host="127.0.0.1",
        port=23000,
    )

    assert sim.loaded_paths == [template]
    assert len(sim.saved_paths) == 1
    staged_scene = sim.saved_paths[0]
    assert staged_scene.parent == output.parent
    assert staged_scene != output
    assert staged_scene.suffix == ".ttt"
    assert output.read_bytes() == b"new-scene"
    assert result["scene"]["sha256"] == _sha256(output)
    assert result["protected_assets_unchanged"] is True
    assert replace_calls[-2][1] == output
    assert replace_calls[-1][1] == manifest
    assert not any(".staged-" in path.name for path in output.parent.iterdir())


def test_builder_failure_cleans_only_own_stage_and_keeps_old_release(
    tmp_path, monkeypatch
):
    _, spec_path, output, manifest, _ = _builder_project(tmp_path, monkeypatch)
    unrelated = output.parent / ".unrelated.staged-scene.ttt"
    unrelated.write_bytes(b"keep")
    sim = _FakeSim(save_error=RuntimeError("simulated save failure"))
    _install_fake_sim(monkeypatch, sim)
    old_manifest = manifest.read_bytes()

    with pytest.raises(RuntimeError, match="simulated save failure"):
        scene_builder.build_scene(
            spec_path=spec_path,
            host="127.0.0.1",
            port=23000,
        )

    assert output.read_bytes() == b"old-scene"
    assert manifest.read_bytes() == old_manifest
    assert unrelated.read_bytes() == b"keep"
    assert len(sim.saved_paths) == 1
    assert not sim.saved_paths[0].exists()


def test_builder_rejects_invalid_source_manifest_before_connecting(
    tmp_path, monkeypatch
):
    root, spec_path, output, manifest, _ = _builder_project(
        tmp_path, monkeypatch
    )
    source_manifest = root / "simulation/vision_lab/source_manifest.json"
    payload = json.loads(source_manifest.read_text(encoding="utf-8"))
    payload["schema_version"] = True
    _write_json(source_manifest, payload)
    _ForbiddenClient.calls = 0
    monkeypatch.setattr(scene_builder, "RemoteAPIClient", _ForbiddenClient)

    with pytest.raises(ValueError, match="schema_version must be integer 1"):
        scene_builder.build_scene(
            spec_path=spec_path,
            host="127.0.0.1",
            port=23000,
        )

    assert _ForbiddenClient.calls == 0
    assert output.read_bytes() == b"old-scene"
    assert manifest.read_text(encoding="utf-8") == '{"old": true}\n'


def test_builder_detects_protected_mutation_before_publishing(
    tmp_path, monkeypatch
):
    root, spec_path, output, manifest, _ = _builder_project(
        tmp_path, monkeypatch
    )
    protected = root / "robot_backends/models/BLX_openr6.ttt"
    sim = _FakeSim(mutate_protected=protected)
    _install_fake_sim(monkeypatch, sim)

    with pytest.raises(RuntimeError, match="protected asset hash mismatch"):
        scene_builder.build_scene(
            spec_path=spec_path,
            host="127.0.0.1",
            port=23000,
        )

    assert output.read_bytes() == b"old-scene"
    assert manifest.read_text(encoding="utf-8") == '{"old": true}\n'
    assert len(sim.saved_paths) == 1
    assert not sim.saved_paths[0].exists()


def test_builder_manifest_publish_failure_never_leaves_old_manifest_for_new_scene(
    tmp_path, monkeypatch
):
    _, spec_path, output, manifest, _ = _builder_project(tmp_path, monkeypatch)
    sim = _FakeSim()
    _install_fake_sim(monkeypatch, sim)
    real_replace = os.replace

    def fail_manifest_replace(source, destination):
        if Path(destination) == manifest:
            raise OSError("manifest publish failed")
        real_replace(source, destination)

    monkeypatch.setattr(scene_builder.os, "replace", fail_manifest_replace)

    with pytest.raises(OSError, match="manifest publish failed"):
        scene_builder.build_scene(
            spec_path=spec_path,
            host="127.0.0.1",
            port=23000,
        )

    assert output.read_bytes() == b"new-scene"
    assert not manifest.exists()
    assert not any(".staged-" in path.name for path in output.parent.iterdir())


def test_builder_fails_fast_when_same_process_already_builds_output(
    tmp_path, monkeypatch
):
    _, spec_path, output, manifest, _ = _builder_project(tmp_path, monkeypatch)
    lock = scene_builder._BuildOutputLock(output)
    lock.acquire()
    _ForbiddenClient.calls = 0
    monkeypatch.setattr(scene_builder, "RemoteAPIClient", _ForbiddenClient)
    try:
        with pytest.raises(RuntimeError, match="build already in progress"):
            scene_builder.build_scene(
                spec_path=spec_path,
                host="127.0.0.1",
                port=23000,
            )
    finally:
        assert lock.release() == []

    assert _ForbiddenClient.calls == 0
    assert output.read_bytes() == b"old-scene"
    assert manifest.read_text(encoding="utf-8") == '{"old": true}\n'


def test_builder_fails_fast_under_cross_process_output_interleaving(
    tmp_path, monkeypatch
):
    spec_path = SCENES[0] / "scene_spec.json"
    output = SCENES[0] / "BL23_robot_basics.ttt"
    manifest = SCENES[0] / "scene_manifest.json"
    old_output = output.read_bytes()
    old_manifest = manifest.read_bytes()
    ready = tmp_path / "lock-ready"
    release = tmp_path / "lock-release"
    child_code = """
import sys
import time
from pathlib import Path
import simulation.training_scenes.build_scene as builder

lock = builder._BuildOutputLock(Path(sys.argv[1]))
lock.acquire()
try:
    Path(sys.argv[2]).write_text('ready', encoding='utf-8')
    deadline = time.monotonic() + 15.0
    while not Path(sys.argv[3]).exists() and time.monotonic() < deadline:
        time.sleep(0.02)
    if not Path(sys.argv[3]).exists():
        raise RuntimeError('parent did not release child lock')
finally:
    errors = lock.release()
    if errors:
        raise RuntimeError('; '.join(errors))
"""
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            child_code,
            str(output),
            str(ready),
            str(release),
        ],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 10.0
        while not ready.exists() and time.monotonic() < deadline:
            if process.poll() is not None:
                stdout, stderr = process.communicate()
                pytest.fail(f"lock child exited early: {stdout} {stderr}")
            time.sleep(0.02)
        assert ready.exists(), "lock child did not become ready"

        _ForbiddenClient.calls = 0
        monkeypatch.setattr(scene_builder, "RemoteAPIClient", _ForbiddenClient)
        monkeypatch.setattr(
            scene_builder,
            "_protected_hashes",
            lambda: pytest.fail("protected snapshot ran outside output lock"),
        )
        with pytest.raises(RuntimeError, match="build already in progress"):
            scene_builder.build_scene(
                spec_path=spec_path,
                host="127.0.0.1",
                port=23000,
            )
        assert _ForbiddenClient.calls == 0
    finally:
        release.write_text("release", encoding="utf-8")
        stdout, stderr = process.communicate(timeout=10)
        assert process.returncode == 0, (stdout, stderr)

    assert output.read_bytes() == old_output
    assert manifest.read_bytes() == old_manifest


def test_output_lock_is_outside_repo_and_released_after_process_crash(tmp_path):
    output = SCENES[0] / "BL23_robot_basics.ttt"
    ready = tmp_path / "crash-lock-ready"
    child_code = """
import os
import sys
from pathlib import Path
import simulation.training_scenes.build_scene as builder

lock = builder._BuildOutputLock(Path(sys.argv[1]))
lock.acquire()
Path(sys.argv[2]).write_text('ready', encoding='utf-8')
os._exit(0)
"""
    process = subprocess.Popen(
        [sys.executable, "-c", child_code, str(output), str(ready)],
        cwd=ROOT,
    )
    assert process.wait(timeout=10) == 0
    assert ready.read_text(encoding="utf-8") == "ready"

    lock = scene_builder._BuildOutputLock(output)
    assert ROOT not in lock._path.parents
    lock.acquire()
    assert lock.release() == []


def test_builder_never_deletes_preexisting_staging_name_collision(
    tmp_path, monkeypatch
):
    _, spec_path, output, manifest, _ = _builder_project(tmp_path, monkeypatch)
    token = "preexisting"
    collision = output.parent / f".{output.stem}.staged-{token}.ttt"
    collision.write_bytes(b"unrelated")

    class FixedUuid:
        hex = token

    monkeypatch.setattr(scene_builder.uuid, "uuid4", lambda: FixedUuid())
    _ForbiddenClient.calls = 0
    monkeypatch.setattr(scene_builder, "RemoteAPIClient", _ForbiddenClient)

    with pytest.raises(RuntimeError, match="staging path already exists"):
        scene_builder.build_scene(
            spec_path=spec_path,
            host="127.0.0.1",
            port=23000,
        )

    assert _ForbiddenClient.calls == 0
    assert collision.read_bytes() == b"unrelated"
    assert output.read_bytes() == b"old-scene"
    assert manifest.read_text(encoding="utf-8") == '{"old": true}\n'


def test_builder_close_error_preserves_primary_error_and_cleans_stage(
    tmp_path, monkeypatch
):
    _, spec_path, output, manifest, _ = _builder_project(tmp_path, monkeypatch)
    sim = _FakeSim(save_error=RuntimeError("primary save failure"))
    _install_fake_sim(
        monkeypatch,
        sim,
        close_error=RuntimeError("client close failure"),
    )

    with pytest.raises(RuntimeError, match="primary save failure") as raised:
        scene_builder.build_scene(
            spec_path=spec_path,
            host="127.0.0.1",
            port=23000,
        )

    assert "client.close" in "\n".join(
        getattr(raised.value, "__notes__", [])
    )
    assert len(sim.saved_paths) == 1
    assert not sim.saved_paths[0].exists()
    assert output.read_bytes() == b"old-scene"
    assert manifest.read_text(encoding="utf-8") == '{"old": true}\n'


def test_builder_temp_cleanup_errors_are_independent_and_preserve_primary(
    tmp_path, monkeypatch
):
    _, spec_path, output, manifest, _ = _builder_project(tmp_path, monkeypatch)
    _write_consistent_old_manifest(spec_path, output, manifest)
    sim = _FakeSim()
    _install_fake_sim(monkeypatch, sim)
    real_replace = os.replace
    real_unlink = Path.unlink

    def fail_scene_replace(source, destination):
        if Path(destination) == output:
            raise PermissionError("scene replace denied")
        real_replace(source, destination)

    failed_stage = []

    def fail_first_stage_unlink(path, *args, **kwargs):
        if path.suffix == ".ttt" and ".staged-" in path.name and not failed_stage:
            failed_stage.append(path)
            raise OSError("first temp unlink failed")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(scene_builder.os, "replace", fail_scene_replace)
    monkeypatch.setattr(Path, "unlink", fail_first_stage_unlink)
    try:
        with pytest.raises(PermissionError, match="scene replace denied") as raised:
            scene_builder.build_scene(
                spec_path=spec_path,
                host="127.0.0.1",
                port=23000,
            )

        assert "first temp unlink failed" in "\n".join(
            getattr(raised.value, "__notes__", [])
        )
        assert failed_stage == sim.saved_paths
        assert failed_stage[0].exists()
        leftovers = [
            path
            for path in output.parent.iterdir()
            if ".staged-" in path.name or ".backup-" in path.name
        ]
        assert leftovers == failed_stage
    finally:
        for path in failed_stage:
            real_unlink(path, missing_ok=True)


def test_builder_success_reports_close_warning_without_failing_commit(
    tmp_path, monkeypatch
):
    _, spec_path, output, manifest, _ = _builder_project(tmp_path, monkeypatch)
    sim = _FakeSim()
    _install_fake_sim(
        monkeypatch,
        sim,
        close_error=RuntimeError("client close failure"),
    )

    with pytest.warns(RuntimeWarning, match="client.close"):
        result = scene_builder.build_scene(
            spec_path=spec_path,
            host="127.0.0.1",
            port=23000,
        )

    assert result["scene"]["sha256"] == _sha256(output)
    assert any("client.close" in item for item in result["cleanup_errors"])
    stored = json.loads(manifest.read_text(encoding="utf-8"))
    assert "cleanup_errors" not in stored
    assert not any(
        ".staged-" in path.name or ".backup-" in path.name
        for path in output.parent.iterdir()
    )


def test_builder_restores_valid_old_manifest_when_scene_replace_fails(
    tmp_path, monkeypatch
):
    _, spec_path, output, manifest, _ = _builder_project(tmp_path, monkeypatch)
    old_manifest = _write_consistent_old_manifest(spec_path, output, manifest)
    sim = _FakeSim()
    _install_fake_sim(monkeypatch, sim)
    real_replace = os.replace

    def fail_scene_replace(source, destination):
        if Path(destination) == output:
            raise PermissionError("scene replace denied")
        real_replace(source, destination)

    monkeypatch.setattr(scene_builder.os, "replace", fail_scene_replace)

    with pytest.raises(PermissionError, match="scene replace denied"):
        scene_builder.build_scene(
            spec_path=spec_path,
            host="127.0.0.1",
            port=23000,
        )

    assert output.read_bytes() == b"old-scene"
    assert manifest.read_bytes() == old_manifest
    assert not any(
        ".staged-" in path.name or ".backup-" in path.name
        for path in output.parent.iterdir()
    )


def test_builder_keeps_manifest_absent_when_scene_replace_state_is_uncertain(
    tmp_path, monkeypatch
):
    _, spec_path, output, manifest, _ = _builder_project(tmp_path, monkeypatch)
    _write_consistent_old_manifest(spec_path, output, manifest)
    sim = _FakeSim()
    _install_fake_sim(monkeypatch, sim)
    real_replace = os.replace

    def mutate_then_fail_scene_replace(source, destination):
        if Path(destination) == output:
            output.write_bytes(b"uncertain-scene")
            raise PermissionError("scene replace state uncertain")
        real_replace(source, destination)

    monkeypatch.setattr(
        scene_builder.os,
        "replace",
        mutate_then_fail_scene_replace,
    )

    with pytest.raises(PermissionError, match="state uncertain"):
        scene_builder.build_scene(
            spec_path=spec_path,
            host="127.0.0.1",
            port=23000,
        )

    assert output.read_bytes() == b"uncertain-scene"
    assert not manifest.exists()
    assert not any(
        ".staged-" in path.name or ".backup-" in path.name
        for path in output.parent.iterdir()
    )


def test_builder_does_not_restore_old_manifest_with_invalid_protection_claim(
    tmp_path, monkeypatch
):
    _, spec_path, output, manifest, _ = _builder_project(tmp_path, monkeypatch)
    _write_consistent_old_manifest(spec_path, output, manifest)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["protected_assets_unchanged"] = False
    _write_json(manifest, payload)
    sim = _FakeSim()
    _install_fake_sim(monkeypatch, sim)
    real_replace = os.replace

    def fail_scene_replace(source, destination):
        if Path(destination) == output:
            raise PermissionError("scene replace denied")
        real_replace(source, destination)

    monkeypatch.setattr(scene_builder.os, "replace", fail_scene_replace)

    with pytest.raises(PermissionError, match="scene replace denied"):
        scene_builder.build_scene(
            spec_path=spec_path,
            host="127.0.0.1",
            port=23000,
        )

    assert output.read_bytes() == b"old-scene"
    assert not manifest.exists()
    assert not any(
        ".staged-" in path.name or ".backup-" in path.name
        for path in output.parent.iterdir()
    )
