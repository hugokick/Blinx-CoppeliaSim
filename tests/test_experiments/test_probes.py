from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from vision_platform.experiments.probes import probe_experiment
from vision_platform.vision_quality.controller import VisionProfileError


ROOT = Path(__file__).resolve().parents[2]
PROFILES = ROOT / "simulation" / "vision_quality_lab" / "profiles.json"
VISION_SCENE = (
    ROOT
    / "simulation"
    / "vision_quality_lab"
    / "BL23_vision_quality_lab.ttt"
)
VALID_PROFILE_ENTRY = {
    "path": "simulation/vision_quality_lab/profiles.json",
    "sha256": hashlib.sha256(PROFILES.read_bytes()).hexdigest(),
}


class FakeSim:
    handle_world = -1

    def __init__(self, positions=None):
        self.positions = dict(positions or {})
        self.object_calls: list[str] = []

    def getObject(self, path):
        self.object_calls.append(path)
        if path not in self.positions:
            raise RuntimeError(path)
        return path

    def getObjectPosition(self, handle, relative_to):
        assert relative_to == self.handle_world
        return self.positions[handle]


class FakeVisionProfileSim:
    visionintparam_resolution_x = 1002
    visionintparam_resolution_y = 1003
    visionfloatparam_perspective_angle = 1004
    visionfloatparam_near_clipping = 1000
    visionfloatparam_far_clipping = 1001
    handle_world = -1

    def __init__(self) -> None:
        self.handles = {
            "/VisionQualityLab/CameraRig/Camera": 1,
            "/VisionQualityLab/CameraRig": 2,
            "/VisionQualityLab/Lighting/KeyLight": 3,
            "/VisionQualityLab/Lighting/FillLight": 4,
        }
        self.object_calls: list[str] = []
        self.int_params = {(1, 1002): 512, (1, 1003): 512}
        self.float_params = {
            (1, 1004): math.radians(60.0),
            (1, 1000): 0.05,
            (1, 1001): 2.0,
        }
        self.positions = {2: [0.35, -0.125, 0.70]}
        self.lights = {
            3: (1, [0.8, 0.8, 0.8], [0.1, 0.2, 0.3]),
            4: (1, [0.35, 0.35, 0.35], [0.4, 0.5, 0.6]),
        }

    @classmethod
    def standard(cls):
        return cls()

    @classmethod
    def wide_dim(cls):
        sim = cls()
        sim.int_params[(1, 1002)] = 256
        sim.int_params[(1, 1003)] = 256
        sim.float_params[(1, 1004)] = math.radians(75.0)
        sim.positions[2][2] = 0.80
        sim.lights[3] = (1, [0.35, 0.35, 0.35], [0.1, 0.2, 0.3])
        sim.lights[4] = (1, [0.15, 0.15, 0.15], [0.4, 0.5, 0.6])
        return sim

    def getObject(self, path):
        self.object_calls.append(path)
        return self.handles[path]

    def getObjectInt32Param(self, handle, parameter):
        return self.int_params[(handle, parameter)]

    def getObjectFloatParam(self, handle, parameter):
        return self.float_params[(handle, parameter)]

    def getObjectPosition(self, handle, relative_to):
        assert relative_to == self.handle_world
        return list(self.positions[handle])

    def getLightParameters(self, handle):
        state, diffuse, specular = self.lights[handle]
        return state, [0.0, 0.0, 0.0], list(diffuse), list(specular)


class FlippingVisionProfileSim(FakeVisionProfileSim):
    def __init__(self) -> None:
        super().__init__()
        self.light_read_count = 0

    def getLightParameters(self, handle):
        self.light_read_count += 1
        state, reserved, diffuse, specular = super().getLightParameters(handle)
        if self.light_read_count <= 2:
            state = 0
        return state, reserved, diffuse, specular


def _definition(experiment_id, probe_kind, parameters):
    return SimpleNamespace(
        experiment_id=experiment_id,
        public_parameters=parameters,
        acceptance=SimpleNamespace(probe_kind=probe_kind),
        scene=VISION_SCENE,
    )


def _manifest(task_contracts=None):
    return {"task_contracts": dict(task_contracts or {})}


def _stack_definition(slots_mm):
    return _definition(
        "R1-05",
        "stack_2x3",
        {
            "scene_group_path": "/LogisticsLab/Tasks/Stack",
            "stack_slots_mm": slots_mm,
        },
    )


def test_stack_probe_accepts_two_columns_and_three_layers():
    slots_mm = [
        [118, -45, 20],
        [118, 45, 20],
        [118, -45, 38],
        [118, 45, 38],
        [118, -45, 56],
        [118, 45, 56],
    ]
    positions = {
        f"/LogisticsLab/Tasks/Stack/Pickables/stack_{index:02d}": [
            value / 1000 for value in slot
        ]
        for index, slot in enumerate(slots_mm, start=1)
    }

    report = probe_experiment(
        FakeSim(positions),
        _stack_definition(slots_mm),
        phase="final",
        scene_manifest=_manifest(),
    )

    assert report["status"] == "PASS"
    assert report["matched"] == 6
    assert report["expected"] == 6
    assert report["hardware_status"] == "PENDING_HARDWARE"
    assert all("expected_mm" in row for row in report["rows"])
    assert all("distance_mm" in row for row in report["rows"])
    assert json.loads(json.dumps(report, allow_nan=False)) == report
    forbidden = {"grade", "score", "teaching_effectiveness"}
    assert forbidden.isdisjoint(report)


def test_stack_probe_uses_maximum_matching_instead_of_greedy_nearest():
    slots_mm = [
        [0, 0, 0],
        [10, 0, 0],
        [100, 0, 0],
        [200, 0, 0],
        [300, 0, 0],
        [400, 0, 0],
    ]
    actual_mm = [
        [5, 0, 0],  # can use either of the first two slots
        [0, 0, 0],  # can only use the first slot at tolerance 6
        *slots_mm[2:],
    ]
    positions = {
        f"/LogisticsLab/Tasks/Stack/Pickables/stack_{index:02d}": [
            value / 1000 for value in position
        ]
        for index, position in enumerate(actual_mm, start=1)
    }

    report = probe_experiment(
        FakeSim(positions),
        _stack_definition(slots_mm),
        phase="final",
        scene_manifest=_manifest(),
        tolerance_mm=6,
    )

    assert report["status"] == "PASS"
    assert report["matched"] == 6
    assert report["rows"][0]["expected_mm"] == [10.0, 0.0, 0.0]
    assert report["rows"][1]["expected_mm"] == [0.0, 0.0, 0.0]


def test_class_probe_reports_misrouted_object_without_assigning_grade():
    positions = {
        "/LogisticsLab/Tasks/Classes/Pickables/red_block": [0.118, 0.060, 0.020],
        "/LogisticsLab/Tasks/Classes/Pickables/blue_block": [0.118, -0.020, 0.020],
        "/LogisticsLab/Tasks/Classes/Pickables/green_cylinder": [0.118, 0.020, 0.020],
        "/LogisticsLab/Tasks/Classes/Pickables/yellow_cylinder": [0.118, -0.060, 0.020],
    }
    definition = _definition(
        "R1-07",
        "class_zones",
        {
            "scene_group_path": "/LogisticsLab/Tasks/Classes",
            "classes": [
                "red_block",
                "blue_block",
                "green_cylinder",
                "yellow_cylinder",
            ],
            "drop_poses_mm": {
                "red_block": [118, -60, 20],
                "blue_block": [118, -20, 20],
                "green_cylinder": [118, 20, 20],
                "yellow_cylinder": [118, 60, 20],
            },
        },
    )

    report = probe_experiment(
        FakeSim(positions),
        definition,
        phase="final",
        scene_manifest=_manifest(),
    )

    assert report["status"] == "FAIL"
    assert report["matched"] == 2
    assert "grade" not in report
    assert "score" not in report


def test_initial_probe_uses_only_selected_manifested_reset_contract():
    objects = [
        {
            "alias": f"stack_{index:02d}",
            "position_mm": [40 + index * 10, -60, 20],
        }
        for index in range(1, 7)
    ]
    positions = {
        f"/LogisticsLab/Tasks/Stack/Pickables/{item['alias']}": [
            value / 1000 for value in item["position_mm"]
        ]
        for item in objects
    }
    sim = FakeSim(positions)

    report = probe_experiment(
        sim,
        _stack_definition([[118, -45, 20]] * 6),
        phase="initial",
        scene_manifest=_manifest(
            {
                "Stack": {"objects": objects},
                # An inactive task is deliberately malformed and must not be
                # consulted as part of the selected task reset contract.
                "Digits": {"objects": "not-a-list"},
            }
        ),
    )

    assert report["status"] == "PASS"
    assert report["matched"] == 6
    assert sim.object_calls == [
        f"/LogisticsLab/Tasks/Stack/Pickables/stack_{index:02d}"
        for index in range(1, 7)
    ]


def test_motion_probe_uses_only_fixed_joint_and_tool_paths():
    paths = [f"/BLX_joint{index}" for index in range(1, 7)]
    paths.append("/BLX_tool_suction")
    sim = FakeSim({path: None for path in paths})
    definition = _definition("R1-01", "motion_observation", {})

    report = probe_experiment(
        sim,
        definition,
        phase="final",
        scene_manifest=_manifest(),
    )

    assert report["status"] == "PASS"
    assert report["matched"] == 7
    assert sim.object_calls == paths


@pytest.mark.parametrize(
    "phase,tolerance",
    [
        ("middle", 6),
        (1, 6),
        ("final", True),
        ("final", 0),
        ("final", -1),
        ("final", float("nan")),
        ("final", float("inf")),
        ("final", "6"),
    ],
)
def test_probe_rejects_invalid_phase_and_tolerance(phase, tolerance):
    with pytest.raises(ValueError):
        probe_experiment(
            FakeSim(),
            _definition("R1-01", "motion_observation", {}),
            phase=phase,
            scene_manifest=_manifest(),
            tolerance_mm=tolerance,
        )


@pytest.mark.parametrize(
    "position",
    [
        [0.1, 0.2],
        [0.1, 0.2, True],
        [0.1, 0.2, float("nan")],
        [0.1, 0.2, float("inf")],
        "0.1,0.2,0.3",
    ],
)
def test_probe_rejects_invalid_sim_positions_as_value_errors(position):
    slots = [[118, -45, 20]] * 6
    positions = {
        f"/LogisticsLab/Tasks/Stack/Pickables/stack_{index:02d}": [0, 0, 0]
        for index in range(1, 7)
    }
    positions["/LogisticsLab/Tasks/Stack/Pickables/stack_01"] = position

    with pytest.raises(ValueError, match="position"):
        probe_experiment(
            FakeSim(positions),
            _stack_definition(slots),
            phase="final",
            scene_manifest=_manifest(),
        )


def test_probe_rejects_finite_metres_that_overflow_world_mm_conversion():
    slots = [
        [118, -45, 20],
        [118, 45, 20],
        [118, -45, 38],
        [118, 45, 38],
        [118, -45, 56],
        [118, 45, 56],
    ]
    positions = {
        f"/LogisticsLab/Tasks/Stack/Pickables/stack_{index:02d}": [
            value / 1000 for value in slot
        ]
        for index, slot in enumerate(slots, start=1)
    }
    positions["/LogisticsLab/Tasks/Stack/Pickables/stack_01"] = [
        1e308,
        0.0,
        0.0,
    ]

    with pytest.raises(ValueError, match="world position_mm must be finite"):
        probe_experiment(
            FakeSim(positions),
            _stack_definition(slots),
            phase="final",
            scene_manifest=_manifest(),
        )


def test_probe_rejects_distance_overflow_between_finite_endpoints():
    slots = [[-1e308, 0.0, 0.0] for _ in range(6)]
    positions = {
        f"/LogisticsLab/Tasks/Stack/Pickables/stack_{index:02d}": [
            1e305 if index == 1 else -1e305,
            0.0,
            0.0,
        ]
        for index in range(1, 7)
    }

    with pytest.raises(ValueError, match="distance_mm must be finite"):
        probe_experiment(
            FakeSim(positions),
            _stack_definition(slots),
            phase="final",
            scene_manifest=_manifest(),
        )


@pytest.mark.parametrize(
    "definition,manifest",
    [
        (_definition("R1-05", "stack_2x3", {}), _manifest()),
        (_definition("R1-05", "stack_2x3", {"scene_group_path": 1}), _manifest()),
        (_definition("R1-05", "stack_2x3", {"scene_group_path": "/LogisticsLab/Tasks/Stack"}), _manifest()),
        (_stack_definition([[1, 2, 3]] * 6), {}),
        (_stack_definition([[1, 2, 3]] * 6), {"task_contracts": []}),
    ],
)
def test_probe_rejects_missing_or_wrong_typed_contracts_with_value_error(
    definition,
    manifest,
):
    with pytest.raises(ValueError):
        probe_experiment(
            FakeSim(),
            definition,
            phase="initial",
            scene_manifest=manifest,
        )


def test_probe_rejects_unknown_kind_without_dynamic_dispatch():
    sim = FakeSim()
    with pytest.raises(ValueError, match="unsupported probe_kind"):
        probe_experiment(
            sim,
            _definition("R1-99", "__getattribute__", {}),
            phase="final",
            scene_manifest=_manifest(),
        )
    assert sim.object_calls == []


def _vision_definition():
    return _definition(
        "V1-01",
        "vision_profile_observation",
        {
            "baseline_profile_id": "standard",
            "allowed_profile_ids": [
                "standard",
                "wide_dim",
                "detail_bright",
            ],
            "camera_path": "/VisionQualityLab/CameraRig/Camera",
        },
    )


def _vision_manifest(profile_entry=None):
    return {
        "task_contracts": {},
        "profile_catalog": dict(
            VALID_PROFILE_ENTRY if profile_entry is None else profile_entry
        ),
    }


def test_vision_profile_probe_reports_exact_standard_state():
    report = probe_experiment(
        FakeVisionProfileSim.standard(),
        _vision_definition(),
        phase="initial",
        scene_manifest=_vision_manifest(),
    )

    assert report["status"] == "PASS"
    assert report["matched"] == report["expected"] == 4
    assert report["profile_id"] == "standard"
    assert report["hardware_status"] == "PENDING_HARDWARE"
    assert report["profile"] == {
        "profile_id": "standard",
        "resolution": [512, 512],
        "perspective_angle_deg": 60,
        "camera_rig_z_m": 0.7,
        "key_diffuse_rgb": [0.8, 0.8, 0.8],
        "fill_diffuse_rgb": [0.35, 0.35, 0.35],
    }
    assert [row["component"] for row in report["rows"]] == [
        "sensor",
        "camera_rig",
        "key_light",
        "fill_light",
    ]
    assert report["rows"][2]["state"] == 1
    assert report["rows"][2]["enabled"] is True
    assert report["rows"][3]["state"] == 1
    assert report["rows"][3]["enabled"] is True
    assert all(row["matched"] is True for row in report["rows"])
    assert json.loads(json.dumps(report, allow_nan=False)) == report


def test_vision_profile_probe_reports_fail_for_unknown_parameter_combination():
    sim = FakeVisionProfileSim.standard()
    sim.float_params[(1, sim.visionfloatparam_perspective_angle)] = math.radians(
        61.0
    )

    report = probe_experiment(
        sim,
        _vision_definition(),
        phase="final",
        scene_manifest=_vision_manifest(),
    )

    assert report["status"] == "FAIL"
    assert report["matched"] == 0
    assert report["expected"] == 4
    assert report["profile_id"] is None
    assert report["profile"] is None
    assert report["rows"][0]["resolution"] == [512, 512]
    assert report["rows"][0]["perspective_angle_deg"] == pytest.approx(
        61.0
    )
    assert report["rows"][1]["camera_rig_z_m"] == pytest.approx(0.7)
    assert report["rows"][2]["diffuse_rgb"] == [0.8, 0.8, 0.8]
    assert report["rows"][3]["diffuse_rgb"] == [0.35, 0.35, 0.35]
    assert all(row["matched"] is False for row in report["rows"])


@pytest.mark.parametrize("phase", ["initial", "final"])
def test_vision_profile_probe_requires_standard_baseline_for_both_phases(
    phase,
):
    report = probe_experiment(
        FakeVisionProfileSim.wide_dim(),
        _vision_definition(),
        phase=phase,
        scene_manifest=_vision_manifest(),
    )

    assert report["status"] == "FAIL"
    assert report["profile_id"] == "wide_dim"
    assert report["matched"] == 0
    assert report["expected"] == 4


def test_vision_profile_probe_raises_when_a_fixed_light_path_is_missing():
    sim = FakeVisionProfileSim.standard()
    del sim.handles["/VisionQualityLab/Lighting/FillLight"]

    with pytest.raises(VisionProfileError) as raised:
        probe_experiment(
            sim,
            _vision_definition(),
            phase="initial",
            scene_manifest=_vision_manifest(),
        )

    assert raised.value.code == "VISION_PROFILE_OBJECT_MISSING"


@pytest.mark.parametrize("phase", ["initial", "final"])
@pytest.mark.parametrize(
    ("light_path", "row_index"),
    [
        ("/VisionQualityLab/Lighting/KeyLight", 2),
        ("/VisionQualityLab/Lighting/FillLight", 3),
    ],
)
def test_vision_profile_probe_fails_when_a_published_light_is_disabled(
    phase,
    light_path,
    row_index,
):
    sim = FakeVisionProfileSim.standard()
    handle = sim.handles[light_path]
    _, diffuse, specular = sim.lights[handle]
    sim.lights[handle] = (0, diffuse, specular)

    report = probe_experiment(
        sim,
        _vision_definition(),
        phase=phase,
        scene_manifest=_vision_manifest(),
    )

    assert report["status"] == "FAIL"
    assert report["profile_id"] is None
    assert report["rows"][row_index]["state"] == 0
    assert report["rows"][row_index]["enabled"] is False


def test_vision_profile_probe_uses_one_snapshot_for_match_and_report_rows():
    sim = FlippingVisionProfileSim()

    report = probe_experiment(
        sim,
        _vision_definition(),
        phase="initial",
        scene_manifest=_vision_manifest(),
    )

    assert sim.light_read_count == 2
    assert report["status"] == "FAIL"
    assert report["profile_id"] is None
    assert report["rows"][2]["state"] == 0
    assert report["rows"][3]["state"] == 0
    assert all(row["matched"] is False for row in report["rows"])


@pytest.mark.parametrize(
    "entry",
    [
        {
            "path": "simulation/vision_quality_lab/other.json",
            "sha256": VALID_PROFILE_ENTRY["sha256"],
        },
        {
            "path": VALID_PROFILE_ENTRY["path"],
            "sha256": "0" * 64,
        },
    ],
)
def test_vision_profile_probe_rejects_manifest_path_or_hash_before_readback(
    entry,
):
    sim = FakeVisionProfileSim.standard()

    with pytest.raises(ValueError, match="profile catalog"):
        probe_experiment(
            sim,
            _vision_definition(),
            phase="initial",
            scene_manifest=_vision_manifest(entry),
        )

    assert sim.object_calls == []
