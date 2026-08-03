from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from vision_platform.experiments.catalog import ExperimentCatalog
from vision_platform.experiments.probes import (
    probe_experiment,
    probe_ocr_entry,
    probe_ocr_final,
)


ROOT = Path(__file__).resolve().parents[2]


class FakeOcrSim:
    handle_world = -1

    def __init__(self, positions, *, robot_pose=(0.0, 0.0, 0.0), tool_on=False):
        self.positions = dict(positions)
        self.robot_pose = tuple(robot_pose)
        self.tool_on = bool(tool_on)
        self.object_calls: list[str] = []

    def getObject(self, path):
        self.object_calls.append(path)
        if path == "/BLX_tool_suction":
            return path
        if path not in self.positions:
            raise RuntimeError(path)
        return path

    def getObjectPosition(self, handle, relative_to):
        assert relative_to == self.handle_world
        if handle == "/BLX_tool_suction":
            return [value / 1000.0 for value in self.robot_pose]
        return [value / 1000.0 for value in self.positions[handle]]

    def get_tool_state(self):
        return self.tool_on


def _definition():
    return ExperimentCatalog.load(
        ROOT / "config" / "experiments" / "catalog.json",
        project_root=ROOT,
    ).require("V1-08")


def _scene_hash():
    return hashlib.sha256(
        _definition().scene.read_bytes()
    ).hexdigest()


def _positions():
    return {
        "/VisionOcrSortingLab/Parts/part_a": [116.0, -60.0, 22.0],
        "/VisionOcrSortingLab/Parts/part_b": [128.0, -60.0, 22.0],
        "/VisionOcrSortingLab/Parts/part_c": [116.0, 60.0, 22.0],
        "/VisionOcrSortingLab/Parts/part_d": [128.0, 60.0, 22.0],
    }


def _refs(*, run_id="run-v1-08", scene_hash=None):
    scene_hash = scene_hash or _scene_hash()
    return [
        {
            "run_id": run_id,
            "scene_hash": scene_hash,
            "snapshot_id": "frame-000001",
            "entry_id": entry_id,
            "part_id": part_id,
            "route_id": route_id,
            "slot_id": slot_id,
            "evidence_id": f"ocr-entry-{entry_id}-001",
        }
        for entry_id, part_id, route_id, slot_id in (
            ("entry_a", "part_a", "route_alpha", "slot_1"),
            ("entry_b", "part_b", "route_alpha", "slot_2"),
            ("entry_c", "part_c", "route_beta", "slot_1"),
            ("entry_d", "part_d", "route_beta", "slot_2"),
        )
    ]


def test_entry_probe_returns_bounded_same_run_reference_for_configured_slot():
    report = probe_ocr_entry(
        FakeOcrSim(_positions()),
        _definition(),
        entry_id="entry_a",
        run_id="run-v1-08",
        scene_hash=_scene_hash(),
        snapshot_id="frame-000001",
    )

    assert report["status"] == "PASS"
    assert report["part_id"] == "part_a"
    assert report["route_id"] == "route_alpha"
    assert report["slot_id"] == "slot_1"
    assert report["run_id"] == "run-v1-08"
    assert report["scene_hash"] == _scene_hash()
    assert report["snapshot_id"] == "frame-000001"
    assert report["evidence_id"].startswith("ocr-entry-entry_a-")
    assert json.loads(json.dumps(report, allow_nan=False)) == report


@pytest.mark.parametrize(
    "kwargs",
    [
        {"run_id": ""},
        {"scene_hash": "not-a-digest"},
        {"snapshot_id": "bad\nline"},
    ],
)
def test_entry_probe_rejects_invalid_runtime_binding(kwargs):
    base = {
        "entry_id": "entry_a",
        "run_id": "run-v1-08",
        "scene_hash": _scene_hash(),
        "snapshot_id": "frame-000001",
    }
    base.update(kwargs)
    with pytest.raises(ValueError):
        probe_ocr_entry(FakeOcrSim(_positions()), _definition(), **base)


def test_final_probe_requires_exact_occupancy_home_tool_off_and_four_refs():
    report = probe_ocr_final(
        FakeOcrSim(_positions(), robot_pose=(0.0, 0.0, 0.0), tool_on=False),
        _definition(),
        run_id="run-v1-08",
        scene_hash=_scene_hash(),
        snapshot_id="frame-000001",
        entry_evidence=_refs(),
        consumed_entry_ids=("entry_a", "entry_b", "entry_c", "entry_d"),
        robot_home=True,
        tool_on=False,
    )

    assert report["status"] == "PASS"
    assert report["matched"] == report["expected"] == 4
    assert report["final_occupancy"] == {
        "route_alpha": {"slot_1": "part_a", "slot_2": "part_b"},
        "route_beta": {"slot_1": "part_c", "slot_2": "part_d"},
    }
    assert report["same_run_evidence"] is True
    assert report["robot_home"] is True
    assert report["tool_off"] is True
    assert len(report["entry_evidence"]) == 4


@pytest.mark.parametrize(
    "overrides",
    [
        {"robot_home": False},
        {"tool_on": True},
        {"consumed_entry_ids": ("entry_a", "entry_b", "entry_c")},
        {"entry_evidence": _refs()[:3]},
        {"run_id": "other-run"},
        {"scene_hash": "0" * 64},
    ],
)
def test_final_probe_fails_closed_for_incomplete_or_cross_run_evidence(overrides):
    values = {
        "run_id": "run-v1-08",
        "scene_hash": _scene_hash(),
        "snapshot_id": "frame-000001",
        "entry_evidence": _refs(),
        "consumed_entry_ids": ("entry_a", "entry_b", "entry_c", "entry_d"),
        "robot_home": True,
        "tool_on": False,
    }
    values.update(overrides)
    report = probe_ocr_final(
        FakeOcrSim(_positions(), robot_pose=(0.0, 0.0, 0.0), tool_on=False),
        _definition(),
        **values,
    )
    assert report["status"] == "FAIL"


def test_probe_experiment_routes_v1_08_final_without_changing_old_probe_contract():
    report = probe_experiment(
        FakeOcrSim(_positions(), robot_pose=(0.0, 0.0, 0.0), tool_on=False),
        _definition(),
        phase="final",
        scene_manifest=json.loads(
            _definition().scene_manifest.read_text(encoding="utf-8")
        ),
        run_context={
            "run_id": "run-v1-08",
            "scene_hash": _scene_hash(),
            "snapshot_id": "frame-000001",
            "entry_evidence": _refs(),
            "consumed_entry_ids": ("entry_a", "entry_b", "entry_c", "entry_d"),
            "robot_home": True,
            "tool_on": False,
        },
    )
    assert report["status"] == "PASS"
    assert report["same_run_evidence"] is True
