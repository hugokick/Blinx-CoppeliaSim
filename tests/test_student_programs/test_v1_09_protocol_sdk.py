from __future__ import annotations

from copy import deepcopy
import json

import pytest

from vision_platform.experiments.defect_sorting import (
    DefectSortReceipt,
    defect_sort_receipt_to_dict,
)
from vision_platform.student.protocol import ALLOWED_COMMANDS, CommandMessage, ResponseMessage
from vision_platform.student.sdk import StudentContext


class Connection:
    def __init__(self, value: object) -> None:
        self.value = value
        self.sent: list[dict] = []

    def send(self, payload: dict) -> None:
        self.sent.append(payload)

    def recv(self) -> dict:
        value = self.value
        if self.sent[-1]["name"] == "vision2d.defect_sort_entry":
            value = {
                "schema_version": 1,
                "run_id": "run-v1-09",
                "plan_id": "a" * 64,
                "entry_id": self.sent[-1]["args"]["entry_id"],
                "part_id": "part_a",
                "decision": "qualified",
                "defect_type": None,
                "slot_id": "slot_qualified",
                "status": "COMPLETED",
                "evidence_id": "evidence-entry-a",
                "evidence_sha256": "b" * 64,
                "hardware_status": "PENDING_HARDWARE",
            }
        return ResponseMessage(
            command_id=self.sent[-1]["command_id"],
            status="PASS",
            value=value,
            error=None,
        ).to_dict()


class ReceiptConnection:
    def __init__(self, value: dict[str, object]) -> None:
        self.value = value
        self.sent: list[dict] = []

    def send(self, payload: dict) -> None:
        self.sent.append(payload)

    def recv(self) -> dict:
        return ResponseMessage(
            command_id=self.sent[-1]["command_id"],
            status="PASS",
            value=self.value,
            error=None,
        ).to_dict()


def _value() -> dict[str, object]:
    digest = "a" * 64
    return {
        "schema_version": 1,
        "run_id": "run-v1-09",
        "frame_id": "frame-000001",
        "scene_sha256": digest,
        "config_sha256": "b" * 64,
        "asset_manifest_sha256": "c" * 64,
        "plan_id": "d" * 64,
        "status": "PASS",
        "image_size": [1024, 1024],
        "entries": [
            {
                "entry_id": entry_id,
                "part_id": part_id,
                "decision": decision,
                "defect_type": None if decision == "qualified" else decision,
                "findings": [] if decision == "qualified" else [{
                    "defect_type": decision,
                    "bbox_px": [10, 10, 20, 20],
                    "area_px2": 400.0,
                    "relative_area": 0.02,
                    "metric": 0.02,
                    "threshold": 0.01,
                    "confidence": 0.95,
                }],
                "findings_sha256": "e" * 64,
                "reference_crop_sha256": "f" * 64,
                "candidate_crop_sha256": "1" * 64,
                "route_id": route_id,
                "status": "APPROVED",
            }
            for entry_id, part_id, decision, route_id in (
                ("entry_a", "part_a", "qualified", "route_qualified"),
                ("entry_b", "part_b", "missing", "route_missing"),
                ("entry_c", "part_c", "hole", "route_hole"),
                ("entry_d", "part_d", "foreign", "route_foreign"),
                ("entry_e", "part_e", "broken", "route_broken"),
                ("entry_f", "part_f", "dimension", "route_dimension"),
            )
        ],
        "evidence": {
            "raw_sha256": "2" * 64,
            "reference_sha256": "3" * 64,
            "candidate_sha256": "4" * 64,
            "annotated_sha256": "5" * 64,
            "mask_sha256": "6" * 64,
        },
        "hardware_status": "PENDING_HARDWARE",
    }


def test_surface_defect_commands_are_public_without_raw_device_api() -> None:
    assert "vision2d.surface_defects" in ALLOWED_COMMANDS
    assert "vision2d.defect_sort_entry" in ALLOWED_COMMANDS
    CommandMessage("000001", "vision2d.surface_defects", {})
    CommandMessage("000002", "vision2d.defect_sort_entry", {"entry_id": "entry_a"})
    assert "robot.move_world" in ALLOWED_COMMANDS  # blocked by the V1-09 runner, not exposed by the DTO


@pytest.mark.parametrize(
    ("entry_id", "part_id", "decision", "slot_id", "expected_defect"),
    [
        ("entry_a", "part_a", "qualified", "slot_qualified", None),
        ("entry_b", "part_b", "missing", "slot_missing", "missing"),
    ],
)
def test_real_receipt_serializer_is_accepted_by_student_sdk(
    entry_id: str,
    part_id: str,
    decision: str,
    slot_id: str,
    expected_defect: str | None,
) -> None:
    payload = defect_sort_receipt_to_dict(
        DefectSortReceipt(
            run_id="run-v1-09",
            plan_id="a" * 64,
            entry_id=entry_id,
            part_id=part_id,
            decision=decision,
            slot_id=slot_id,
            status="COMPLETED",
            evidence_id=f"post-{entry_id}",
            evidence_sha256="b" * 64,
            hardware_status="PENDING_HARDWARE",
        )
    )
    connection = ReceiptConnection(payload)

    receipt = StudentContext(connection).vision2d.defect_sort_entry(entry_id)

    assert receipt.entry_id == entry_id
    assert receipt.decision == decision
    assert receipt.defect_type == expected_defect
    assert receipt.status == "COMPLETED"
    assert connection.sent[0]["args"] == {"entry_id": entry_id}


def test_surface_defect_sdk_returns_decisions_and_digests_only() -> None:
    connection = Connection(_value())
    context = StudentContext(connection)
    analysis = context.vision2d.surface_defects()
    receipt = context.vision2d.defect_sort_entry("entry_a")

    assert analysis.plan_id == "d" * 64
    assert tuple(item.decision for item in analysis.entries) == (
        "qualified", "missing", "hole", "foreign", "broken", "dimension"
    )
    assert receipt.decision == "qualified"
    assert receipt.slot_id == "slot_qualified"
    assert [item["name"] for item in connection.sent] == [
        "vision2d.surface_defects",
        "vision2d.defect_sort_entry",
    ]
    assert connection.sent[0]["args"] == {}
    assert connection.sent[1]["args"] == {"entry_id": "entry_a"}
    public = analysis.to_dict()
    assert all("roi_px" not in entry and "pick_xyz_mm" not in entry for entry in public["entries"])
    json.dumps(public, allow_nan=False)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.update({"extra": 1}),
        lambda value: value["entries"][0].update({"roi_px": [0, 0, 1, 1]}),
        lambda value: value["evidence"].update({"raw_path": "secret.png"}),
        lambda value: value["entries"][0]["findings"].append({"bbox_px": [0, 0, 1, 1]}),
        lambda value: value.update({"scene_sha256": "not-a-sha"}),
        lambda value: value["entries"][0].update({"candidate_crop_sha256": float("nan")}),
    ],
)
def test_surface_defect_response_is_strictly_validated(mutation) -> None:
    value = deepcopy(_value())
    mutation(value)
    with pytest.raises(RuntimeError, match="PROTOCOL_RESPONSE_INVALID: vision2d.surface_defects"):
        StudentContext(Connection(value)).vision2d.surface_defects()


@pytest.mark.parametrize("entry_id", ["", "入口_a", "x" * 81, True, 7, {"entry_id": "entry_a"}])
def test_defect_entry_rejects_non_strict_id_before_send(entry_id) -> None:
    connection = Connection(_value())
    with pytest.raises((TypeError, ValueError)):
        StudentContext(connection).vision2d.defect_sort_entry(entry_id)
    assert connection.sent == []


def test_defect_command_arguments_are_exact() -> None:
    # The wire envelope only whitelists command names.  Exact argument
    # validation belongs to the gateway, where it can fail before capture.
    assert CommandMessage("000001", "vision2d.surface_defects", {"roi": [0, 0, 1, 1]}).name == "vision2d.surface_defects"
    assert CommandMessage("000002", "vision2d.defect_sort_entry", {"entry_id": "entry_a", "speed": 1}).name == "vision2d.defect_sort_entry"
