from __future__ import annotations

from copy import deepcopy
import json

import pytest

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
        if self.sent[-1]["name"] == "vision2d.ocr_sort_entry":
            value = {
                "schema_version": 1,
                "entry_id": self.sent[-1]["args"]["entry_id"],
                "status": "COMPLETED",
                "plan_id": "b" * 64,
                "run_id": "run-1",
                "evidence_id": "ocr-entry-entry-c-1",
                "hardware_status": "PENDING_HARDWARE",
            }
        return ResponseMessage(
            command_id=self.sent[-1]["command_id"],
            status="PASS",
            value=value,
            error=None,
        ).to_dict()


def _value() -> dict[str, object]:
    return {
        "schema_version": 1,
        "snapshot_id": "frame-000001",
        "vision_bundle_path": "vision-bundle-V1-08-frame-000001.json",
        "manifest_sha256": "a" * 64,
        "scene_id": "V1-08",
        "status": "PASS",
        "training": {
            "train_count": 36,
            "test_count": 12,
            "held_out_accuracy": 1.0,
            "seed": 20260802,
        },
        "results": [
            {
                "identifier": identifier,
                "status": "PASS",
                "text": identifier,
                "characters": [
                    {
                        "character": identifier[0],
                        "bbox_px": [5, 5, 20, 60],
                        "confidence": 0.98,
                        "failure_code": None,
                        "confidence_method": "knn_neighbor_distance",
                    },
                    {
                        "character": identifier[1],
                        "bbox_px": [30, 5, 20, 60],
                        "confidence": 0.98,
                        "failure_code": None,
                        "confidence_method": "knn_neighbor_distance",
                    },
                ],
                "image_size": [96, 128],
                "threshold_method": "otsu",
                "character_count": 2,
                "failure_code": None,
                "processing_ms": 1.0,
                "schema_version": 1,
                "confidence_method": "knn_neighbor_distance",
            }
            for identifier in ("A1", "A2", "B1", "B2")
        ],
        "entries": [
            {
                "entry_id": entry_id,
                "part_id": part_id,
                "identifier": identifier,
                "route_id": route_id,
                "roi_px": [100 + index * 10, 100, 96, 128],
                "confidence": 0.98,
                "status": "APPROVED",
            }
            for index, (entry_id, part_id, identifier, route_id) in enumerate(
                (
                    ("entry_a", "part_a", "A1", "route_alpha"),
                    ("entry_b", "part_b", "A2", "route_alpha"),
                    ("entry_c", "part_c", "B1", "route_beta"),
                    ("entry_d", "part_d", "B2", "route_beta"),
                )
            )
        ],
        "plan_id": "b" * 64,
        "safe_z_mm": 110.0,
        "speed_mm_s": 15.0,
        "evidence": {
            "raw_path": "frames/frame-000001.png",
            "annotated_path": "frames/vision-V1-08-frame-000001-annotated.png",
            "bundle_path": "vision-bundle-V1-08-frame-000001.json",
        },
        "hardware_status": "PENDING_HARDWARE",
    }


def test_ocr_sorting_commands_are_strict_and_json_native() -> None:
    value = _value()
    connection = Connection(value)
    ctx = StudentContext(connection)

    result = ctx.vision2d.ocr_sorting()
    receipt = ctx.vision2d.sort_ocr_entry("entry_c")

    assert result.training.held_out_accuracy == 1.0
    assert tuple(item.entry_id for item in result.entries) == (
        "entry_a",
        "entry_b",
        "entry_c",
        "entry_d",
    )
    assert receipt.entry_id == "entry_c"
    assert receipt.status == "COMPLETED"
    assert [item["name"] for item in connection.sent] == [
        "vision2d.ocr_sorting",
        "vision2d.ocr_sort_entry",
    ]
    assert connection.sent[0]["args"] == {}
    assert connection.sent[1]["args"] == {"entry_id": "entry_c"}
    json.dumps(result.to_dict(), ensure_ascii=False, allow_nan=False)

    value["entries"][0]["confidence"] = 0.1
    assert result.entries[0].confidence == 0.98


def test_ocr_commands_are_published_without_raw_device_capabilities() -> None:
    assert "vision2d.ocr_sorting" in ALLOWED_COMMANDS
    assert "vision2d.ocr_sort_entry" in ALLOWED_COMMANDS
    CommandMessage("000001", "vision2d.ocr_sorting", {})
    CommandMessage("000002", "vision2d.ocr_sort_entry", {"entry_id": "entry_a"})


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.update({"extra": 1}),
        lambda value: value["training"].update({"extra": 1}),
        lambda value: value["entries"].__setitem__(0, {"entry_id": "entry_a"}),
        lambda value: value.update({"status": "REJECTED"}),
        lambda value: value["entries"][0].update({"confidence": float("nan")}),
        lambda value: value["results"][0]["characters"][0].update(
            {"bbox_px": [90, 5, 20, 60]}
        ),
        lambda value: value["evidence"].update(
            {"raw_path": "frames/ocr:raw.png"}
        ),
        lambda value: value.update({"hardware_status": "PASS"}),
    ],
)
def test_ocr_response_is_strictly_validated(mutation) -> None:
    value = deepcopy(_value())
    mutation(value)
    with pytest.raises(RuntimeError, match="PROTOCOL_RESPONSE_INVALID"):
        StudentContext(Connection(value)).vision2d.ocr_sorting()


@pytest.mark.parametrize(
    "confidence, accepted",
    [(0.41, False), (0.899999, False), (0.90, True)],
)
def test_ocr_sorting_character_confidence_uses_release_threshold(
    confidence: float,
    accepted: bool,
) -> None:
    value = deepcopy(_value())
    value["results"][0]["characters"][0]["confidence"] = confidence

    if accepted:
        result = StudentContext(Connection(value)).vision2d.ocr_sorting()
        assert result.results[0].characters[0].confidence == confidence
    else:
        with pytest.raises(RuntimeError, match="PROTOCOL_RESPONSE_INVALID"):
            StudentContext(Connection(value)).vision2d.ocr_sorting()


@pytest.mark.parametrize(
    "confidence, accepted",
    [(0.41, False), (0.899999, False), (0.90, True)],
)
def test_ocr_sorting_entry_confidence_uses_release_threshold(
    confidence: float,
    accepted: bool,
) -> None:
    value = deepcopy(_value())
    value["entries"][0]["confidence"] = confidence

    if accepted:
        result = StudentContext(Connection(value)).vision2d.ocr_sorting()
        assert result.entries[0].confidence == confidence
    else:
        with pytest.raises(RuntimeError, match="PROTOCOL_RESPONSE_INVALID"):
            StudentContext(Connection(value)).vision2d.ocr_sorting()


@pytest.mark.parametrize(
    "args",
    [
        {"entry_id": "entry_a", "extra": 1},
        {"entry_id": True},
        {"entry_id": 7},
        {},
    ],
)
def test_ocr_entry_sdk_rejects_non_strict_entry_id_before_send(args) -> None:
    connection = Connection({
        "schema_version": 1,
        "entry_id": "entry_a",
        "status": "COMPLETED",
        "plan_id": "b" * 64,
        "run_id": "run-1",
        "evidence_id": "ocr-entry-entry-a-1",
        "hardware_status": "PENDING_HARDWARE",
    })
    with pytest.raises((TypeError, ValueError)):
        StudentContext(connection).vision2d.sort_ocr_entry(**args)
    assert connection.sent == []
