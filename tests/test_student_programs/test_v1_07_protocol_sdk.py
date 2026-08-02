from __future__ import annotations

from copy import deepcopy

import pytest

from vision_platform.student.protocol import ResponseMessage
from vision_platform.student.sdk import StudentContext


class Connection:
    def __init__(self, value: object) -> None:
        self.value = value
        self.sent: list[dict] = []

    def send(self, payload: dict) -> None:
        self.sent.append(payload)

    def recv(self) -> dict:
        return ResponseMessage(
            command_id=self.sent[-1]["command_id"], status="PASS", value=self.value, error=None
        ).to_dict()


def _entry(index: int) -> dict[str, object]:
    return {
        "entry_id": f"entry_{index}", "part_id": f"part_{index}",
        "code_type": "qr" if index < 2 else "ean13",
        "payload": f"payload-{index}", "route_id": "route_red" if index % 2 == 0 else "route_blue",
        "pick_xyz_mm": [40.0 + 10.0 * index, -45.0 + index, 18.0],
        "drop_xyz_mm": [116.0 + index, -60.0 if index % 2 == 0 else 60.0, 22.0],
        "confidence": 0.98,
    }


def _value() -> dict[str, object]:
    return {
        "schema_version": 1,
        "snapshot_id": "frame-000001",
        "vision_bundle_path": "vision-bundle-V1-07-frame-000001.json",
        "plan_id": "a" * 64,
        "status": "PASS",
        "safe_z_mm": 110.0,
        "speed_mm_s": 15.0,
        "entries": [_entry(index) for index in range(4)],
    }


def test_code_routes_sends_no_arguments_and_returns_frozen_result() -> None:
    raw = _value()
    connection = Connection(raw)
    result = StudentContext(connection).vision2d.code_routes()
    raw["entries"][0]["pick_xyz_mm"][0] = 999.0
    assert connection.sent[0]["name"] == "vision2d.code_routes"
    assert connection.sent[0]["args"] == {}
    assert result.plan_id == "a" * 64
    assert result.entries[0].pick_xyz_mm == (40.0, -45.0, 18.0)
    with pytest.raises((AttributeError, TypeError)):
        result.entries[0].pick_xyz_mm[0] = 1.0


@pytest.mark.parametrize(
    ("mutation", "field"),
    [
        (lambda value: value.update({"extra": 1}), "fields"),
        (lambda value: value.update({"schema_version": 2}), "schema_version"),
        (lambda value: value.update({"plan_id": "../plan"}), "plan_id"),
        (lambda value: value.update({"status": "PARTIAL"}), "status"),
        (lambda value: value.update({"safe_z_mm": float("nan")}), "safe_z_mm"),
        (lambda value: value["entries"].append(_entry(5)), "entries"),
        (lambda value: value["entries"].__setitem__(1, deepcopy(value["entries"][0])), "entries"),
        (lambda value: value["entries"][0].update({"payload": True}), "payload"),
    ],
)
def test_code_route_response_is_strict(mutation, field: str) -> None:
    value = deepcopy(_value())
    mutation(value)
    with pytest.raises(RuntimeError, match=field):
        StudentContext(Connection(value)).vision2d.code_routes()


def test_code_route_response_rejects_invalid_numbers_and_identifiers() -> None:
    for field, invalid in (
        ("speed_mm_s", float("inf")),
        ("confidence", -0.1),
        ("entry_id", "../entry"),
        ("code_type", "python"),
    ):
        value = deepcopy(_value())
        if field == "confidence":
            value["entries"][0][field] = invalid
        elif field in {"entry_id", "code_type"}:
            value["entries"][0][field] = invalid
        else:
            value[field] = invalid
        with pytest.raises(RuntimeError, match=field):
            StudentContext(Connection(value)).vision2d.code_routes()
