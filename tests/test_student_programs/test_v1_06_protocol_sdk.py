from __future__ import annotations

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
        return ResponseMessage(
            command_id=self.sent[-1]["command_id"],
            status="PASS",
            value=self.value,
            error=None,
        ).to_dict()


def _value(**overrides: object) -> dict:
    value: dict[str, object] = {
        "snapshot_id": "frame-000001",
        "vision_bundle_path": "vision-bundle-V1-06-frame-000001.json",
        "template_id": "v1_06_red_rectangle",
        "template_version": "1.0.0",
        "matched": True,
        "status": "MATCHED",
        "score": 0.93,
        "threshold": 0.72,
        "bbox_px": [221, 205, 67, 40],
        "center_px": [254.5, 225.0],
        "image_size": [512, 512],
        "search_roi_px": [0, 0, 512, 512],
        "method": "TM_CCOEFF_NORMED",
    }
    value.update(overrides)
    return value


def test_template_match_command_and_student_alias_return_frozen_result() -> None:
    connection = Connection(_value())
    ctx = StudentContext(connection)

    result = ctx.vision2d.template_match()
    alias = ctx.vision2d.match_template()

    assert result.template_id == "v1_06_red_rectangle"
    assert result.matched is True
    assert result.bbox_px == (221, 205, 67, 40)
    assert result.center_px == (254.5, 225.0)
    assert result.image_size == (512, 512)
    assert result.search_roi_px == (0, 0, 512, 512)
    assert alias == result
    assert [item["name"] for item in connection.sent] == [
        "vision2d.template_match",
        "vision2d.template_match",
    ]
    assert connection.sent[0]["args"] == {}


def test_template_match_protocol_command_is_whitelisted_without_arguments() -> None:
    assert "vision2d.template_match" in ALLOWED_COMMANDS
    command = CommandMessage("000001", "vision2d.template_match", {})
    assert command.to_dict()["args"] == {}


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"extra": 1}, "fields"),
        ({"matched": "yes"}, "matched"),
        ({"score": float("nan")}, "score"),
        ({"bbox_px": [1, 2, 3]}, "bbox_px"),
        ({"search_roi_px": [0, 0, 999, 999]}, "search_roi_px"),
        ({"method": "opencv.execute"}, "method"),
    ],
)
def test_template_match_response_is_strictly_validated(overrides, message: str) -> None:
    with pytest.raises(RuntimeError, match=message):
        StudentContext(Connection(_value(**overrides))).vision2d.template_match()
