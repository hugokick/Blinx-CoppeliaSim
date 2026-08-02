from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2

from tools.vision_lab.generate_v1_07_code_assets import generate
from vision_platform.vision2d.code_recognition import recognize_codes


EXPECTED = {
    ("part_a", "qr", "V1-07-A"),
    ("part_b", "qr", "V1-07-B"),
    ("part_c", "ean13", "6901234567892"),
    ("part_d", "ean13", "6901234567809"),
}


def test_generated_assets_are_bound_and_decoded_by_production(tmp_path: Path) -> None:
    manifest_path = generate(tmp_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert set(payload) == {"schema_version", "entries"}
    assert payload["schema_version"] == 1
    assert {
        (entry["part_id"], entry["code_type"], entry["payload"])
        for entry in payload["entries"]
    } == EXPECTED
    for entry in payload["entries"]:
        assert set(entry) == {
            "asset_id", "part_id", "code_type", "payload", "path", "sha256",
            "size_px", "channels", "generator",
        }
        path = tmp_path / Path(entry["path"]).name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        assert image is not None
        assert [image.shape[1], image.shape[0]] == entry["size_px"]
        assert entry["channels"] == 3
        result = recognize_codes(image, max_codes=1)
        assert result.status == "PASS"
        assert len(result.readings) == 1
        assert result.readings[0].code_type == entry["code_type"]
        assert result.readings[0].data == entry["payload"]


def test_check_mode_detects_no_drift(tmp_path: Path) -> None:
    generate(tmp_path)
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    generate(tmp_path, check=True)
    after = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    assert after == before
