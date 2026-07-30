import json

import cv2
import numpy as np

from vision_platform.cli import main


def test_recognize_cli_writes_machine_readable_detections(tmp_path):
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    cv2.rectangle(image, (80, 60), (180, 160), (0, 0, 255), -1)
    source = tmp_path / "source.png"
    output = tmp_path / "result.json"
    annotated = tmp_path / "annotated.png"
    assert cv2.imwrite(str(source), image)

    exit_code = main(
        [
            "recognize",
            "--image",
            str(source),
            "--output",
            str(output),
            "--annotated",
            str(annotated),
        ]
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert payload["source"] == str(source.resolve())
    assert payload["detections"][0]["color"] == "red"
    assert payload["detections"][0]["shape"] == "square"
    assert annotated.is_file()


def test_recognize_cli_reads_and_writes_unicode_paths(tmp_path):
    unicode_dir = tmp_path / "中文目录"
    unicode_dir.mkdir()
    source = unicode_dir / "输入图像.png"
    output = unicode_dir / "识别结果.json"
    annotated = unicode_dir / "标注图像.png"
    image = np.zeros((160, 200, 3), dtype=np.uint8)
    cv2.circle(image, (100, 80), 35, (255, 0, 0), -1)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    source.write_bytes(encoded.tobytes())

    exit_code = main(
        [
            "recognize",
            "--image",
            str(source),
            "--output",
            str(output),
            "--annotated",
            str(annotated),
        ]
    )

    assert exit_code == 0
    assert json.loads(output.read_text(encoding="utf-8"))["detections"][0][
        "color"
    ] == "blue"
    assert annotated.is_file()
