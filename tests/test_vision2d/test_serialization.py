import json

import numpy as np

from tests.test_vision2d.synthetic_factory import SyntheticObject, make_scene
from vision_platform.vision2d import analyze_image
from vision_platform.vision2d.serialization import result_to_dict


def _contains_numpy(value):
    if isinstance(value, (np.ndarray, np.generic)):
        return True
    if isinstance(value, dict):
        return any(_contains_numpy(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_numpy(item) for item in value)
    return False


def test_result_serialization_is_json_safe_and_excludes_intermediate_images():
    image, _ = make_scene(
        (240, 180),
        (SyntheticObject("rectangle", "red", (120, 90), (70, 35), 15.0),),
    )
    analysis = analyze_image(image)

    payload = result_to_dict(analysis.result)
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True)

    assert payload["schema_version"] == 1
    assert payload["targets"][0]["detection_id"] == "det-001"
    assert "intermediate_images" not in payload
    assert not _contains_numpy(payload)
    assert encoded == json.dumps(
        result_to_dict(analyze_image(image).result),
        ensure_ascii=False,
        sort_keys=True,
    )
