from __future__ import annotations

import json

from vision_platform.cli import build_parser, main


def test_v1_07_uses_the_generic_formal_experiment_cli(capsys) -> None:
    arguments = build_parser().parse_args(
        ["experiment-run", "--experiment", "V1-07", "--port", "23007"]
    )
    assert arguments.experiment == "V1-07"
    assert arguments.port == "23007"
    assert main(["experiment-show", "--experiment", "V1-07"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["student_template"].endswith("v1_07_code_routing.py")
    assert payload["hardware_status"] == "PENDING_HARDWARE"
