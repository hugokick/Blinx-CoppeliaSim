from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from simulation.training_scenes.build_scene import build_scene


def main() -> int:
    build_scene(
        spec_path=PROJECT_ROOT / "simulation" / "vision_code_routing_lab" / "scene_spec.json",
        host="127.0.0.1",
        port=23007,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
