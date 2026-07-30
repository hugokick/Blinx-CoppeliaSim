from __future__ import annotations

import argparse
import json
from pathlib import Path

from coppeliasim_zmqremoteapi_client import RemoteAPIClient


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--probe-stl", type=Path)
    args = parser.parse_args()

    client = RemoteAPIClient()
    sim = client.require("sim")
    handles = sim.getObjectsInTree(sim.handle_scene, sim.handle_all, 0)
    rows = []
    for handle in handles:
        object_type = sim.getObjectType(handle)
        row = {
            "handle": handle,
            "path": sim.getObjectAlias(handle, 2),
            "alias": sim.getObjectAlias(handle, -1),
            "type": object_type,
            "parent": sim.getObjectParent(handle),
            "pose_world": sim.getObjectPose(handle, -1),
        }
        if object_type == sim.object_shape_type:
            size, pose = sim.getShapeBB(handle)
            row["bbox_size_m"] = size
            row["bbox_pose_local"] = pose
        rows.append(row)

    probes = []
    if args.probe_stl:
        for scale in (1.0, 0.001):
            handle = sim.importShape(
                0,
                args.probe_stl.resolve().as_posix(),
                0,
                0.0,
                scale,
            )
            size, pose = sim.getShapeBB(handle)
            probes.append(
                {
                    "scaling_factor": scale,
                    "bbox_size_m": size,
                    "bbox_pose_local": pose,
                    "pose_world": sim.getObjectPose(handle, -1),
                }
            )
            sim.removeObject(handle)

    payload = {
        "scene_object_count": len(rows),
        "rows": rows,
        "import_probes": probes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "PASS", "objects": len(rows), "probes": probes}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
