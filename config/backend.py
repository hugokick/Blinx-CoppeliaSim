import os


def get_backend_settings():
    backend = os.getenv("ROBOT_BACKEND", "sim").strip().lower()
    return {
        "backend": backend,
        "sim": {
            "host": os.getenv("COPPELIA_HOST", "127.0.0.1"),
            "port": int(os.getenv("COPPELIA_PORT", "23000")),
            "scene": os.getenv(
                "COPPELIA_SCENE",
                os.path.join(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "robot_backends", "models", "BLX_openr6.ttt",
                ),
            ),
            "joint_paths": [
                "/BLX_joint1",
                "/BLX_joint2",
                "/BLX_joint3",
                "/BLX_joint4",
                "/BLX_joint5",
                "/BLX_joint6",
            ],
            "tool_path": "/BLX_tool_suction",
            "base_path": "/BLX_base_link",
            "tip_path": "/BLX_tool_suction",
            "target_path": "/BLX_target",
            "home_angles": [0, 0, 0, 0, 0, 0],
            "joint5_offset_deg": 0,  # BLX has no mechanical offset
            "ik_search_time": 0.25,
            "ik_search_precision": 0.001,
            "ik_metric": [1, 1, 1, 0.1],
            # Workspace constraints (verified 2026-06-03)
            #   <=150mm from base: IK precision < 0.2mm
            #   >150mm from base: may hit joint limits
            "workspace_radius_warn_mm": 150,
        },
    }
