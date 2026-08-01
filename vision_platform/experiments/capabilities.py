from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from vision_platform.experiments.models import CapabilityReport


_ROBOT = frozenset({"robot.home", "robot.pose", "robot.move_world"})
_PROFILE_PAIR = frozenset({"camera.profile", "lighting.profile"})
_PROFILE_PAIR_REASON = (
    "视觉配置能力必须同时声明 camera.profile 和 lighting.profile"
)
_KNOWN = _ROBOT | frozenset(
    {
        "tool.suction",
        "camera.rgb",
        "camera.profile",
        "lighting.profile",
        "experiment.info",
        "scene.probe",
    }
)


def check_capabilities(
    application: Any,
    required: Iterable[str],
) -> CapabilityReport:
    requested = tuple(required)
    requested_set = set(requested)
    profile_pair_incomplete = bool(requested_set & _PROFILE_PAIR) and not (
        _PROFILE_PAIR <= requested_set
    )
    available: list[str] = []
    missing: list[str] = []
    reasons: dict[str, str] = {}
    backend = str(getattr(application.config, "robot_backend", ""))

    for capability in requested:
        if capability not in _KNOWN:
            missing.append(capability)
            reasons[capability] = f"未注册能力：{capability}"
        elif capability in _ROBOT and backend != "sim":
            missing.append(capability)
            reasons[capability] = "V2.2 禁止真实机器人后端"
        elif capability in _ROBOT and getattr(application, "robot", None) is None:
            missing.append(capability)
            reasons[capability] = "当前应用没有可用机械臂"
        elif capability == "tool.suction":
            tool = getattr(application, "tool", None)
            if tool is None:
                missing.append(capability)
                reasons[capability] = "当前应用没有可用吸盘"
            else:
                validate = getattr(tool, "validate", None)
                try:
                    if callable(validate):
                        validate()
                except Exception:
                    missing.append(capability)
                    reasons[capability] = "吸盘 TCP 或可抓取集合不可用"
                else:
                    available.append(capability)
        elif capability == "camera.rgb" and getattr(application, "camera", None) is None:
            missing.append(capability)
            reasons[capability] = "当前应用没有可用相机"
        elif capability in _PROFILE_PAIR and profile_pair_incomplete:
            missing.append(capability)
            reasons[capability] = _PROFILE_PAIR_REASON
        elif capability in _PROFILE_PAIR:
            camera_backend = str(
                getattr(application.config, "camera_backend", "")
            )
            if camera_backend != "sim":
                missing.append(capability)
                reasons[capability] = (
                    "视觉配置仅支持 CoppeliaSim 相机后端"
                )
            elif getattr(application, "sim", None) is None:
                missing.append(capability)
                reasons[capability] = "当前应用没有 CoppeliaSim 场景连接"
            elif getattr(application, "camera", None) is None:
                missing.append(capability)
                reasons[capability] = "当前应用没有可用相机"
            else:
                available.append(capability)
        elif capability == "scene.probe" and getattr(application, "sim", None) is None:
            missing.append(capability)
            reasons[capability] = "当前应用没有 CoppeliaSim 场景连接"
        else:
            available.append(capability)

    return CapabilityReport(
        available=tuple(available),
        missing=tuple(missing),
        reasons=reasons,
    )
