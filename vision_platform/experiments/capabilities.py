from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from vision_platform.experiments.models import CapabilityReport


_ROBOT = frozenset({"robot.home", "robot.pose", "robot.move_world"})
_PROFILE_PAIR = frozenset({"camera.profile", "lighting.profile"})
_VISION2D_DEPENDENCIES = frozenset(
    {"camera.rgb", "camera.profile", "lighting.profile"}
)
_TEMPLATE_MATCH_DEPENDENCIES = frozenset(
    {"camera.rgb", "camera.profile", "lighting.profile"}
)
_CODE_ROUTING_DEPENDENCIES = frozenset(
    {
        "camera.rgb", "camera.profile", "lighting.profile",
        "robot.home", "robot.pose", "robot.move_world",
        "tool.suction", "scene.probe",
    }
)
_OCR_SORTING_DEPENDENCIES = frozenset(
    {"camera.rgb", "camera.profile", "lighting.profile"}
)
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
        "vision2d.analysis",
        "vision2d.template_matching",
        "vision2d.code_routing",
        "vision2d.ocr_sorting",
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
        elif capability == "vision2d.analysis":
            if not _VISION2D_DEPENDENCIES <= requested_set:
                missing.append(capability)
                reasons[capability] = (
                    "二维视觉分析必须同时声明相机与成对视觉配置能力"
                )
            elif str(
                getattr(application.config, "camera_backend", "")
            ) != "sim":
                missing.append(capability)
                reasons[capability] = (
                    "二维视觉分析仅支持 CoppeliaSim 相机后端"
                )
            elif getattr(application, "sim", None) is None:
                missing.append(capability)
                reasons[capability] = "当前应用没有 CoppeliaSim 场景连接"
            elif getattr(application, "camera", None) is None:
                missing.append(capability)
                reasons[capability] = "当前应用没有可用相机"
            else:
                available.append(capability)
        elif capability == "vision2d.template_matching":
            if not _TEMPLATE_MATCH_DEPENDENCIES <= requested_set:
                missing.append(capability)
                reasons[capability] = (
                    "模板匹配必须同时声明相机与成对视觉配置能力"
                )
            elif str(getattr(application.config, "camera_backend", "")) != "sim":
                missing.append(capability)
                reasons[capability] = "模板匹配仅支持 CoppeliaSim 相机后端"
            elif getattr(application, "sim", None) is None:
                missing.append(capability)
                reasons[capability] = "当前应用没有 CoppeliaSim 场景连接"
            elif getattr(application, "camera", None) is None:
                missing.append(capability)
                reasons[capability] = "当前应用没有可用相机"
            else:
                available.append(capability)
        elif capability == "vision2d.code_routing":
            missing_dependencies = _CODE_ROUTING_DEPENDENCIES - requested_set
            if missing_dependencies:
                missing.append(capability)
                reasons[capability] = (
                    "代码路由必须同时声明：" + ", ".join(sorted(missing_dependencies))
                )
            elif str(getattr(application.config, "camera_backend", "")) != "sim":
                missing.append(capability)
                reasons[capability] = "代码路由仅支持 CoppeliaSim 相机"
            elif str(getattr(application.config, "robot_backend", "")) != "sim":
                missing.append(capability)
                reasons[capability] = "代码路由仅支持 CoppeliaSim 机器人"
            elif any(
                getattr(application, name, None) is None
                for name in ("sim", "camera", "robot", "tool")
            ):
                missing.append(capability)
                reasons[capability] = "代码路由缺少场景、相机、机器人或吸盘"
            else:
                available.append(capability)
        elif capability == "vision2d.ocr_sorting":
            missing_dependencies = _OCR_SORTING_DEPENDENCIES - requested_set
            if missing_dependencies:
                missing.append(capability)
                reasons[capability] = (
                    "OCR 分拣必须同时声明："
                    + ", ".join(sorted(missing_dependencies))
                )
            elif str(getattr(application.config, "camera_backend", "")) != "sim":
                missing.append(capability)
                reasons[capability] = "OCR 分拣仅支持 CoppeliaSim 相机"
            elif str(getattr(application.config, "robot_backend", "")) != "sim":
                missing.append(capability)
                reasons[capability] = "OCR 分拣仅支持 CoppeliaSim 机器人"
            elif any(
                getattr(application, name, None) is None
                for name in ("sim", "camera", "robot", "tool")
            ):
                missing.append(capability)
                reasons[capability] = "OCR 分拣缺少场景、相机、机器人或吸盘"
            else:
                available.append(capability)
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
