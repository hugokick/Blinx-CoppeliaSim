from collections.abc import Mapping

from student_programs.templates.r1_common import (
    _finite_float,
    _finite_vector,
    MIN_VISUAL_CONFIDENCE,
    detect_colored_objects,
    pick_and_place,
)


def _class_id(item):
    return f"{item.color}_{item.shape}"


def _build_plan(parameters, objects):
    classes = parameters["classes"]
    routes = parameters["drop_poses_mm"]
    if (
        not isinstance(classes, (list, tuple))
        or not classes
        or any(type(class_id) is not str or not class_id for class_id in classes)
        or len(set(classes)) != len(classes)
    ):
        raise RuntimeError("classes 必须包含不重复的非空类别")
    if not isinstance(routes, Mapping) or set(routes) != set(classes):
        raise RuntimeError("drop_poses_mm 必须与 classes 一一对应")
    if not isinstance(objects, (list, tuple)) or len(objects) != len(classes):
        raise RuntimeError("构件识别数量与 classes 不一致")

    try:
        pick_z = _finite_float(parameters["pick_z_mm"], "pick_z_mm")
        safe_z = _finite_float(parameters["safe_z_mm"], "safe_z_mm")
        plan = []
        detected_ids = []
        for index, item in enumerate(objects):
            class_id = _class_id(item)
            if class_id in detected_ids:
                raise ValueError(f"构件类别 {class_id} 重复识别")
            confidence = _finite_float(
                item.confidence,
                f"objects[{index}].confidence",
            )
            if not 0.0 <= confidence <= 1.0:
                raise ValueError("分类置信度必须在 [0, 1] 范围内")
            if confidence < MIN_VISUAL_CONFIDENCE:
                raise ValueError(
                    "分类置信度低于 "
                    f"{MIN_VISUAL_CONFIDENCE:.1f}，停止运动"
                )
            pick_xy = _finite_vector(
                item.world_xy_mm,
                2,
                f"objects[{index}].world_xy_mm",
            )
            slot = _finite_vector(routes[class_id], 3, f"drop_poses_mm.{class_id}")
            if safe_z <= max(pick_z, slot[2]):
                raise ValueError("safe_z_mm 必须高于抓取和全部放置高度")
            detected_ids.append(class_id)
            plan.append((class_id, pick_xy, slot))
        if set(detected_ids) != set(classes):
            raise ValueError("构件识别类别与 classes 不一致")
    except (AttributeError, KeyError, TypeError, ValueError) as error:
        raise RuntimeError(f"构件识别结果或运动参数无效：{error}") from error
    return plan, pick_z, safe_z


def main(ctx):
    parameters = ctx.experiment.info()["public_parameters"]
    frame = ctx.camera.capture()
    objects = detect_colored_objects(
        frame.image_bgr,
        calibration_matrix=parameters["calibration_matrix"],
        pick_x_max_mm=parameters["pick_region_world_mm"]["x_max"],
    )
    plan, pick_z, safe_z = _build_plan(parameters, objects)
    ctx.robot.home()
    for class_id, pick_xy, slot in plan:
        pick_and_place(
            ctx,
            pick_xy=pick_xy,
            drop_xyz=slot,
            pick_z_mm=pick_z,
            safe_z_mm=safe_z,
            speed=12,
        )
        ctx.checkpoint(f"完成 {class_id} 分仓")
    ctx.robot.home()
