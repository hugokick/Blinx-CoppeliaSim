from student_programs.templates.r1_common import (
    _finite_float,
    _finite_vector,
    MIN_VISUAL_CONFIDENCE,
    detect_colored_objects,
    pick_and_place,
)


def _build_plan(parameters, objects):
    slots = parameters["stack_slots_mm"]
    if not isinstance(objects, (list, tuple)) or len(objects) != 6:
        raise RuntimeError(f"期望识别 6 个码垛物体，实际 {len(objects)} 个")
    if not isinstance(slots, (list, tuple)) or len(slots) != 6:
        raise RuntimeError("stack_slots_mm 必须恰好包含 6 个目标位")
    try:
        pick_z = _finite_float(parameters["pick_z_mm"], "pick_z_mm")
        safe_z = _finite_float(parameters["safe_z_mm"], "safe_z_mm")
        plan = []
        for index in range(6):
            item = objects[index]
            confidence = _finite_float(
                item.confidence,
                f"objects[{index}].confidence",
            )
            if not 0.0 <= confidence <= 1.0:
                raise ValueError("confidence 必须在 [0, 1] 范围内")
            if confidence < MIN_VISUAL_CONFIDENCE:
                raise ValueError(
                    "识别置信度低于 "
                    f"{MIN_VISUAL_CONFIDENCE:.1f}，停止运动"
                )
            pick_xy = _finite_vector(
                item.world_xy_mm,
                2,
                f"objects[{index}].world_xy_mm",
            )
            slot = _finite_vector(slots[index], 3, f"stack_slots_mm[{index}]")
            if safe_z <= max(pick_z, slot[2]):
                raise ValueError("safe_z_mm 必须高于抓取和全部放置高度")
            plan.append((pick_xy, slot))
    except (AttributeError, KeyError, TypeError, ValueError) as error:
        raise RuntimeError(f"码垛识别结果或运动参数无效：{error}") from error
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
    for index, (pick_xy, slot) in enumerate(plan, start=1):
        pick_and_place(
            ctx,
            pick_xy=pick_xy,
            drop_xyz=slot,
            pick_z_mm=pick_z,
            safe_z_mm=safe_z,
            speed=12,
        )
        ctx.checkpoint(f"完成第 {index} 块码垛")
    ctx.robot.home()
