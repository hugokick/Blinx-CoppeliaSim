from student_programs.templates.r1_common import (
    _finite_float,
    _finite_vector,
    MIN_VISUAL_CONFIDENCE,
    detect_digit_objects,
    pick_and_place,
)


def _build_plan(parameters, detected):
    order = parameters["order"]
    slots = parameters["drop_slots_mm"]
    if (
        not isinstance(order, (list, tuple))
        or not order
        or any(type(digit) is not int for digit in order)
        or len(set(order)) != len(order)
    ):
        raise RuntimeError("order 必须包含不重复的整数数字")
    if not isinstance(slots, (list, tuple)) or len(slots) != len(order):
        raise RuntimeError("drop_slots_mm 必须与 order 等长")
    if not isinstance(detected, (list, tuple)) or len(detected) != len(order):
        raise RuntimeError("数字识别数量与 order 不一致")

    by_digit = {}
    try:
        for index, entry in enumerate(detected):
            if not isinstance(entry, (list, tuple)) or len(entry) != 3:
                raise ValueError("识别结果必须包含数字、坐标和置信度")
            digit, xy, confidence_value = entry
            if type(digit) is not int:
                raise ValueError("数字类别必须是整数")
            if digit in by_digit:
                raise ValueError(f"数字 {digit} 重复识别")
            confidence = _finite_float(
                confidence_value,
                f"detected[{index}].confidence",
            )
            if not 0.0 <= confidence <= 1.0:
                raise ValueError("匹配置信度必须在 [0, 1] 范围内")
            if confidence < MIN_VISUAL_CONFIDENCE:
                raise ValueError(
                    "匹配置信度低于 "
                    f"{MIN_VISUAL_CONFIDENCE:.1f}，停止运动"
                )
            by_digit[digit] = (
                _finite_vector(xy, 2, f"detected[{index}].world_xy_mm"),
                confidence,
            )
        if set(by_digit) != set(order):
            raise ValueError("数字识别类别与 order 不一致")
        pick_z = _finite_float(parameters["pick_z_mm"], "pick_z_mm")
        safe_z = _finite_float(parameters["safe_z_mm"], "safe_z_mm")
        plan = []
        for index, digit in enumerate(order):
            slot = _finite_vector(slots[index], 3, f"drop_slots_mm[{index}]")
            if safe_z <= max(pick_z, slot[2]):
                raise ValueError("safe_z_mm 必须高于抓取和全部放置高度")
            xy, confidence = by_digit[digit]
            plan.append((digit, xy, confidence, slot))
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError(f"数字识别结果或运动参数无效：{error}") from error
    return plan, pick_z, safe_z


def main(ctx):
    parameters = ctx.experiment.info()["public_parameters"]
    frame = ctx.camera.capture()
    detected = detect_digit_objects(
        frame.image_bgr,
        calibration_matrix=parameters["calibration_matrix"],
        pick_x_max_mm=parameters["pick_region_world_mm"]["x_max"],
        reference_dir=parameters["digit_reference_dir"],
    )
    plan, pick_z, safe_z = _build_plan(parameters, detected)
    ctx.robot.home()
    for digit, xy, confidence, slot in plan:
        ctx.log(f"数字 {digit}，匹配置信度 {confidence:.3f}")
        pick_and_place(
            ctx,
            pick_xy=xy,
            drop_xyz=slot,
            pick_z_mm=pick_z,
            safe_z_mm=safe_z,
            speed=12,
        )
    ctx.robot.home()
