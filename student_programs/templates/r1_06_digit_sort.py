from student_programs.templates.r1_common import (
    detect_digit_objects,
    pick_and_place,
)


def main(ctx):
    parameters = ctx.experiment.info()["public_parameters"]
    frame = ctx.camera.capture()
    detected = detect_digit_objects(
        frame.image_bgr,
        calibration_matrix=parameters["calibration_matrix"],
        pick_x_max_mm=parameters["pick_region_world_mm"]["x_max"],
        reference_dir=parameters["digit_reference_dir"],
    )
    by_digit = {digit: (xy, confidence) for digit, xy, confidence in detected}
    order = parameters["order"]
    if set(by_digit) != set(order):
        raise RuntimeError(f"数字识别不完整：{sorted(by_digit)}")
    ctx.robot.home()
    for index, digit in enumerate(order):
        xy, confidence = by_digit[digit]
        ctx.log(f"数字 {digit}，匹配置信度 {confidence:.3f}")
        pick_and_place(
            ctx,
            pick_xy=xy,
            drop_xyz=parameters["drop_slots_mm"][index],
            pick_z_mm=parameters["pick_z_mm"],
            safe_z_mm=parameters["safe_z_mm"],
            speed=12,
        )
    ctx.robot.home()
