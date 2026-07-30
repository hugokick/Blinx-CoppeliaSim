from student_programs.templates.r1_common import (
    detect_colored_objects,
    pick_and_place,
)


def _class_id(item):
    return f"{item.color}_{item.shape}"


def main(ctx):
    parameters = ctx.experiment.info()["public_parameters"]
    frame = ctx.camera.capture()
    objects = detect_colored_objects(
        frame.image_bgr,
        calibration_matrix=parameters["calibration_matrix"],
        pick_x_max_mm=parameters["pick_region_world_mm"]["x_max"],
    )
    routes = parameters["drop_poses_mm"]
    detected_ids = {_class_id(item) for item in objects}
    if detected_ids != set(parameters["classes"]):
        raise RuntimeError(f"构件类别不完整：{sorted(detected_ids)}")
    ctx.robot.home()
    for item in objects:
        class_id = _class_id(item)
        pick_and_place(
            ctx,
            pick_xy=item.world_xy_mm,
            drop_xyz=routes[class_id],
            pick_z_mm=parameters["pick_z_mm"],
            safe_z_mm=parameters["safe_z_mm"],
            speed=12,
        )
        ctx.checkpoint(f"完成 {class_id} 分仓")
    ctx.robot.home()
