from student_programs.templates.r1_common import (
    detect_colored_objects,
    pick_and_place,
)


def main(ctx):
    parameters = ctx.experiment.info()["public_parameters"]
    frame = ctx.camera.capture()
    objects = detect_colored_objects(
        frame.image_bgr,
        calibration_matrix=parameters["calibration_matrix"],
        pick_x_max_mm=parameters["pick_region_world_mm"]["x_max"],
    )
    if len(objects) != 6:
        raise RuntimeError(f"期望识别 6 个码垛物体，实际 {len(objects)} 个")
    ctx.robot.home()
    for index, (item, slot) in enumerate(
        zip(objects, parameters["stack_slots_mm"]),
        start=1,
    ):
        pick_and_place(
            ctx,
            pick_xy=item.world_xy_mm,
            drop_xyz=slot,
            pick_z_mm=parameters["pick_z_mm"],
            safe_z_mm=parameters["safe_z_mm"],
            speed=12,
        )
        ctx.checkpoint(f"完成第 {index} 块码垛")
    ctx.robot.home()
