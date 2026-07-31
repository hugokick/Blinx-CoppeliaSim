from student_programs.templates.r1_common import _finite_vector


def main(ctx):
    parameters = ctx.experiment.info()["public_parameters"]
    teach_points = parameters["teach_points_mm"]
    if not isinstance(teach_points, (list, tuple)) or len(teach_points) != 4:
        raise RuntimeError("teach_points_mm 必须恰好包含四个示教点")
    try:
        points = [
            _finite_vector(point, 3, f"teach_points_mm[{index}]")
            for index, point in enumerate(teach_points)
        ]
    except ValueError as error:
        raise RuntimeError("示教点必须是有限三维坐标") from error
    ctx.robot.home()
    for index, point in enumerate(points, start=1):
        ctx.robot.move_world(*point, speed=12)
        ctx.log(f"到达示教点 {index}: {ctx.robot.pose()}")
        ctx.checkpoint(f"示教点 {index}")
    ctx.robot.home()
