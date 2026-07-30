def main(ctx):
    parameters = ctx.experiment.info()["public_parameters"]
    ctx.robot.home()
    for index, point in enumerate(parameters["teach_points_mm"], start=1):
        ctx.robot.move_world(*point, speed=12)
        ctx.log(f"到达示教点 {index}: {ctx.robot.pose()}")
        ctx.checkpoint(f"示教点 {index}")
    ctx.robot.home()
