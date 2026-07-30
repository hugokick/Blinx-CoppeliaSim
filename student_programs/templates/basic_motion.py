def main(ctx):
    ctx.log("基础点位运动开始")
    ctx.robot.home()
    ctx.robot.move_world(100, 60, 120, speed=15)
    ctx.robot.move_world(120, 20, 120, speed=15)
    ctx.robot.move_world(100, -20, 120, speed=15)
    ctx.robot.home()
    ctx.log("基础点位运动完成")
