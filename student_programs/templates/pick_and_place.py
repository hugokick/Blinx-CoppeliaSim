def main(ctx):
    pick = (55, -55, 20)
    drop = (122, -66, 20)
    safe_z = 100

    ctx.log("单物体吸取与放置开始")
    ctx.robot.home()
    ctx.robot.move_world(pick[0], pick[1], safe_z, speed=15)
    ctx.robot.move_world(*pick, speed=8)
    ctx.tool.on()
    ctx.robot.move_world(pick[0], pick[1], safe_z, speed=12)
    ctx.robot.move_world(drop[0], drop[1], safe_z, speed=15)
    ctx.robot.move_world(*drop, speed=8)
    ctx.tool.off()
    ctx.robot.move_world(drop[0], drop[1], safe_z, speed=12)
    ctx.robot.home()
    ctx.log("单物体吸取与放置完成")
