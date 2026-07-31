def main(ctx):
    pick = (55, -55, 20)
    drop = (122, -66, 20)
    safe_z = 110

    ctx.log("单物体吸取与放置开始")
    ctx.robot.home()
    ctx.robot.move_world(pick[0], pick[1], safe_z, speed=15)
    pick_approach_pose = ctx.robot.pose()
    ctx.robot.move_world(
        pick_approach_pose[0], pick_approach_pose[1], pick[2], speed=8
    )
    ctx.tool.on()
    pick_pose = ctx.robot.pose()
    ctx.robot.move_world(pick_pose[0], pick_pose[1], safe_z, speed=12)
    ctx.robot.move_world(drop[0], drop[1], safe_z, speed=15)
    drop_approach_pose = ctx.robot.pose()
    ctx.robot.move_world(
        drop_approach_pose[0], drop_approach_pose[1], drop[2], speed=8
    )
    ctx.tool.off()
    drop_pose = ctx.robot.pose()
    ctx.robot.move_world(drop_pose[0], drop_pose[1], safe_z, speed=12)
    ctx.robot.home()
    ctx.log("单物体吸取与放置完成")
