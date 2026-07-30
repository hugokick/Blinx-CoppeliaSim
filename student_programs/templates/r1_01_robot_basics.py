def main(ctx):
    info = ctx.experiment.info()
    ctx.log(f"实验 {info['experiment_id']} 开始")
    ctx.robot.home()
    home_pose = ctx.robot.pose()
    ctx.log(f"回零后 TCP={home_pose}")
    observation = info["public_parameters"]["observation_pose_mm"]
    ctx.robot.move_world(*observation, speed=12)
    ctx.checkpoint("观察六个关节和 TCP")
    ctx.robot.home()
    ctx.log("实验结束；硬件状态保持 PENDING_HARDWARE")
