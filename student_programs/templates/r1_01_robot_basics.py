from student_programs.templates.r1_common import _finite_vector


def main(ctx):
    info = ctx.experiment.info()
    observation = _finite_vector(
        info["public_parameters"]["observation_pose_mm"],
        3,
        "observation_pose_mm",
    )
    ctx.log(f"实验 {info['experiment_id']} 开始")
    ctx.robot.home()
    home_pose = ctx.robot.pose()
    ctx.log(f"回零后 TCP={home_pose}")
    ctx.robot.move_world(*observation, speed=12)
    ctx.checkpoint("观察六个关节和 TCP")
    ctx.robot.home()
    ctx.log("实验结束；硬件状态保持 PENDING_HARDWARE")
