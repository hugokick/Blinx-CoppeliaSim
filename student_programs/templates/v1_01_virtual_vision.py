def main(ctx):
    info = ctx.experiment.info()
    parameters = info["public_parameters"]
    order = tuple(parameters["comparison_order"])
    allowed = set(parameters["allowed_profile_ids"])
    if set(order) - allowed:
        raise RuntimeError("实验配置档顺序超出公开白名单")

    primary_error = None
    try:
        for profile_id in order:
            state = ctx.camera.apply_profile(profile_id)
            frame = ctx.camera.capture()
            ctx.log(
                f"{profile_id}: {frame.width}x{frame.height}, "
                f"FOV={state.perspective_angle_deg:.1f} deg, "
                f"camera_z={state.camera_rig_z_m:.2f} m"
            )
            ctx.checkpoint(f"已记录 {profile_id} 配置档")
    except Exception as error:
        primary_error = error
        raise
    finally:
        try:
            ctx.camera.reset_profile()
        except Exception as reset_error:
            if primary_error is None:
                raise
            primary_error.add_note(
                f"reset_profile also failed: {reset_error}"
            )
