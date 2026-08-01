def main(ctx):
    info = ctx.experiment.info()
    parameters = info["public_parameters"]
    if parameters["analysis_focus"] != "size":
        raise RuntimeError("实验未发布尺寸测量配置")
    profile_id = parameters["vision2d"]["profile_id"]
    if profile_id not in parameters["allowed_profile_ids"]:
        raise RuntimeError("二维分析配置档超出公开白名单")

    primary_error = None
    try:
        ctx.camera.apply_profile(profile_id)
        analysis = ctx.vision2d.analyze()
        if analysis.status not in {"PASS", "PARTIAL"}:
            raise RuntimeError("二维分析没有产生可用目标")
        expected_count = parameters["expected_target_count"]
        if len(analysis.targets) != expected_count:
            raise RuntimeError("检测目标数量与实验发布值不一致")
        expected_shapes = set(parameters["expected_shapes"])
        actual_shapes = {target["shape"] for target in analysis.targets}
        if actual_shapes != expected_shapes:
            raise RuntimeError("检测形状与实验发布值不一致")

        for target in analysis.targets:
            size_mm = target["size_mm"]
            if size_mm is None or target["area_mm2"] is None:
                raise RuntimeError("实验没有返回显式毫米尺寸")
            ctx.log(
                f"{target['detection_id']} {target['shape']}: "
                f"{target['long_side_px']:.1f} x "
                f"{target['short_side_px']:.1f} px; "
                f"{size_mm[0]:.1f} x {size_mm[1]:.1f} mm"
            )
        ctx.checkpoint(f"已记录 {len(analysis.targets)} 个构件尺寸")
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
