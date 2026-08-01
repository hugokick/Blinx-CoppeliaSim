def main(ctx):
    parameters = ctx.experiment.info()["public_parameters"]
    if parameters["analysis_focus"] != "geometry":
        raise RuntimeError("实验未发布周长与面积配置")
    profile_id = parameters["vision2d"]["profile_id"]
    if profile_id not in parameters["allowed_profile_ids"]:
        raise RuntimeError("二维分析配置档超出公开白名单")

    primary_error = None
    try:
        ctx.camera.apply_profile(profile_id)
        analysis = ctx.vision2d.analyze()
        if analysis.status not in {"PASS", "PARTIAL"}:
            raise RuntimeError("二维分析没有产生可用目标")
        if len(analysis.targets) != parameters["expected_target_count"]:
            raise RuntimeError("检测目标数量与实验发布值不一致")

        for target in analysis.targets:
            perimeter_mm = target["perimeter_mm"]
            area_mm2 = target["area_mm2"]
            if perimeter_mm is None or area_mm2 is None:
                raise RuntimeError("实验没有返回毫米周长或面积")
            values = (
                target["perimeter_px"],
                target["area_px2"],
                perimeter_mm,
                area_mm2,
                target["circularity"],
                target["aspect_ratio"],
            )
            if any(value <= 0 for value in values):
                raise RuntimeError("几何测量值必须为正数")
            ctx.log(
                f"{target['detection_id']} {target['shape']}: "
                f"perimeter={target['perimeter_px']:.1f} px/"
                f"{perimeter_mm:.1f} mm, "
                f"area={target['area_px2']:.1f} px2/"
                f"{area_mm2:.1f} mm2, "
                f"circularity={target['circularity']:.3f}, "
                f"aspect={target['aspect_ratio']:.3f}"
            )
        ctx.checkpoint(f"已记录 {len(analysis.targets)} 个目标几何量")
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
