def main(ctx):
    parameters = ctx.experiment.info()["public_parameters"]
    if parameters["analysis_focus"] != "pose":
        raise RuntimeError("实验未发布定位与角度配置")
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
            center = target["center_px"]
            box = target["rotated_box_px"]
            if len(center) != 2 or len(box) != 4:
                raise RuntimeError("目标中心或旋转框字段无效")
            angle = target["angle_deg"]
            flags = set(target["quality_flags"])
            if target["shape"] == "circle":
                if angle is not None:
                    raise RuntimeError("圆形角度必须保持未定义")
                if "ANGLE_UNDEFINED_FOR_CIRCLE" not in flags:
                    raise RuntimeError("圆形缺少角度未定义标志")
                angle_text = "undefined"
            else:
                if angle is None:
                    raise RuntimeError("非圆形目标缺少旋转角")
                angle_text = f"{angle:.1f} deg"
            ctx.log(
                f"{target['detection_id']} {target['shape']}: "
                f"center=({center[0]:.1f}, {center[1]:.1f}), "
                f"angle={angle_text}"
            )
        ctx.checkpoint(f"已记录 {len(analysis.targets)} 个目标位姿")
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
