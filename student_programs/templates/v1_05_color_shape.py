def main(ctx):
    parameters = ctx.experiment.info()["public_parameters"]
    if parameters["analysis_focus"] != "appearance":
        raise RuntimeError("实验未发布颜色与形状识别配置")
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

        actual_shapes = set()
        actual_colors = set()
        for target in analysis.targets:
            color = target["color"]
            shape = target["shape"]
            if color == "unknown" or shape == "unknown":
                raise RuntimeError(
                    f"保留 unknown 标签：{target['detection_id']} "
                    f"{color}/{shape}"
                )
            contour = target["contour_px"]
            if len(contour) < 3 or target["vertex_count"] < 3:
                raise RuntimeError("目标轮廓或顶点字段无效")
            actual_colors.add(color)
            actual_shapes.add(shape)
            ctx.log(
                f"{target['detection_id']}: {color}/{shape}, "
                f"vertices={target['vertex_count']}, "
                f"contour_points={len(contour)}, "
                f"circularity={target['circularity']:.3f}, "
                f"aspect={target['aspect_ratio']:.3f}"
            )
        if actual_shapes != set(parameters["expected_shapes"]):
            raise RuntimeError("检测形状与实验发布值不一致")
        if actual_colors != set(parameters["expected_colors"]):
            raise RuntimeError("检测颜色与实验发布值不一致")
        ctx.checkpoint(f"已记录 {len(analysis.targets)} 个识别结果")
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
