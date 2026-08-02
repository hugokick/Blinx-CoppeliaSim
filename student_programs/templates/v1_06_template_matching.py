def main(ctx):
    parameters = ctx.experiment.info()["public_parameters"]
    if parameters["template_id"] != "v1_06_red_rectangle":
        raise RuntimeError("实验模板标识不是已发布的 V1-06 模板")
    if parameters["method"] != "TM_CCOEFF_NORMED":
        raise RuntimeError("实验匹配方法不是已发布方法")
    if parameters["threshold"] != 0.72:
        raise RuntimeError("实验阈值不是已发布阈值")
    profile_id = parameters["baseline_profile_id"]
    if profile_id not in parameters["allowed_profile_ids"]:
        raise RuntimeError("视觉配置档超出公开白名单")

    primary_error = None
    try:
        ctx.camera.apply_profile(profile_id)
        result = ctx.vision2d.template_match()
        if result.template_id != parameters["template_id"]:
            raise RuntimeError("返回模板标识与发布配置不一致")
        if result.method != parameters["method"]:
            raise RuntimeError("返回匹配方法与发布配置不一致")
        if result.search_roi_px != tuple(parameters["search_roi_px"]):
            raise RuntimeError("返回搜索区域与发布配置不一致")
        if result.bbox_px is None or result.center_px is None:
            raise RuntimeError("模板匹配没有产生像素坐标证据")
        ctx.log(
            f"{result.template_id}: matched={result.matched}, "
            f"score={result.score:.3f}, bbox={result.bbox_px}, "
            f"center={result.center_px}"
        )
        ctx.checkpoint("已记录模板匹配结果和三层视觉证据")
    except Exception as error:
        primary_error = error
        raise
    finally:
        try:
            ctx.camera.reset_profile()
        except Exception as reset_error:
            if primary_error is None:
                raise
            primary_error.add_note(f"reset_profile also failed: {reset_error}")
