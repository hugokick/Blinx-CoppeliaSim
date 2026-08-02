from student_programs.templates.r1_common import pick_and_place


def main(ctx):
    parameters = ctx.experiment.info()["public_parameters"]
    profile_id = parameters["baseline_profile_id"]
    if profile_id != "standard" or parameters["allowed_profile_ids"] != ["standard"]:
        raise RuntimeError("V1-07 只允许已发布的 standard 配置档")
    published = parameters["code_routing"]
    primary_error = None
    try:
        ctx.camera.apply_profile(profile_id)
        plan = ctx.vision2d.code_routes()
        if plan.status != "PASS" or len(plan.entries) != published["expected_count"]:
            raise RuntimeError("主机没有返回完整路线计划")
        if plan.safe_z_mm != published["safe_z_mm"] or plan.speed_mm_s != published["speed_mm_s"]:
            raise RuntimeError("路线计划安全参数与发布值不一致")
        published_routes = {
            route["entry_id"]: route for route in published["routes"]
        }
        if set(published_routes) != {entry.entry_id for entry in plan.entries}:
            raise RuntimeError("路线计划条目与发布白名单不一致")
        for entry in sorted(plan.entries, key=lambda item: item.entry_id):
            expected = published_routes[entry.entry_id]
            if (
                entry.part_id != expected["part_id"]
                or entry.code_type != expected["code_type"]
                or entry.payload != expected["payload"]
                or entry.route_id != expected["route_id"]
                or entry.drop_xyz_mm != tuple(expected["drop_xyz_mm"])
            ):
                raise RuntimeError(f"路线 {entry.entry_id} 未通过完整白名单校验")
        for entry in sorted(plan.entries, key=lambda item: item.entry_id):
            pick_and_place(
                ctx,
                pick_xy=entry.pick_xyz_mm[:2],
                drop_xyz=entry.drop_xyz_mm,
                pick_z_mm=entry.pick_xyz_mm[2],
                safe_z_mm=plan.safe_z_mm,
                speed=plan.speed_mm_s,
                use_command_xy=True,
            )
            ctx.log(f"{entry.part_id}: {entry.code_type}/{entry.payload} -> {entry.route_id}")
        ctx.robot.home()
        ctx.checkpoint(f"V1-07 完成，plan_id={plan.plan_id}")
    except Exception as error:
        primary_error = error
        raise
    finally:
        cleanup_errors = []
        for label, cleanup in (
            ("tool.off", ctx.tool.off),
            ("camera.reset_profile", ctx.camera.reset_profile),
        ):
            try:
                cleanup()
            except Exception as cleanup_error:
                cleanup_errors.append((label, cleanup_error))
        if primary_error is not None:
            for label, cleanup_error in cleanup_errors:
                primary_error.add_note(f"{label} also failed: {cleanup_error}")
        elif cleanup_errors:
            label, first_error = cleanup_errors[0]
            for later_label, later_error in cleanup_errors[1:]:
                first_error.add_note(f"{later_label} also failed: {later_error}")
            raise first_error
