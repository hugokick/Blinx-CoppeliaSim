from types import SimpleNamespace

import pytest

from vision_platform.experiments.capabilities import check_capabilities


def _application(
    *,
    camera=True,
    backend="sim",
    camera_backend="sim",
    tool=True,
    sim=True,
):
    return SimpleNamespace(
        camera=object() if camera else None,
        robot=object(),
        tool=object() if tool else None,
        sim=object() if sim else None,
        config=SimpleNamespace(
            robot_backend=backend,
            camera_backend=camera_backend,
        ),
    )


def test_sim_application_satisfies_first_batch_capabilities():
    report = check_capabilities(
        _application(),
        (
            "robot.home",
            "robot.pose",
            "robot.move_world",
            "tool.suction",
            "camera.rgb",
            "experiment.info",
            "scene.probe",
        ),
    )

    assert report.ready
    assert report.missing == ()


def test_real_backend_is_rejected_even_when_objects_exist():
    report = check_capabilities(
        _application(backend="real"),
        ("robot.home", "camera.rgb"),
    )

    assert not report.ready
    assert "robot.home" in report.missing
    assert report.reasons["robot.home"] == "V2.2 禁止真实机器人后端"


def test_missing_camera_has_stable_reason():
    report = check_capabilities(
        _application(camera=False),
        ("camera.rgb",),
    )

    assert report.missing == ("camera.rgb",)
    assert report.reasons["camera.rgb"] == "当前应用没有可用相机"


@pytest.mark.parametrize(
    ("application", "capability", "reason"),
    [
        (_application(tool=False), "tool.suction", "当前应用没有可用吸盘"),
        (
            _application(sim=False),
            "scene.probe",
            "当前应用没有 CoppeliaSim 场景连接",
        ),
    ],
)
def test_missing_runtime_component_is_not_silently_accepted(
    application,
    capability,
    reason,
):
    report = check_capabilities(application, (capability,))

    assert report.missing == (capability,)
    assert report.reasons[capability] == reason


def test_suction_readiness_failure_has_stable_missing_reason():
    class UnreadySuction:
        def __init__(self):
            self.validate_calls = 0

        def validate(self):
            self.validate_calls += 1
            raise KeyError("/missing/Pickables")

    tool = UnreadySuction()
    application = _application()
    application.tool = tool

    report = check_capabilities(application, ("tool.suction",))

    assert tool.validate_calls == 1
    assert report.missing == ("tool.suction",)
    assert report.reasons["tool.suction"] == (
        "吸盘 TCP 或可抓取集合不可用"
    )


def test_unknown_capability_is_not_silently_accepted():
    report = check_capabilities(_application(), ("robot.fly",))

    assert report.missing == ("robot.fly",)
    assert "未注册能力" in report.reasons["robot.fly"]


def test_sim_camera_and_scene_expose_profile_capabilities():
    report = check_capabilities(
        _application(),
        ("camera.profile", "lighting.profile"),
    )

    assert report.ready
    assert report.available == ("camera.profile", "lighting.profile")


@pytest.mark.parametrize(
    ("application", "reason"),
    [
        (
            _application(camera_backend="replay"),
            "视觉配置仅支持 CoppeliaSim 相机后端",
        ),
        (
            _application(sim=False),
            "当前应用没有 CoppeliaSim 场景连接",
        ),
        (
            _application(camera=False),
            "当前应用没有可用相机",
        ),
    ],
)
def test_unavailable_profile_runtime_has_stable_reasons(application, reason):
    report = check_capabilities(
        application,
        ("camera.profile", "lighting.profile"),
    )

    assert report.missing == ("camera.profile", "lighting.profile")
    assert report.reasons == {
        "camera.profile": reason,
        "lighting.profile": reason,
    }


def test_profile_capability_check_does_not_probe_arbitrary_scene_objects():
    class NoSceneProbe:
        def __getattr__(self, name):
            raise AssertionError(f"unexpected scene probe: {name}")

    application = _application()
    application.sim = NoSceneProbe()

    report = check_capabilities(
        application,
        ("camera.profile", "lighting.profile"),
    )

    assert report.ready


@pytest.mark.parametrize("capability", ["camera.profile", "lighting.profile"])
def test_profile_capabilities_are_an_indivisible_pair(capability):
    report = check_capabilities(_application(), (capability,))

    assert report.missing == (capability,)
    assert report.reasons[capability] == (
        "视觉配置能力必须同时声明 camera.profile 和 lighting.profile"
    )


def test_profile_pair_accepts_generator_without_touching_runtime_objects():
    class PresenceOnly:
        def __getattribute__(self, name):
            raise AssertionError(f"unexpected runtime inspection: {name}")

    application = _application()
    application.sim = PresenceOnly()
    application.camera = PresenceOnly()

    report = check_capabilities(
        application,
        (name for name in ("camera.profile", "lighting.profile")),
    )

    assert report.ready


def test_sim_camera_profiles_expose_controlled_vision2d_analysis():
    report = check_capabilities(
        _application(),
        (
            "camera.rgb",
            "camera.profile",
            "lighting.profile",
            "vision2d.analysis",
        ),
    )

    assert report.ready
    assert report.available[-1] == "vision2d.analysis"


@pytest.mark.parametrize(
    ("application", "reason"),
    [
        (
            _application(camera_backend="replay"),
            "二维视觉分析仅支持 CoppeliaSim 相机后端",
        ),
        (
            _application(sim=False),
            "当前应用没有 CoppeliaSim 场景连接",
        ),
        (
            _application(camera=False),
            "当前应用没有可用相机",
        ),
    ],
)
def test_vision2d_analysis_rejects_unavailable_runtime(application, reason):
    report = check_capabilities(
        application,
        (
            "camera.rgb",
            "camera.profile",
            "lighting.profile",
            "vision2d.analysis",
        ),
    )

    assert "vision2d.analysis" in report.missing
    assert report.reasons["vision2d.analysis"] == reason


def test_vision2d_analysis_requires_camera_and_profile_declarations():
    report = check_capabilities(
        _application(),
        ("vision2d.analysis",),
    )

    assert report.missing == ("vision2d.analysis",)
    assert report.reasons["vision2d.analysis"] == (
        "二维视觉分析必须同时声明相机与成对视觉配置能力"
    )


def test_code_routing_requires_complete_sim_camera_robot_and_tool_contract() -> None:
    required = (
        "camera.rgb", "camera.profile", "lighting.profile", "vision2d.code_routing",
        "robot.home", "robot.pose", "robot.move_world", "tool.suction", "scene.probe",
    )
    ready = SimpleNamespace(
        config=SimpleNamespace(camera_backend="sim", robot_backend="sim"),
        sim=object(), camera=object(), robot=object(), tool=object(),
    )
    assert check_capabilities(ready, required).ready
    for field in ("camera", "robot", "tool", "sim"):
        broken = SimpleNamespace(**ready.__dict__)
        setattr(broken, field, None)
        assert "vision2d.code_routing" in check_capabilities(broken, required).missing
    replay = SimpleNamespace(**ready.__dict__)
    replay.config = SimpleNamespace(camera_backend="replay", robot_backend="sim")
    assert "vision2d.code_routing" in check_capabilities(replay, required).missing
