from types import SimpleNamespace

import pytest

from vision_platform.experiments.capabilities import check_capabilities


def _application(*, camera=True, backend="sim", tool=True, sim=True):
    return SimpleNamespace(
        camera=object() if camera else None,
        robot=object(),
        tool=object() if tool else None,
        sim=object() if sim else None,
        config=SimpleNamespace(robot_backend=backend),
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


def test_unknown_capability_is_not_silently_accepted():
    report = check_capabilities(_application(), ("robot.fly",))

    assert report.missing == ("robot.fly",)
    assert "未注册能力" in report.reasons["robot.fly"]
