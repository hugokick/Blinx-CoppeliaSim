import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.backend import get_backend_settings
from robot_backends.factory import create_robot_backend


def test_backend_settings_default_to_sim(monkeypatch):
    monkeypatch.delenv("ROBOT_BACKEND", raising=False)
    settings = get_backend_settings()
    assert settings["backend"] == "sim"
    assert len(settings["sim"]["joint_paths"]) == 6
    assert settings["sim"]["joint_paths"][0] == "/BLX_joint1"
    assert settings["sim"]["joint_paths"][5] == "/BLX_joint6"
    assert settings["sim"]["tool_path"] == "/BLX_tool_suction"
    assert settings["sim"]["joint5_offset_deg"] == 0


def test_backend_settings_real(monkeypatch):
    monkeypatch.setenv("ROBOT_BACKEND", "real")
    settings = get_backend_settings()
    assert settings["backend"] == "real"


def test_factory_creates_real_backend(monkeypatch):
    monkeypatch.setenv("ROBOT_BACKEND", "real")
    from robot_backends import factory as factory_module

    class FakeRealBackend:
        pass

    monkeypatch.setattr(factory_module, "RealRobotBackend", FakeRealBackend, raising=False)
    backend = create_robot_backend()
    assert backend.__class__.__name__ == "FakeRealBackend"


def test_factory_creates_sim_backend(monkeypatch):
    monkeypatch.setenv("ROBOT_BACKEND", "sim")
    from robot_backends import factory as factory_module

    class FakeSimBackend:
        def __init__(self, settings):
            self.settings = settings

    monkeypatch.setattr(factory_module, "CoppeliaSimRobotBackend", FakeSimBackend, raising=False)
    backend = create_robot_backend()
    assert backend.__class__.__name__ == "FakeSimBackend"
    assert "joint_paths" in backend.settings
