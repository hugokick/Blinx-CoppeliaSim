import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from validation.sim_smoke_check import run_sim_smoke_check


class FakeBackend:
    def __init__(self):
        self.calls = []

    def home(self):
        self.calls.append("home")

    def move_angle(self, joint_id, speed, value):
        self.calls.append(("move_angle", joint_id, speed, value))

    def pump_on(self):
        self.calls.append("pump_on")

    def pump_off(self):
        self.calls.append("pump_off")

    def close(self):
        self.calls.append("close")


def test_sim_smoke_check_runs_core_actions():
    backend = FakeBackend()
    result = run_sim_smoke_check(backend=backend)
    assert result["status"] == "PASS"
    assert "home" in backend.calls
    assert ("move_angle", 1, 50, 10) in backend.calls
    assert "pump_on" in backend.calls
    assert "pump_off" in backend.calls
    assert "close" in backend.calls


def test_sim_smoke_check_reports_failure():
    class BrokenBackend:
        def home(self):
            raise RuntimeError("connection refused")
        def close(self):
            pass

    result = run_sim_smoke_check(backend=BrokenBackend())
    assert result["status"] == "FAIL"
    assert "connection refused" in result["message"]


def test_sim_smoke_check_backend_creation_failure(monkeypatch):
    """When create_robot_backend() raises, smoke check returns FAIL."""
    import validation.sim_smoke_check as module

    def fake_create():
        raise RuntimeError("CoppeliaSim not running")

    monkeypatch.setattr(module, "create_robot_backend", fake_create, raising=False)
    # Patch the import inside run_sim_smoke_check by injecting into module
    import robot_backends.factory as factory_mod
    original = factory_mod.create_robot_backend
    factory_mod.create_robot_backend = fake_create
    try:
        result = run_sim_smoke_check(backend=None)
        assert result["status"] == "FAIL"
        assert "backend creation failed" in result["message"]
    finally:
        factory_mod.create_robot_backend = original
