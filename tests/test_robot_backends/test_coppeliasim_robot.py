"""Tests for CoppeliaSimRobotBackend."""
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from robot_backends.coppeliasim_robot import CoppeliaSimRobotBackend


# ------------------------------------------------------------------
# Fakes
# ------------------------------------------------------------------

class FakeSimClient:
    def __init__(self, ik_state=None, fallback_state=None):
        self._sim = FakeSim()
        self._simIK = FakeSimIK(
            sim=self._sim,
            state=ik_state,
            fallback_state=fallback_state,
        )

    def getObject(self, name):
        return self._sim

    def require(self, name):
        if name == "simIK":
            return self._simIK
        return self._sim


class FakeSim:
    # Pre-assigned handles for known paths (deterministic)
    _KNOWN_HANDLES = {
        "/BLX_base_link": 15,
        "/BLX_tool_suction": 40,
        "/BLX_target": 20,
        "/BLX_joint1": 21,
        "/BLX_joint2": 22,
        "/BLX_joint3": 23,
        "/BLX_joint4": 24,
        "/BLX_joint5": 25,
        "/BLX_joint6": 26,
    }

    def __init__(self):
        self.joint_positions = {}
        self.tool_color = None
        self.colorcomponent_ambient_diffuse = 0
        self._handles = dict(self._KNOWN_HANDLES)
        self._next_handle = 100
        self._tip_pos = [0.45, 0.0, 0.65]
        self._tip_orient = [0.0, 0.0, 0.0]
        self.target_positions = {}   # (handle, ref) -> [x,y,z]
        self.target_orientations = {}  # (handle, ref) -> [rx,ry,rz]

    def getObject(self, path, opts=None):
        if path not in self._handles:
            self._handles[path] = self._next_handle
            self._next_handle += 1
        return self._handles[path]

    def setJointPosition(self, handle, pos_rad):
        self.joint_positions[handle] = pos_rad

    def getJointPosition(self, handle):
        return self.joint_positions.get(handle, 0.0)

    def setObjectColor(self, handle, idx, component, rgb):
        self.tool_color = rgb

    def getObjectPosition(self, handle, ref):
        return list(self._tip_pos)

    def getObjectOrientation(self, handle, ref):
        return list(self._tip_orient)

    def setObjectPosition(self, handle, position, relative_to):
        self.target_positions[(handle, relative_to)] = position

    def setObjectOrientation(self, handle, orientation, relative_to):
        self.target_orientations[(handle, relative_to)] = orientation


class FakeSimIK:
    """Minimal position-only simIK contract used by the BLX backend."""
    constraint_position = 1

    def __init__(self, sim, state=None, fallback_state=None):
        self._sim = sim
        self._state = state
        self._fallback_state = fallback_state
        self.created = False
        self.last_handle_group_args = None
        self.find_configs_calls = []
        self.ik_joint_positions = {}
        self.sync_to_calls = []
        self.apply_calls = 0

    def createEnvironment(self):
        self.created = True
        return 100

    def createGroup(self, env):
        return 200

    def addElementFromScene(self, env, group, base, tip, target, constraint):
        # Return 3 values matching real CoppeliaSim API:
        # (ik_element, scene_to_ik_mapping, extra_info)
        mapping = {}
        for i in range(6):
            mapping[21 + i] = 201 + i
        return 300, mapping, {}

    def handleGroup(self, env, group, options):
        self.last_handle_group_args = (env, group, options)
        if self._state is None:
            return 0, {}
        return 1, {}

    def findConfigs(self, env, group, joints, params):
        self.find_configs_calls.append((env, group, list(joints), dict(params)))
        if self._fallback_state is None:
            return []
        return [list(self._fallback_state)]

    def setJointPosition(self, env, joint, position):
        self.ik_joint_positions[joint] = position

    def syncToSim(self, env, groups):
        self.sync_to_calls.append((env, list(groups)))
        if self._fallback_state is None:
            return
        for i, position in enumerate(self._fallback_state):
            self._sim.joint_positions[21 + i] = position

    def applyIkEnvironmentToScene(self, env, group, apply_all):
        self.apply_calls += 1
        if self._state is None:
            return
        for i, position in enumerate(self._state):
            self._sim.joint_positions[21 + i] = position


# ------------------------------------------------------------------
# Shared fixture
# ------------------------------------------------------------------

def _make_settings():
    return {
        "joint_paths": [
            "/BLX_joint1", "/BLX_joint2", "/BLX_joint3",
            "/BLX_joint4", "/BLX_joint5", "/BLX_joint6",
        ],
        "tool_path": "/BLX_tool_suction",
        "base_path": "/BLX_base_link",
        "tip_path": "/BLX_tool_suction",
        "target_path": "/BLX_target",
        "home_angles": [0, 0, 0, 0, 0, 0],
        "joint5_offset_deg": 0,
        "ik_search_time": 0.25,
        "ik_search_precision": 0.001,
        "ik_metric": [1, 1, 1, 0.1],
    }


def _make_backend(ik_state=None, fallback_state=None):
    client = FakeSimClient(
        ik_state=ik_state,
        fallback_state=fallback_state,
    )
    backend = CoppeliaSimRobotBackend(settings=_make_settings(), sim_client=client)
    return backend, client


# ------------------------------------------------------------------
# Tests: joint control (unchanged)
# ------------------------------------------------------------------

def test_home_sets_all_joints():
    backend, client = _make_backend()
    sim = client.getObject("sim")
    backend.home()
    for i in range(6):
        handle = sim._handles[f"/BLX_joint{i+1}"]
        assert abs(sim.joint_positions[handle]) < 1e-9


def test_move_angle_single_joint():
    backend, client = _make_backend()
    sim = client.getObject("sim")
    backend.home()
    backend.move_angle(1, 50, 30)
    handle1 = sim._handles["/BLX_joint1"]
    assert abs(sim.joint_positions[handle1] - math.radians(30)) < 1e-9
    assert backend.current_angles[0] == 30


def test_blx_joint5_has_no_mechanical_offset():
    backend, client = _make_backend()
    sim = client.getObject("sim")
    backend.home()
    backend.move_angle(5, 50, 0)
    handle5 = sim._handles["/BLX_joint5"]
    assert abs(sim.joint_positions[handle5]) < 1e-9


def test_blx_joint5_nonzero_angle_is_not_shifted():
    backend, client = _make_backend()
    sim = client.getObject("sim")
    backend.home()
    backend.move_angle(5, 50, 90)
    handle5 = sim._handles["/BLX_joint5"]
    assert abs(sim.joint_positions[handle5] - math.radians(90)) < 1e-9


def test_other_joints_no_offset():
    backend, client = _make_backend()
    sim = client.getObject("sim")
    backend.home()
    for joint_id in [1, 2, 3, 4, 6]:
        backend.move_angle(joint_id, 50, 45)
        handle = sim._handles[f"/BLX_joint{joint_id}"]
        assert abs(sim.joint_positions[handle] - math.radians(45)) < 1e-9


def test_move_angle_all():
    backend, client = _make_backend()
    sim = client.getObject("sim")
    backend.home()
    backend.move_angle_all(10, 20, 30, 40, 50, 60, 50)
    for i, expected_deg in enumerate([10, 20, 30, 40, 50, 60]):
        handle = sim._handles[f"/BLX_joint{i+1}"]
        assert abs(sim.joint_positions[handle] - math.radians(expected_deg)) < 1e-9
    assert backend.current_angles == [10, 20, 30, 40, 50, 60]


# ------------------------------------------------------------------
# Tests: pump
# ------------------------------------------------------------------

def test_pump_on():
    backend, client = _make_backend()
    sim = client.getObject("sim")
    backend.pump_on()
    assert backend._pump_state is True
    assert sim.tool_color == [0.0, 1.0, 0.0]


def test_pump_off():
    backend, client = _make_backend()
    sim = client.getObject("sim")
    backend.pump_on()
    backend.pump_off()
    assert backend._pump_state is False
    assert sim.tool_color == [1.0, 0.0, 0.0]


# ------------------------------------------------------------------
# Tests: positive_solution
# ------------------------------------------------------------------

def test_positive_solution_returns_cartesian_pose():
    backend, client = _make_backend()
    result = backend.positive_solution()
    assert len(result) == 6
    assert abs(result[0] - 450.0) < 1
    assert abs(result[1] - 0.0) < 1
    assert abs(result[2] - 650.0) < 1


def test_positive_solution_six_elements():
    backend, _ = _make_backend()
    pose = backend.positive_solution()
    assert isinstance(pose, list)
    assert len(pose) == 6


def test_simulation_pose_tolerance_uses_millimetres() -> None:
    assert CoppeliaSimRobotBackend._pose_within_mm(
        (0.0400, -0.0450, 0.1110),
        (0.0402, -0.0451, 0.1112),
        0.25,
    ) is True
    assert CoppeliaSimRobotBackend._pose_within_mm(
        (0.0400, -0.0450, 0.1110),
        (0.0403, -0.0450, 0.1110),
        0.25,
    ) is False


# ------------------------------------------------------------------
# Tests: state maintenance
# ------------------------------------------------------------------

def test_state_maintained_across_calls():
    backend, _ = _make_backend()
    backend.home()
    backend.move_angle(1, 50, 15)
    backend.move_angle(3, 50, 25)
    assert backend.current_angles == [15, 0, 25, 0, 0, 0]
    backend.move_angle(6, 50, 60)
    assert backend.current_angles == [15, 0, 25, 0, 0, 60]


# ------------------------------------------------------------------
# Tests: IK config
# ------------------------------------------------------------------

def test_backend_settings_include_ik_paths():
    from config.backend import get_backend_settings
    settings = get_backend_settings()
    sim_settings = settings["sim"]
    assert sim_settings["base_path"] == "/BLX_base_link"
    assert sim_settings["tip_path"] == "/BLX_tool_suction"
    assert sim_settings["target_path"] == "/BLX_target"
    assert sim_settings["ik_search_time"] == 0.25
    assert sim_settings["ik_search_precision"] == 0.001
    assert sim_settings["ik_metric"] == [1, 1, 1, 0.1]


# ------------------------------------------------------------------
# Tests: IK move_coordinate_all
# ------------------------------------------------------------------

def test_move_coordinate_all_applies_ik_solution():
    """IK returns a valid joint state -> joints updated, current_angles set."""
    ik_state = [0.0, 0.1, 0.2, 0.3, -1.2, 0.5]
    backend, client = _make_backend(ik_state=ik_state)
    backend.move_coordinate_all(200, 0, 180, 180, 0, 0, 50)

    sim = client.getObject("sim")
    assert len(sim.joint_positions) >= 6
    assert len(backend.current_angles) == 6
    # joint1: 0.0 rad -> 0.0 deg
    assert abs(backend.current_angles[0] - math.degrees(0.0)) < 1e-6
    # BLX joint5 has no mechanical offset.
    expected_j5 = math.degrees(-1.2)
    assert abs(backend.current_angles[4] - expected_j5) < 1e-6


def test_move_coordinate_all_raises_when_ik_has_no_solution():
    """IK returns None -> RuntimeError, joints not modified."""
    backend, client = _make_backend(ik_state=None)
    sim = client.getObject("sim")
    before = dict(sim.joint_positions)
    try:
        backend.move_coordinate_all(200, 0, 180, 180, 0, 0, 50)
        assert False, "Expected RuntimeError"
    except RuntimeError as exc:
        assert "ik failed" in str(exc).lower()
    assert sim.joint_positions == before


def test_move_coordinate_all_falls_back_to_global_configuration_search():
    """A distant but reachable target uses findConfigs after local IK fails."""
    fallback_state = [0.2, -0.4, 0.6, -0.8, 1.0, -1.2]
    backend, client = _make_backend(
        ik_state=None,
        fallback_state=fallback_state,
    )

    backend.move_coordinate_all(55, -55, 100, 0, 0, 0, 15)

    simIK = client._simIK
    assert len(simIK.find_configs_calls) == 1
    env, group, joints, params = simIK.find_configs_calls[0]
    assert (env, group, joints) == (
        100,
        200,
        [201, 202, 203, 204, 205, 206],
    )
    assert params["maxDist"] == 0.5
    assert params["maxTime"] == 0.25
    assert params["pMetric"] == [1, 1, 1, 0.1]
    assert params["findAlt"] is False
    assert params["findMultiple"] is False
    assert simIK.sync_to_calls == [(100, [200])]
    assert backend.current_angles == [
        math.degrees(value) for value in fallback_state
    ]


def test_move_coordinate_all_applies_ik_environment_to_scene():
    """A successful position-only solve is applied to scene joints."""
    backend, client = _make_backend(ik_state=[0.0]*6)
    simIK = client._simIK

    backend.move_coordinate_all(200, 0, 180, 180, 0, 0, 50)

    assert simIK.last_handle_group_args == (100, 200, {"syncWorlds": True})
    assert simIK.apply_calls == 1


def test_ensure_ik_extracts_ik_joint_handles():
    """_ensure_ik must populate _ik_joint_handles from the mapping."""
    backend, _ = _make_backend(ik_state=[0.0]*6)
    backend.move_coordinate_all(200, 0, 180, 180, 0, 0, 50)
    assert backend._ik_joint_handles == [201, 202, 203, 204, 205, 206]


def test_ensure_ik_lazy_init():
    backend, _ = _make_backend(ik_state=[0.0]*6)
    assert backend._simIK is None
    backend.move_coordinate_all(200, 0, 180, 180, 0, 0, 50)
    assert backend._simIK is not None
    assert backend._ik_env == 100
    assert backend._ik_group == 200


def test_ensure_ik_idempotent():
    backend, _ = _make_backend(ik_state=[0.0]*6)
    backend.move_coordinate_all(200, 0, 180, 180, 0, 0, 50)
    env1 = backend._ik_env
    backend.move_coordinate_all(210, -20, 190, 180, 0, 0, 50)
    assert backend._ik_env is env1


# ------------------------------------------------------------------
# Tests: coordinate frame consistency
# ------------------------------------------------------------------

def test_move_coordinate_all_sets_target_in_world_frame():
    """Target position/orientation must be set in world frame (-1),
    matching positive_solution() which also reads world frame."""
    backend, client = _make_backend(ik_state=[0.0]*6)
    sim = client.getObject("sim")
    backend.move_coordinate_all(200, 50, 180, 90, 45, -30, 50)

    target_h = sim._handles["/BLX_target"]

    # Must use world frame (ref = -1), not base-relative
    pos_key = (target_h, -1)
    assert pos_key in sim.target_positions, \
        "Target must be set in world frame (ref=-1)"
    pos = sim.target_positions[pos_key]
    assert abs(pos[0] - 0.2) < 1e-6   # 200mm -> 0.2m
    assert abs(pos[1] - 0.05) < 1e-6  # 50mm -> 0.05m
    assert abs(pos[2] - 0.18) < 1e-6  # 180mm -> 0.18m

    # RX/RY/RZ are intentionally ignored in position-only mode.  The target
    # keeps the current TCP orientation copied during IK initialisation.
    orient = sim.target_orientations[pos_key]
    assert orient == [0.0, 0.0, 0.0]


def test_move_coordinate_all_positive_solution_same_frame():
    """move_coordinate_all input and positive_solution output must use
    the same world-frame convention, even when base is not at origin.

    The FakeSim always returns tip at [0.45, 0.0, 0.65] m.
    move_coordinate_all sets target to that same world position.
    positive_solution reads world position.
    Both should agree on the mm values.
    """
    backend, client = _make_backend(ik_state=[0.0]*6)
    # Command to the exact position our FakeSim returns
    backend.move_coordinate_all(450, 0, 650, 0, 0, 0, 50)
    pose = backend.positive_solution()
    # Both read/write in world frame -> should match
    assert abs(pose[0] - 450.0) < 1
    assert abs(pose[1] - 0.0) < 1
    assert abs(pose[2] - 650.0) < 1


# ------------------------------------------------------------------
# Tests: close() IK cleanup
# ------------------------------------------------------------------

def test_close_clears_ik_state():
    backend, _ = _make_backend(ik_state=[0.0]*6)
    backend.move_coordinate_all(200, 0, 180, 180, 0, 0, 50)
    assert backend._simIK is not None

    backend.close()
    assert backend._sim is None
    assert backend._client is None
    assert backend._joint_handles is None
    assert backend._tool_handle is None
    assert backend._simIK is None
    assert backend._ik_env is None
    assert backend._ik_group is None
    assert backend._ik_base_handle is None
    assert backend._ik_tip_handle is None
    assert backend._ik_target_handle is None
    assert backend._ik_mapping is None
    assert backend._ik_joint_handles is None


def test_close_then_reconnect_works():
    backend, _ = _make_backend(ik_state=[0.1]*6)
    backend.move_coordinate_all(200, 0, 180, 180, 0, 0, 50)
    backend.close()
    assert backend._sim is None

    new_client = FakeSimClient(ik_state=[0.1]*6)
    backend._client = new_client
    backend.home()
    assert backend._sim is not None
    assert backend._joint_handles is not None
