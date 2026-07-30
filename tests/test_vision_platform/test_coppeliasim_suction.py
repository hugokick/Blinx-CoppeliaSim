import pytest

from vision_platform.errors import SuctionError
from vision_platform.robot.coppeliasim_suction import CoppeliaSimSuction


class FakeScene:
    handle_all = -1
    shapeintparam_static = 3003

    def __init__(self, *, tcp, objects, pickables=None):
        self.handles = {
            "/BLX_tool_suction": 10,
            "/VisionLab/Pickables": 20,
        }
        self.positions = {10: list(tcp), **{key: list(value) for key, value in objects.items()}}
        self.poses = {
            handle: [*position, 0.0, 0.0, 0.0, 1.0]
            for handle, position in self.positions.items()
        }
        self.pickables = list(pickables if pickables is not None else objects)
        self.parents = {handle: 20 for handle in objects}
        self.static = {handle: 0 for handle in objects}
        self.parent_calls = []
        self.static_calls = []
        self.reset_calls = []

    def getObject(self, path):
        return self.handles[path]

    def getObjectsInTree(self, root, object_type, options):
        assert (root, object_type, options) == (20, self.handle_all, 1)
        return list(self.pickables)

    def getObjectPosition(self, handle, relative_to):
        assert relative_to == -1
        return list(self.positions[handle])

    def getObjectPose(self, handle, relative_to):
        assert relative_to == -1
        return list(self.poses[handle])

    def getObjectParent(self, handle):
        return self.parents[handle]

    def getObjectInt32Param(self, handle, parameter):
        assert parameter == self.shapeintparam_static
        return self.static[handle]

    def setObjectInt32Param(self, handle, parameter, value):
        assert parameter == self.shapeintparam_static
        self.static[handle] = value
        self.static_calls.append((handle, parameter, value))

    def setObjectParent(self, handle, parent, keep_in_place):
        self.parents[handle] = parent
        self.parent_calls.append((handle, parent, keep_in_place))

    def resetDynamicObject(self, handle):
        self.reset_calls.append(handle)


def test_suction_attaches_nearest_eligible_object_and_preserves_world_pose():
    sim = FakeScene(
        tcp=(0.08, 0.00, 0.03),
        objects={
            101: (0.081, 0.002, 0.02),
            102: (0.12, 0.00, 0.02),
        },
    )
    before = sim.getObjectPose(101, -1)
    tool = CoppeliaSimSuction(sim, tcp_path="/BLX_tool_suction")

    evidence = tool.on()

    assert evidence.object_handle == 101
    assert sim.parents[101] == sim.handles["/BLX_tool_suction"]
    assert sim.parent_calls[-1] == (101, 10, True)
    assert sim.getObjectPose(101, -1) == before
    assert sim.static[101] == 1


def test_no_object_within_tolerance_reports_stable_attach_failure():
    sim = FakeScene(
        tcp=(0.08, 0.00, 0.03),
        objects={101: (0.20, 0.00, 0.02)},
    )
    tool = CoppeliaSimSuction(sim, max_attach_distance_m=0.025)

    with pytest.raises(SuctionError) as exc:
        tool.on()

    assert exc.value.code == "SUCTION_ATTACH_FAILED"
    assert sim.parent_calls == []


def test_second_on_is_idempotent_and_returns_same_attachment():
    sim = FakeScene(
        tcp=(0.08, 0.00, 0.03),
        objects={101: (0.081, 0.002, 0.02)},
    )
    tool = CoppeliaSimSuction(sim)

    first = tool.on()
    second = tool.on()

    assert first == second
    assert sim.parent_calls == [(101, 10, True)]


def test_off_restores_original_parent_and_dynamic_state():
    sim = FakeScene(
        tcp=(0.08, 0.00, 0.03),
        objects={101: (0.081, 0.002, 0.02)},
    )
    tool = CoppeliaSimSuction(sim)
    tool.on()

    tool.off()
    tool.off()

    assert sim.parents[101] == 20
    assert sim.static[101] == 0
    assert sim.parent_calls == [(101, 10, True), (101, 20, True)]
    assert sim.reset_calls[-1] == 101
    assert tool.is_attached() is False


def test_objects_outside_pickables_tree_are_never_attached():
    sim = FakeScene(
        tcp=(0.08, 0.00, 0.03),
        objects={
            101: (0.081, 0.002, 0.02),
            102: (0.082, 0.001, 0.02),
        },
        pickables=[102],
    )
    tool = CoppeliaSimSuction(sim)

    evidence = tool.on()

    assert evidence.object_handle == 102
