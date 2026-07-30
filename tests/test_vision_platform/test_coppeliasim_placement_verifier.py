from vision_platform.tasks.coppeliasim_verifier import (
    CoppeliaSimPlacementVerifier,
)


class FakeScene:
    objfloatparam_objbbox_min_x = 1
    objfloatparam_objbbox_max_x = 2
    objfloatparam_objbbox_min_y = 3
    objfloatparam_objbbox_max_y = 4
    objfloatparam_objbbox_min_z = 5
    objfloatparam_objbbox_max_z = 6

    def __init__(self, *, relative_position):
        self.handles = {
            "/VisionLab/Zones/red": 201,
            "/VisionLab/Zones/blue": 202,
        }
        self.relative_position = list(relative_position)
        self.get_object_calls = []
        self.bounds = {
            1: -0.05,
            2: 0.05,
            3: -0.04,
            4: 0.04,
            5: -0.01,
            6: 0.10,
        }

    def getObject(self, path):
        self.get_object_calls.append(path)
        return self.handles[path]

    def getObjectPosition(self, object_handle, relative_to):
        assert object_handle == 101
        assert relative_to in self.handles.values()
        return list(self.relative_position)

    def getObjectFloatParam(self, handle, parameter):
        assert handle in self.handles.values()
        return self.bounds[parameter]


def test_verifier_passes_only_when_object_center_is_inside_expected_zone():
    sim = FakeScene(relative_position=(0.01, -0.02, 0.03))
    verifier = CoppeliaSimPlacementVerifier(
        sim=sim,
        zone_paths={"red": "/VisionLab/Zones/red"},
    )

    evidence = verifier.verify(101, expected_zone="red")

    assert evidence.ok is True
    assert evidence.expected_zone == "red"
    assert evidence.details["relative_position_m"] == [0.01, -0.02, 0.03]


def test_verifier_fails_when_center_is_outside_zone_xy_bounds():
    sim = FakeScene(relative_position=(0.051, 0.0, 0.03))
    verifier = CoppeliaSimPlacementVerifier(
        sim=sim,
        zone_paths={"red": "/VisionLab/Zones/red"},
    )

    evidence = verifier.verify(101, expected_zone="red")

    assert evidence.ok is False
    assert evidence.details["inside_x"] is False
    assert evidence.details["inside_y"] is True


def test_zone_handle_and_bounds_are_cached():
    sim = FakeScene(relative_position=(0.0, 0.0, 0.03))
    verifier = CoppeliaSimPlacementVerifier(
        sim=sim,
        zone_paths={"red": "/VisionLab/Zones/red"},
    )

    verifier.verify(101, expected_zone="red")
    verifier.verify(101, expected_zone="red")

    assert sim.get_object_calls == ["/VisionLab/Zones/red"]


def test_optional_z_check_can_reject_object_above_zone_volume():
    sim = FakeScene(relative_position=(0.0, 0.0, 0.11))
    verifier = CoppeliaSimPlacementVerifier(
        sim=sim,
        zone_paths={"red": "/VisionLab/Zones/red"},
        check_z=True,
    )

    evidence = verifier.verify(101, expected_zone="red")

    assert evidence.ok is False
    assert evidence.details["inside_z"] is False
