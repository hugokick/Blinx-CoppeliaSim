from itertools import pairwise
from math import dist

import pytest

from vision_platform.errors import MotionSafetyError
from vision_platform.robot.safety import WorkspacePolicy, plan_gate_path


def _policy(max_step_mm=250):
    return WorkspacePolicy(
        x_mm=(20, 140),
        y_mm=(-90, 90),
        z_mm=(10, 140),
        safe_z_mm=100,
        max_step_mm=max_step_mm,
    )


def test_gate_path_never_moves_horizontally_below_safe_z():
    policy = _policy()

    path = plan_gate_path(
        pick=(70, -30, 20),
        drop=(110, 50, 20),
        policy=policy,
    )

    for previous, current in pairwise(path):
        horizontal = previous[:2] != current[:2]
        if horizontal:
            assert previous[2] >= 100
            assert current[2] >= 100


def test_gate_path_visits_pick_and_drop_and_finishes_at_safe_height():
    policy = _policy()

    path = plan_gate_path(
        pick=(70, -30, 20),
        drop=(110, 50, 20),
        policy=policy,
    )

    assert (70.0, -30.0, 20.0) in path
    assert (110.0, 50.0, 20.0) in path
    assert path[-1] == (110.0, 50.0, 100.0)


def test_gate_path_subdivides_every_segment_to_maximum_step():
    policy = _policy(max_step_mm=25)

    path = plan_gate_path(
        pick=(70, -30, 20),
        drop=(130, 70, 20),
        policy=policy,
    )

    assert all(dist(left, right) <= 25.000001 for left, right in pairwise(path))


@pytest.mark.parametrize(
    "point",
    [
        (19, 0, 20),
        (80, -91, 20),
        (80, 0, 141),
    ],
)
def test_workspace_policy_rejects_each_out_of_range_axis(point):
    with pytest.raises(MotionSafetyError) as exc:
        _policy().validate(point)

    assert exc.value.code == "TARGET_OUT_OF_WORKSPACE"
