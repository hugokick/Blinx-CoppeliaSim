from __future__ import annotations

import numpy as np
import pytest

from tests.test_rgbd.synthetic_factory import (
    make_box,
    make_hole,
    make_plane,
    make_seeded_noise,
    make_step,
    make_tilted_plane,
)


def test_seeded_factory_is_deterministic_and_independent() -> None:
    first_image, first_depth = make_seeded_noise(32, 24, seed=1707)
    second_image, second_depth = make_seeded_noise(32, 24, seed=1707)
    assert np.array_equal(first_image, second_image)
    assert np.array_equal(first_depth, second_depth)
    first_depth[0, 0] = 9.0
    assert second_depth[0, 0] != 9.0


def test_all_factories_have_exact_contracts() -> None:
    for factory in (make_plane, make_step, make_box, make_tilted_plane, make_hole):
        image, depth = factory(12, 10)
        assert image.shape == (10, 12, 3)
        assert image.dtype == np.uint8
        assert depth.shape == (10, 12)
        assert depth.dtype == np.float32


def test_depth_fixture_values_are_analytic() -> None:
    _image, plane = make_plane(8, 6, 0.9)
    assert np.all(plane == np.float32(0.9))

    _image, step = make_step(8, 6, left_m=0.8, right_m=1.2)
    assert np.all(step[:, :4] == np.float32(0.8))
    assert np.all(step[:, 4:] == np.float32(1.2))

    _image, box = make_box(8, 8, plane_m=1.0, box_m=0.7)
    assert box[2, 2] == np.float32(0.7)
    assert box[0, 0] == np.float32(1.0)

    _image, tilted = make_tilted_plane(8, 6, base_m=0.6, du_m=0.001, dv_m=0.002)
    assert tilted[5, 7] == pytest.approx(0.6 + 0.001 * 7 + 0.002 * 5, abs=1e-7)

    _image, hole = make_hole(9, 9)
    assert hole[4, 4] == np.float32(0.0)
    assert hole[0, 0] == np.float32(1.0)
