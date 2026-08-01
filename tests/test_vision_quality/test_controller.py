from __future__ import annotations

from copy import deepcopy
import math
import threading
from typing import Callable

import pytest

from vision_platform.cameras.coppeliasim import CoppeliaSimCamera
from vision_platform.cameras.hikvision import HikvisionCamera
from vision_platform.cameras.replay import ReplayCamera
from vision_platform.vision_quality.controller import VisionProfileController
from vision_platform.vision_quality.models import AppliedVisionProfile, VisionProfile, VisionProfileCatalog


class FakeSim:
    visionintparam_resolution_x = 1002
    visionintparam_resolution_y = 1003
    visionfloatparam_perspective_angle = 1004
    visionfloatparam_near_clipping = 1000
    visionfloatparam_far_clipping = 1001
    handle_world = -1

    def __init__(self) -> None:
        self.handles = {
            "/VisionQualityLab/CameraRig/Camera": 1,
            "/VisionQualityLab/CameraRig": 2,
            "/VisionQualityLab/Lighting/KeyLight": 3,
            "/VisionQualityLab/Lighting/FillLight": 4,
        }
        self.int_params = {(1, 1002): 512, (1, 1003): 512}
        self.float_params = {
            (1, 1004): math.radians(60.0),
            (1, 1000): 0.05,
            (1, 1001): 2.0,
        }
        self.positions = {2: [0.35, -0.125, 0.70]}
        self.lights: dict[int, tuple[object, list[object], list[object]]] = {
            3: (3, [0.8, 0.8, 0.8], [0.10, 0.20, 0.30]),
            4: (2, [0.35, 0.35, 0.35], [0.40, 0.50, 0.60]),
        }
        self.light_reserved: dict[int, object] = {3: [0.0] * 3, 4: [0.0] * 3}
        self.light_return_override: dict[int, object] = {}
        self.light_set_api_calls: list[tuple[object, ...]] = []
        self.calls: list[tuple[object, ...]] = []
        self.setter_calls: list[tuple[object, ...]] = []
        self.setter_threads: list[int] = []
        self.fail_once_on: tuple[object, ...] | None = None
        self.fail_always_on: set[tuple[object, ...]] = set()
        self.mutate_then_raise_always_on: set[tuple[object, ...]] = set()
        self.fail_once_exception: BaseException = RuntimeError("simulated one-shot set failure")
        self.fail_always_exception: BaseException = RuntimeError("simulated permanent set failure")
        self.fail_read_on: tuple[object, ...] | None = None
        self.read_failure: BaseException = RuntimeError("simulated read failure")
        self.missing_path: str | None = None
        self.get_object_failure: BaseException | None = None
        self.corrupt_angle_after_setters: int | None = None
        self._angle_corrupted = False
        self.block_first_setter_entered: threading.Event | None = None
        self.release_first_setter: threading.Event | None = None

    def getObject(self, path: str) -> int:
        self.calls.append(("getObject", path))
        if self.get_object_failure is not None:
            raise self.get_object_failure
        if path == self.missing_path:
            raise KeyError(path)
        return self.handles[path]

    def getObjectInt32Param(self, handle: int, param: int) -> int:
        self._maybe_fail_read(("int", handle, param))
        return self.int_params[(handle, param)]

    def setObjectInt32Param(self, handle: int, param: int, value: int) -> None:
        key = ("int", handle, param)
        self._set(key, lambda: self.int_params.__setitem__((handle, param), int(value)))

    def getObjectFloatParam(self, handle: int, param: int) -> float:
        key = ("float", handle, param)
        self._maybe_fail_read(key)
        value = self.float_params[(handle, param)]
        if (
            param == self.visionfloatparam_perspective_angle
            and self.corrupt_angle_after_setters is not None
            and len(self.setter_calls) >= self.corrupt_angle_after_setters
            and not self._angle_corrupted
        ):
            self._angle_corrupted = True
            return value + math.radians(1.0)
        return value

    def setObjectFloatParam(self, handle: int, param: int, value: float) -> None:
        key = ("float", handle, param)
        self._set(key, lambda: self.float_params.__setitem__((handle, param), float(value)))

    def getObjectPosition(self, handle: int, relative: int) -> list[float]:
        assert relative == self.handle_world
        self._maybe_fail_read(("position", handle))
        return list(self.positions[handle])

    def setObjectPosition(self, handle: int, relative: int, value: list[float]) -> None:
        assert relative == self.handle_world
        key = ("position", handle)
        self._set(key, lambda: self.positions.__setitem__(handle, list(value)))

    def getLightParameters(self, handle: int) -> object:
        self._maybe_fail_read(("light", handle))
        if handle in self.light_return_override:
            return self.light_return_override[handle]
        state, diffuse, specular = self.lights[handle]
        return state, deepcopy(self.light_reserved[handle]), list(diffuse), list(specular)

    def setLightParameters(
        self,
        handle: int,
        state: int,
        reserved: None,
        diffuse: list[float],
        specular: list[float],
    ) -> None:
        key = ("light", handle)
        self.light_set_api_calls.append(
            (handle, state, reserved, tuple(diffuse), tuple(specular))
        )
        value = (state, list(diffuse), list(specular))
        self._set(key, lambda: self.lights.__setitem__(handle, value))

    def _set(self, key: tuple[object, ...], action) -> None:
        self.calls.append(key)
        self.setter_calls.append(key)
        self.setter_threads.append(threading.get_ident())
        if len(self.setter_calls) == 1 and self.block_first_setter_entered is not None:
            self.block_first_setter_entered.set()
            assert self.release_first_setter is not None
            assert self.release_first_setter.wait(timeout=2.0)
        if self.fail_once_on == key:
            self.fail_once_on = None
            raise self.fail_once_exception
        if key in self.mutate_then_raise_always_on:
            action()
            raise self.fail_always_exception
        if key in self.fail_always_on:
            raise self.fail_always_exception
        action()

    def _maybe_fail_read(self, key: tuple[object, ...]) -> None:
        if self.fail_read_on == key:
            raise self.read_failure


class FakeCamera(CoppeliaSimCamera):
    def __init__(
        self,
        sim: FakeSim,
        *,
        sensor_path: str = "/VisionQualityLab/CameraRig/Camera",
    ) -> None:
        super().__init__(sim=sim, sensor_path=sensor_path)
        self.read_count = 0
        self.timeouts: list[float] = []
        self.failure: BaseException | None = None
        self.on_read: Callable[[], None] | None = None

    def read(self, timeout_s: float = 1.0) -> object:
        self.read_count += 1
        self.timeouts.append(timeout_s)
        if self.on_read is not None:
            self.on_read()
        if self.failure is not None:
            raise self.failure
        return object()


def _catalog(*, ambiguous: bool = False) -> VisionProfileCatalog:
    profiles = (
        VisionProfile("standard", "标准", (512, 512), 60.0, 0.70, (0.8,) * 3, (0.35,) * 3),
        VisionProfile("wide_dim", "宽弱", (256, 256), 75.0, 0.80, (0.35,) * 3, (0.15,) * 3),
    )
    if ambiguous:
        profiles += (
            VisionProfile(
                "standard_twin",
                "标准副本",
                (512, 512),
                60.0,
                0.70,
                (0.8,) * 3,
                (0.35,) * 3,
            ),
        )
    return VisionProfileCatalog(
        "standard",
        "/VisionQualityLab/CameraRig/Camera",
        "/VisionQualityLab/CameraRig",
        "/VisionQualityLab/Lighting/KeyLight",
        "/VisionQualityLab/Lighting/FillLight",
        0.05,
        2.0,
        profiles,
    )


def _controller(
    sim: FakeSim | None = None,
    camera: object | None = None,
    *,
    catalog: VisionProfileCatalog | None = None,
    allowed: tuple[str, ...] = ("standard", "wide_dim"),
) -> VisionProfileController:
    selected_sim = sim if sim is not None else FakeSim()
    selected_camera = camera if camera is not None else FakeCamera(selected_sim)
    return VisionProfileController(
        sim=selected_sim,
        camera=selected_camera,
        catalog=catalog or _catalog(),
        allowed_profile_ids=allowed,
    )


def _state(sim: FakeSim) -> tuple[object, ...]:
    return (
        deepcopy(sim.int_params),
        deepcopy(sim.float_params),
        deepcopy(sim.positions),
        deepcopy(sim.lights),
    )


def _setter_order(sim: FakeSim) -> list[tuple[object, ...]]:
    return [
        ("int", 1, sim.visionintparam_resolution_x),
        ("int", 1, sim.visionintparam_resolution_y),
        ("float", 1, sim.visionfloatparam_perspective_angle),
        ("float", 1, sim.visionfloatparam_near_clipping),
        ("float", 1, sim.visionfloatparam_far_clipping),
        ("position", 2),
        ("light", 3),
        ("light", 4),
    ]


def test_apply_sets_all_fields_reads_back_and_discards_exactly_first_frame() -> None:
    sim = FakeSim()
    camera = FakeCamera(sim)
    controller = _controller(sim, camera)

    state = controller.apply("wide_dim")

    assert isinstance(state, AppliedVisionProfile)
    assert state.to_public_dict() == {
        "profile_id": "wide_dim",
        "resolution": [256, 256],
        "perspective_angle_deg": 75.0,
        "camera_rig_z_m": 0.80,
        "key_diffuse_rgb": [0.35, 0.35, 0.35],
        "fill_diffuse_rgb": [0.15, 0.15, 0.15],
    }
    assert "label" not in state.to_public_dict()
    assert sim.int_params == {(1, 1002): 256, (1, 1003): 256}
    assert sim.float_params == {
        (1, 1004): pytest.approx(math.radians(75.0)),
        (1, 1000): pytest.approx(0.05),
        (1, 1001): pytest.approx(2.0),
    }
    assert sim.positions[2] == pytest.approx([0.35, -0.125, 0.80])
    assert sim.lights[3] == (3, [0.35] * 3, [0.10, 0.20, 0.30])
    assert sim.lights[4] == (2, [0.15] * 3, [0.40, 0.50, 0.60])
    assert sim.setter_calls == _setter_order(sim)
    assert camera.read_count == 1
    assert camera.timeouts == [2.0]


def test_light_calls_match_real_coppeliasim_four_return_five_argument_contract() -> None:
    sim = FakeSim()
    controller = _controller(sim)

    controller.apply("wide_dim")

    assert sim.light_set_api_calls == [
        (3, 3, None, (0.35, 0.35, 0.35), (0.10, 0.20, 0.30)),
        (4, 2, None, (0.15, 0.15, 0.15), (0.40, 0.50, 0.60)),
    ]


@pytest.mark.parametrize(
    ("profile_id", "message"),
    [
        (None, "VISION_PROFILE_ID_INVALID"),
        (123, "VISION_PROFILE_ID_INVALID"),
        ("../wide_dim", "VISION_PROFILE_ID_INVALID"),
        ("Wide-Dim", "VISION_PROFILE_ID_INVALID"),
        ("missing", "VISION_PROFILE_ID_INVALID"),
    ],
)
def test_invalid_or_unknown_profile_never_calls_setter(profile_id: object, message: str) -> None:
    sim = FakeSim()
    controller = _controller(sim)

    with pytest.raises(ValueError, match=message):
        controller.apply(profile_id)  # type: ignore[arg-type]

    assert sim.setter_calls == []


def test_published_but_unallowed_profile_never_calls_setter() -> None:
    sim = FakeSim()
    controller = _controller(sim, allowed=("standard",))

    with pytest.raises(ValueError, match="VISION_PROFILE_NOT_ALLOWED"):
        controller.apply("wide_dim")

    assert sim.setter_calls == []


@pytest.mark.parametrize("failure_index", range(8))
def test_each_target_setter_failure_restores_every_field_in_reverse_order(
    failure_index: int,
) -> None:
    sim = FakeSim()
    before = _state(sim)
    order = _setter_order(sim)
    sim.fail_once_on = order[failure_index]
    controller = _controller(sim)

    with pytest.raises(RuntimeError, match="VISION_PROFILE_APPLY_FAILED") as raised:
        controller.apply("wide_dim")

    assert str(raised.value.__cause__) == "simulated one-shot set failure"
    assert _state(sim) == before
    assert sim.setter_calls[: failure_index + 1] == order[: failure_index + 1]
    assert sim.setter_calls[failure_index + 1 :] == list(reversed(order))


def test_readback_mismatch_rolls_back_complete_snapshot() -> None:
    sim = FakeSim()
    before = _state(sim)
    sim.corrupt_angle_after_setters = 8
    controller = _controller(sim)

    with pytest.raises(RuntimeError, match="VISION_PROFILE_APPLY_FAILED") as raised:
        controller.apply("wide_dim")

    assert str(raised.value.__cause__) == "VISION_PROFILE_READBACK_MISMATCH"
    assert _state(sim) == before


@pytest.mark.parametrize("mutate_then_raise", [False, True])
def test_persistent_rollback_failure_still_attempts_all_later_restores(
    mutate_then_raise: bool,
) -> None:
    sim = FakeSim()
    before = _state(sim)
    failure_key = ("light", 3)
    if mutate_then_raise:
        sim.mutate_then_raise_always_on.add(failure_key)
    else:
        sim.fail_always_on.add(failure_key)
    controller = _controller(sim)

    with pytest.raises(RuntimeError, match="VISION_PROFILE_ROLLBACK_FAILED") as raised:
        controller.apply("wide_dim")

    assert _state(sim) == before
    assert sim.setter_calls[-8:] == list(reversed(_setter_order(sim)))
    assert isinstance(raised.value.__cause__, BaseExceptionGroup)
    assert any(
        "simulated permanent set failure" in str(error)
        for error in raised.value.__cause__.exceptions
    )
    assert any(
        "original apply failure" in note
        for note in getattr(raised.value, "__notes__", ())
    )


def test_multiple_rollback_failures_are_collected_before_readback_verification() -> None:
    sim = FakeSim()
    camera = FakeCamera(sim)
    before = _state(sim)

    def arm_rollback_failures() -> None:
        sim.fail_always_on.update(
            {
                ("light", 4),
                ("float", 1, sim.visionfloatparam_perspective_angle),
            }
        )

    camera.on_read = arm_rollback_failures
    camera.failure = RuntimeError("camera read failed")
    controller = _controller(sim, camera)

    with pytest.raises(RuntimeError, match="VISION_PROFILE_ROLLBACK_FAILED") as raised:
        controller.apply("wide_dim")

    assert sim.setter_calls[-8:] == list(reversed(_setter_order(sim)))
    assert _state(sim) != before
    assert isinstance(raised.value.__cause__, BaseExceptionGroup)
    evidence = raised.value.__cause__.exceptions
    assert len(evidence) >= 3
    assert sum("simulated permanent set failure" in str(error) for error in evidence) >= 2
    assert any("VISION_PROFILE_READBACK_MISMATCH" in str(error) for error in evidence)
    assert any("camera read failed" in note for note in raised.value.__notes__)


@pytest.mark.parametrize(
    "missing_path",
    [
        "/VisionQualityLab/CameraRig/Camera",
        "/VisionQualityLab/CameraRig",
        "/VisionQualityLab/Lighting/KeyLight",
        "/VisionQualityLab/Lighting/FillLight",
    ],
)
def test_missing_each_fixed_object_is_stable_and_never_sets(missing_path: str) -> None:
    sim = FakeSim()
    sim.missing_path = missing_path

    with pytest.raises(RuntimeError, match="VISION_PROFILE_OBJECT_MISSING") as raised:
        _controller(sim)

    assert isinstance(raised.value.__cause__, KeyError)
    assert sim.setter_calls == []


@pytest.mark.parametrize("camera", [object(), type("NonCallableCamera", (), {"read": None})()])
def test_camera_without_callable_read_is_rejected(camera: object) -> None:
    with pytest.raises(RuntimeError, match="VISION_PROFILE_BACKEND_UNAVAILABLE"):
        _controller(camera=camera)


@pytest.mark.parametrize(
    "allowed",
    [
        (),
        ("standard", "standard"),
        ("missing",),
        ("../standard",),
        (1,),
        ["standard"],
    ],
)
def test_constructor_rejects_invalid_allowed_profile_ids(allowed: object) -> None:
    with pytest.raises(ValueError, match="VISION_PROFILE"):
        _controller(allowed=allowed)  # type: ignore[arg-type]


def test_constructor_requires_baseline_profile_to_be_allowed_for_reset() -> None:
    with pytest.raises(ValueError, match="VISION_PROFILE_NOT_ALLOWED"):
        _controller(allowed=("wide_dim",))


def test_constructor_accepts_only_vision_profile_catalog() -> None:
    with pytest.raises(TypeError, match="VISION_PROFILE"):
        _controller(catalog=object())  # type: ignore[arg-type]


def test_constructor_accepts_unopened_coppeliasim_camera_with_same_injected_sim() -> None:
    sim = FakeSim()
    camera = FakeCamera(sim)

    controller = _controller(sim, camera)

    assert not camera.is_open
    assert controller.current().profile_id == "standard"


def test_constructor_accepts_open_coppeliasim_camera_bound_to_same_sim() -> None:
    sim = FakeSim()
    camera = FakeCamera(FakeSim())
    camera._sim = sim
    camera._sensor_handle = 1

    assert _controller(sim, camera).current().profile_id == "standard"


@pytest.mark.parametrize(
    "camera_factory",
    [
        lambda sim: ReplayCamera([]),
        lambda sim: HikvisionCamera(),
        lambda sim: FakeCamera(FakeSim()),
        lambda sim: FakeCamera(sim, sensor_path="/VisionLab/Camera"),
    ],
    ids=("replay", "hikvision", "different-sim", "wrong-sensor"),
)
def test_constructor_rejects_nonmatching_camera_backend(camera_factory) -> None:
    sim = FakeSim()

    with pytest.raises(RuntimeError, match="VISION_PROFILE_BACKEND_UNAVAILABLE"):
        _controller(sim, camera_factory(sim))


def test_constructor_rejects_coppeliasim_camera_without_callable_read() -> None:
    sim = FakeSim()
    camera = FakeCamera(sim)
    camera.read = None  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="VISION_PROFILE_BACKEND_UNAVAILABLE"):
        _controller(sim, camera)


def test_controller_documents_runner_owned_deadline_and_quarantine_precondition() -> None:
    documentation = VisionProfileController.__doc__ or ""

    assert "runner" in documentation.lower()
    assert "deadline" in documentation.lower()
    assert "quarantine" in documentation.lower()


def test_current_returns_only_unique_exact_published_profile_without_reading_camera() -> None:
    sim = FakeSim()
    camera = FakeCamera(sim)
    controller = _controller(sim, camera)

    current = controller.current()

    assert current == AppliedVisionProfile(
        "standard", (512, 512), 60.0, 0.70, (0.8,) * 3, (0.35,) * 3
    )
    assert camera.read_count == 0


def test_current_rejects_unknown_combination_instead_of_nearest_profile() -> None:
    sim = FakeSim()
    sim.float_params[(1, sim.visionfloatparam_perspective_angle)] = math.radians(67.5)
    controller = _controller(sim)

    with pytest.raises(RuntimeError, match="VISION_PROFILE_READBACK_MISMATCH"):
        controller.current()


def test_constructor_rejects_exact_twin_published_profiles() -> None:
    catalog = _catalog(ambiguous=True)

    with pytest.raises(ValueError, match="VISION_PROFILE_CATALOG_AMBIGUOUS"):
        _controller(catalog=catalog, allowed=catalog.profile_ids)


def test_constructor_rejects_profiles_with_overlapping_observable_tolerances() -> None:
    standard = _catalog().require("standard")
    overlapping = VisionProfile(
        "standard_near",
        "近重叠",
        standard.resolution,
        60.099,
        0.70099,
        (0.80199,) * 3,
        (0.35199,) * 3,
    )
    base = _catalog()
    catalog = VisionProfileCatalog(
        base.baseline_profile_id,
        base.sensor_path,
        base.camera_rig_path,
        base.key_light_path,
        base.fill_light_path,
        base.near_clip_m,
        base.far_clip_m,
        (standard, overlapping),
    )

    with pytest.raises(ValueError, match="VISION_PROFILE_CATALOG_AMBIGUOUS"):
        _controller(catalog=catalog, allowed=catalog.profile_ids)


def _angle(sim: FakeSim, delta: float) -> None:
    sim.float_params[(1, sim.visionfloatparam_perspective_angle)] = math.radians(60.0 + delta)


def _position(sim: FakeSim, delta: float) -> None:
    sim.positions[2][2] = 0.70 + delta


def _light(sim: FakeSim, handle: int, delta: float) -> None:
    enabled, diffuse, specular = sim.lights[handle]
    diffuse[1] += delta
    sim.lights[handle] = enabled, diffuse, specular


def _clip(sim: FakeSim, param: int, delta: float) -> None:
    sim.float_params[(1, param)] += delta


def _resolution(sim: FakeSim, delta: int) -> None:
    sim.int_params[(1, sim.visionintparam_resolution_x)] += delta


@pytest.mark.parametrize(
    "mutate",
    [
        lambda sim: _angle(sim, 0.05),
        lambda sim: _angle(sim, -0.05),
        lambda sim: _position(sim, 0.0005),
        lambda sim: _position(sim, -0.0005),
        lambda sim: _light(sim, 3, 0.001),
        lambda sim: _light(sim, 3, -0.001),
        lambda sim: _light(sim, 4, 0.001),
        lambda sim: _light(sim, 4, -0.001),
        lambda sim: _clip(sim, sim.visionfloatparam_near_clipping, 0.0005),
        lambda sim: _clip(sim, sim.visionfloatparam_near_clipping, -0.0005),
        lambda sim: _clip(sim, sim.visionfloatparam_far_clipping, 0.0005),
        lambda sim: _clip(sim, sim.visionfloatparam_far_clipping, -0.0005),
        lambda sim: _resolution(sim, 0),
    ],
    ids=(
        "angle-positive",
        "angle-negative",
        "position-positive",
        "position-negative",
        "key-light-positive",
        "key-light-negative",
        "fill-light-positive",
        "fill-light-negative",
        "near-clip-positive",
        "near-clip-negative",
        "far-clip-positive",
        "far-clip-negative",
        "resolution",
    ),
)
def test_current_accepts_exact_readback_tolerance_edges(mutate) -> None:
    sim = FakeSim()
    mutate(sim)

    assert _controller(sim).current().profile_id == "standard"


@pytest.mark.parametrize(
    "mutate",
    [
        lambda sim: _angle(sim, 0.050001),
        lambda sim: _position(sim, 0.000501),
        lambda sim: _light(sim, 3, 0.001001),
        lambda sim: _light(sim, 4, -0.001001),
        lambda sim: _clip(sim, sim.visionfloatparam_near_clipping, 0.000501),
        lambda sim: _clip(sim, sim.visionfloatparam_far_clipping, -0.000501),
        lambda sim: _resolution(sim, 1),
    ],
    ids=("angle", "position", "key-light", "fill-light", "near-clip", "far-clip", "resolution"),
)
def test_current_rejects_values_beyond_readback_tolerances(mutate) -> None:
    sim = FakeSim()
    mutate(sim)

    with pytest.raises(RuntimeError, match="VISION_PROFILE_READBACK_MISMATCH"):
        _controller(sim).current()


def _set_resolution_component(sim: FakeSim, value: object) -> None:
    sim.int_params[(1, sim.visionintparam_resolution_x)] = value  # type: ignore[assignment]


def _set_float_parameter(sim: FakeSim, param: int, value: object) -> None:
    sim.float_params[(1, param)] = value  # type: ignore[assignment]


def _set_position_component(sim: FakeSim, value: object) -> None:
    sim.positions[2][0] = value  # type: ignore[list-item]


def _set_light_state(sim: FakeSim, value: object) -> None:
    _, diffuse, specular = sim.lights[3]
    sim.lights[3] = value, diffuse, specular


def _set_light_channel(sim: FakeSim, part: str, value: object) -> None:
    state, diffuse, specular = sim.lights[3]
    selected = diffuse if part == "diffuse" else specular
    selected[0] = value
    sim.lights[3] = state, diffuse, specular


@pytest.mark.parametrize(
    "mutate",
    [
        lambda sim: _set_resolution_component(sim, "512"),
        lambda sim: _set_resolution_component(sim, True),
        lambda sim: _set_resolution_component(sim, 0),
        lambda sim: _set_float_parameter(sim, sim.visionfloatparam_perspective_angle, "1.0"),
        lambda sim: _set_float_parameter(sim, sim.visionfloatparam_near_clipping, False),
        lambda sim: _set_float_parameter(sim, sim.visionfloatparam_far_clipping, float("inf")),
        lambda sim: _set_position_component(sim, "0.35"),
        lambda sim: _set_position_component(sim, True),
        lambda sim: _set_position_component(sim, float("nan")),
        lambda sim: _set_light_state(sim, True),
        lambda sim: _set_light_state(sim, -1),
        lambda sim: _set_light_state(sim, 1.0),
        lambda sim: _set_light_channel(sim, "diffuse", "0.8"),
        lambda sim: _set_light_channel(sim, "diffuse", False),
        lambda sim: _set_light_channel(sim, "specular", float("nan")),
        lambda sim: sim.light_return_override.__setitem__(3, (3, [0.0] * 3, [0.8] * 3)),
    ],
    ids=(
        "resolution-string",
        "resolution-bool",
        "resolution-zero",
        "fov-string",
        "clip-bool",
        "clip-nonfinite",
        "position-string",
        "position-bool",
        "position-nonfinite",
        "light-state-bool",
        "light-state-negative",
        "light-state-float",
        "light-diffuse-string",
        "light-diffuse-bool",
        "light-specular-nonfinite",
        "light-return-arity",
    ),
)
def test_current_rejects_non_native_or_invalid_snapshot_values(mutate) -> None:
    sim = FakeSim()
    mutate(sim)

    with pytest.raises(RuntimeError, match="VISION_PROFILE_READBACK_MISMATCH"):
        _controller(sim).current()


@pytest.mark.parametrize(
    "reserved",
    [
        None,
        "opaque-reserved-value",
        [1.0, "ignored", float("nan")],
        {"ignored": True},
    ],
    ids=("none", "scalar", "arbitrary-vector", "mapping"),
)
def test_current_ignores_reserved_light_return_content(reserved: object) -> None:
    sim = FakeSim()
    sim.light_reserved[3] = reserved

    assert _controller(sim).current().profile_id == "standard"


def test_reset_applies_baseline_and_is_idempotent() -> None:
    sim = FakeSim()
    camera = FakeCamera(sim)
    controller = _controller(sim, camera)
    controller.apply("wide_dim")

    first = controller.reset()
    second = controller.reset()

    assert first.profile_id == second.profile_id == "standard"
    assert controller.current().profile_id == "standard"
    assert camera.timeouts == [2.0, 2.0, 2.0]


def test_reset_wraps_apply_failure_with_stable_error_and_cause() -> None:
    sim = FakeSim()
    camera = FakeCamera(sim)
    camera.failure = RuntimeError("camera unavailable")
    controller = _controller(sim, camera)

    with pytest.raises(RuntimeError, match="VISION_PROFILE_RESET_FAILED") as raised:
        controller.reset()

    assert str(raised.value.__cause__) == "VISION_PROFILE_APPLY_FAILED"


def test_apply_calls_are_serialized_without_interleaved_setters() -> None:
    sim = FakeSim()
    sim.block_first_setter_entered = threading.Event()
    sim.release_first_setter = threading.Event()
    controller = _controller(sim)
    errors: list[BaseException] = []
    second_started = threading.Event()

    def apply(profile_id: str, started: threading.Event | None = None) -> None:
        if started is not None:
            started.set()
        try:
            controller.apply(profile_id)
        except BaseException as error:  # pragma: no cover - asserted through errors
            errors.append(error)

    first = threading.Thread(target=apply, args=("wide_dim",))
    first.start()
    assert sim.block_first_setter_entered.wait(timeout=2.0)
    first_thread_id = sim.setter_threads[0]

    second = threading.Thread(target=apply, args=("standard", second_started))
    second.start()
    assert second_started.wait(timeout=2.0)
    second.join(timeout=0.05)

    assert second.is_alive()
    assert sim.setter_threads == [first_thread_id]
    sim.release_first_setter.set()
    first.join(timeout=2.0)
    second.join(timeout=2.0)

    assert not errors
    assert not first.is_alive()
    assert not second.is_alive()
    assert len(sim.setter_threads) == 16
    assert len(set(sim.setter_threads[:8])) == 1
    assert len(set(sim.setter_threads[8:])) == 1
    assert sim.setter_threads[0] != sim.setter_threads[8]
    assert sim.setter_calls == _setter_order(sim) * 2


def test_current_converts_backend_read_failure_to_stable_public_error() -> None:
    sim = FakeSim()
    sim.fail_read_on = ("float", 1, sim.visionfloatparam_near_clipping)
    controller = _controller(sim)

    with pytest.raises(RuntimeError, match="VISION_PROFILE_BACKEND_UNAVAILABLE") as raised:
        controller.current()

    assert str(raised.value.__cause__) == "simulated read failure"


@pytest.mark.parametrize("exception_type", [KeyboardInterrupt, SystemExit])
def test_init_does_not_rewrite_control_flow_exceptions(exception_type) -> None:
    sim = FakeSim()
    control = exception_type("stop during object resolution")
    sim.get_object_failure = control

    with pytest.raises(exception_type) as raised:
        _controller(sim)

    assert raised.value is control


@pytest.mark.parametrize("exception_type", [KeyboardInterrupt, SystemExit])
def test_current_does_not_rewrite_control_flow_exceptions(exception_type) -> None:
    sim = FakeSim()
    control = exception_type("stop during readback")
    sim.fail_read_on = ("float", 1, sim.visionfloatparam_near_clipping)
    sim.read_failure = control
    controller = _controller(sim)

    with pytest.raises(exception_type) as raised:
        controller.current()

    assert raised.value is control


@pytest.mark.parametrize("exception_type", [KeyboardInterrupt, SystemExit])
def test_apply_rolls_back_then_reraises_original_control_flow_exception(exception_type) -> None:
    sim = FakeSim()
    camera = FakeCamera(sim)
    before = _state(sim)
    control = exception_type("stop during discarded frame")
    camera.failure = control
    controller = _controller(sim, camera)

    with pytest.raises(exception_type) as raised:
        controller.apply("wide_dim")

    assert raised.value is control
    assert _state(sim) == before
    assert sim.setter_calls[-8:] == list(reversed(_setter_order(sim)))


def test_apply_reraises_control_flow_even_when_best_effort_rollback_fails() -> None:
    sim = FakeSim()
    camera = FakeCamera(sim)
    control = KeyboardInterrupt("stop with rollback failure")

    def arm_rollback_failure() -> None:
        sim.fail_always_on.add(("light", 4))

    camera.on_read = arm_rollback_failure
    camera.failure = control
    controller = _controller(sim, camera)

    with pytest.raises(KeyboardInterrupt) as raised:
        controller.apply("wide_dim")

    assert raised.value is control
    assert sim.setter_calls[-8:] == list(reversed(_setter_order(sim)))
    assert isinstance(raised.value.__cause__, BaseExceptionGroup)
    assert any("rollback" in note.lower() for note in raised.value.__notes__)


@pytest.mark.parametrize("exception_type", [KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize("rollback_source", ["setter", "readback"])
def test_ordinary_apply_failure_reraises_first_rollback_control_flow_after_full_cleanup(
    exception_type,
    rollback_source: str,
) -> None:
    sim = FakeSim()
    camera = FakeCamera(sim)
    original = RuntimeError("ordinary camera apply failure")
    control = exception_type(f"rollback {rollback_source} interrupted")

    def arm_rollback_failure() -> None:
        if rollback_source == "setter":
            sim.fail_always_on.add(("light", 4))
            sim.fail_always_exception = control
        else:
            sim.fail_read_on = ("int", 1, sim.visionintparam_resolution_x)
            sim.read_failure = control

    camera.on_read = arm_rollback_failure
    camera.failure = original
    controller = _controller(sim, camera)

    with pytest.raises(exception_type) as raised:
        controller.apply("wide_dim")

    assert raised.value is control
    assert sim.setter_calls[-8:] == list(reversed(_setter_order(sim)))
    assert isinstance(raised.value.__cause__, BaseExceptionGroup)
    assert original in raised.value.__cause__.exceptions
    assert any("original apply failure" in note for note in raised.value.__notes__)
    if rollback_source == "setter":
        assert any(
            "VISION_PROFILE_READBACK_MISMATCH" in str(error)
            for error in raised.value.__cause__.exceptions
        )


def test_reset_does_not_rewrite_control_flow_exception() -> None:
    sim = FakeSim()
    camera = FakeCamera(sim)
    control = SystemExit("stop during reset")
    camera.failure = control
    controller = _controller(sim, camera)

    with pytest.raises(SystemExit) as raised:
        controller.reset()

    assert raised.value is control
