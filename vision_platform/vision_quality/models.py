from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VisionProfile:
    profile_id: str
    label: str
    resolution: tuple[int, int]
    perspective_angle_deg: float | int
    camera_rig_z_m: float | int
    key_diffuse_rgb: tuple[float, float, float]
    fill_diffuse_rgb: tuple[float, float, float]


@dataclass(frozen=True)
class VisionProfileCatalog:
    baseline_profile_id: str
    sensor_path: str
    camera_rig_path: str
    key_light_path: str
    fill_light_path: str
    near_clip_m: float | int
    far_clip_m: float | int
    profiles: tuple[VisionProfile, ...]

    @property
    def profile_ids(self) -> tuple[str, ...]:
        return tuple(profile.profile_id for profile in self.profiles)

    def require(self, profile_id: str) -> VisionProfile:
        for profile in self.profiles:
            if profile.profile_id == profile_id:
                return profile
        raise KeyError(f"unknown vision profile: {profile_id}")


@dataclass(frozen=True)
class AppliedVisionProfile:
    profile_id: str
    resolution: tuple[int, int]
    perspective_angle_deg: float | int
    camera_rig_z_m: float | int
    key_diffuse_rgb: tuple[float, float, float]
    fill_diffuse_rgb: tuple[float, float, float]

    def to_public_dict(self) -> dict[str, object]:
        return {
            "profile_id": self.profile_id,
            "resolution": list(self.resolution),
            "perspective_angle_deg": self.perspective_angle_deg,
            "camera_rig_z_m": self.camera_rig_z_m,
            "key_diffuse_rgb": list(self.key_diffuse_rgb),
            "fill_diffuse_rgb": list(self.fill_diffuse_rgb),
        }
