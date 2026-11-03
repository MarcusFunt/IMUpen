"""Measured OnHW sensor groups and deterministic profile resolution."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations


@dataclass(frozen=True)
class SensorGroup:
    code: str
    name: str
    channel_names: tuple[str, ...]
    channel_indices: tuple[int, ...]
    units: tuple[str, ...]


@dataclass(frozen=True)
class SensorProfile:
    profile_id: str
    group_code: str
    groups: tuple[SensorGroup, ...]
    channel_names: tuple[str, ...]
    channel_indices: tuple[int, ...]
    units: tuple[str, ...]
    physical_description: str
    preprocessing: tuple[tuple[str, str], ...] = (
        ("resampling", "none"),
        ("normalization", "train_partition_zscore"),
    )
    dataset_compatibility: str = "onhw_words500_13_channel"

    @property
    def input_channels(self) -> int:
        return len(self.channel_indices)


SENSOR_GROUPS = (
    SensorGroup("F", "acc_front", ("acc_front_x", "acc_front_y", "acc_front_z"),
                (0, 1, 2), ("m/s^2",) * 3),
    SensorGroup("R", "acc_rear", ("acc_rear_x", "acc_rear_y", "acc_rear_z"),
                (3, 4, 5), ("m/s^2",) * 3),
    SensorGroup("G", "gyro", ("gyro_x", "gyro_y", "gyro_z"),
                (6, 7, 8), ("deg/s",) * 3),
    SensorGroup("M", "mag", ("mag_x", "mag_y", "mag_z"),
                (9, 10, 11), ("mT",) * 3),
    SensorGroup("P", "force", ("tip_force",), (12,), ("N",)),
)
ONHW_CHANNEL_NAMES = tuple(name for group in SENSOR_GROUPS for name in group.channel_names)
ONHW_CHANNEL_UNITS = tuple(unit for group in SENSOR_GROUPS for unit in group.units)
ONHW_SAMPLE_RATE_HZ = 100.0
PRIMARY_SENSOR_COMPARISONS = (
    {"first": "FG", "second": "RG", "question": "front versus rear accelerometer with shared gyro"},
    {"first": "FG", "second": "FRG", "question": "contribution of the second accelerometer"},
    {"first": "FG", "second": "FGM", "question": "magnetometer contribution with front accelerometer"},
    {"first": "FG", "second": "FGP", "question": "force contribution with front accelerometer"},
    {"first": "FRG", "second": "FRGM", "question": "magnetometer contribution with both accelerometers"},
    {"first": "FRG", "second": "FRGP", "question": "force contribution with both accelerometers"},
)


def _profile_id(groups: tuple[SensorGroup, ...]) -> str:
    code = "".join(group.code for group in groups)
    aliases = {
        "F": "front_acc3",
        "R": "rear_acc3",
        "G": "gyro3",
        "M": "mag3",
        "P": "force1",
        "FG": "front_acc_gyro6",
        "RG": "rear_acc_shared_gyro6",
        "FR": "dual_acc6",
        "FRG": "dual_acc_shared_gyro9",
        "FGM": "front_acc_gyro_mag9",
        "RGM": "rear_acc_shared_gyro_mag9",
        "FRGMP": "full13",
    }
    if code in aliases:
        return aliases[code]
    components = []
    for group in groups:
        if group.code == "F":
            components.append("front_acc")
        elif group.code == "R":
            components.append("rear_acc")
        elif group.code == "G":
            components.append("shared_gyro")
        elif group.code == "M":
            components.append("mag")
        else:
            components.append("force")
    count = sum(len(group.channel_indices) for group in groups)
    return f"{'_'.join(components)}{count}"


def _physical_description(groups: tuple[SensorGroup, ...]) -> str:
    codes = {group.code for group in groups}
    names = "+".join(group.name for group in groups)
    if "R" in codes and "G" in codes:
        return f"{names}; G is OnHW's single shared gyroscope, not a rear-mounted gyro"
    if "F" in codes and "G" in codes:
        return f"{names}; front accelerometer and gyroscope are the associated IMU channels"
    return f"Measured OnHW channel subset: {names}"


def _build_profiles() -> tuple[SensorProfile, ...]:
    profiles: list[SensorProfile] = []
    for size in range(1, len(SENSOR_GROUPS) + 1):
        for groups in combinations(SENSOR_GROUPS, size):
            profiles.append(
                SensorProfile(
                    profile_id=_profile_id(groups),
                    group_code="".join(group.code for group in groups),
                    groups=groups,
                    channel_names=tuple(name for group in groups for name in group.channel_names),
                    channel_indices=tuple(index for group in groups for index in group.channel_indices),
                    units=tuple(unit for group in groups for unit in group.units),
                    physical_description=_physical_description(groups),
                )
            )
    return tuple(profiles)


SENSOR_PROFILES = _build_profiles()
SENSOR_PROFILE_BY_ID = {profile.profile_id: profile for profile in SENSOR_PROFILES}
SENSOR_PROFILE_BY_CODE = {profile.group_code: profile for profile in SENSOR_PROFILES}
if len(SENSOR_PROFILE_BY_ID) != len(SENSOR_PROFILES):
    raise RuntimeError("sensor profile IDs must be unique")
if len(SENSOR_PROFILE_BY_CODE) != len(SENSOR_PROFILES):
    raise RuntimeError("sensor group codes must be unique")


def resolve_sensor_profile(profile: str | SensorProfile) -> SensorProfile:
    """Resolve either a descriptive profile ID or compact group code."""
    if isinstance(profile, SensorProfile):
        if profile.group_code not in SENSOR_PROFILE_BY_CODE:
            raise ValueError("sensor profile is not part of the measured OnHW registry")
        registered = SENSOR_PROFILE_BY_CODE[profile.group_code]
        if profile != registered:
            raise ValueError("sensor profile metadata differs from the registered definition")
        return registered
    key = str(profile).strip()
    resolved = SENSOR_PROFILE_BY_ID.get(key) or SENSOR_PROFILE_BY_CODE.get(key.upper())
    if resolved is None:
        raise ValueError(f"unknown OnHW sensor profile {profile!r}")
    return resolved


def extract_profile(values, profile: str | SensorProfile):
    """Select a registered profile from a full `[time, 13]` OnHW array."""
    import numpy as np

    resolved = resolve_sensor_profile(profile)
    sequence = np.asarray(values)
    if sequence.ndim != 2 or sequence.shape[1] != len(ONHW_CHANNEL_NAMES):
        raise ValueError(
            f"sensor profile selection requires a [time, 13] array, got {sequence.shape}"
        )
    return np.ascontiguousarray(sequence[:, resolved.channel_indices], dtype=np.float32)


def list_sensor_profiles() -> list[dict]:
    """Return JSON-serializable profile definitions in stable registry order."""
    return [
        {
            "profile_id": profile.profile_id,
            "group_code": profile.group_code,
            "groups": [group.name for group in profile.groups],
            "channel_indices": list(profile.channel_indices),
            "channel_names": list(profile.channel_names),
            "units": list(profile.units),
            "input_channels": profile.input_channels,
            "physical_description": profile.physical_description,
            "preprocessing": dict(profile.preprocessing),
            "dataset_compatibility": profile.dataset_compatibility,
            "sample_rate_hz": ONHW_SAMPLE_RATE_HZ,
        }
        for profile in SENSOR_PROFILES
    ]
