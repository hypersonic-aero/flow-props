"""Entropy layer thickness from wall-normal CFD profiles."""

from __future__ import annotations

import numpy as np
from cfd_io.dataset import Dataset

# --------------------------------------------------
# public constants
# --------------------------------------------------

ENTROPY_LAYER_WALL_FRACTION = "wall_fraction"


# --------------------------------------------------
# public API
# --------------------------------------------------


def entropy_difference(
    temp: np.ndarray,
    pressure: np.ndarray,
    temp_ref: float,
    pressure_ref: float,
    gamma: float = 1.4,
    gas_constant: float = 287.05,
) -> np.ndarray:
    """Compute entropy difference relative to a reference state.

    Args:
        temp: Temperature profile.
        pressure: Pressure profile.
        temp_ref: Reference temperature.
        pressure_ref: Reference pressure.
        gamma: Specific heat ratio.
        gas_constant: Gas constant.

    Returns:
        Entropy-difference profile.
    """
    # build specific heat from ideal-gas relations
    cp = gamma * gas_constant / (gamma - 1.0)

    # build entropy difference relative to the reference state
    delta_s = cp * np.log(temp / temp_ref) - gas_constant * np.log(pressure / pressure_ref)
    return delta_s


def entropy_layer_thickness(
    eta: np.ndarray,
    temp: np.ndarray,
    pressure: np.ndarray,
    temp_ref: float,
    pressure_ref: float,
    criterion: str = ENTROPY_LAYER_WALL_FRACTION,
    threshold: float = 0.25,
    gamma: float = 1.4,
    gas_constant: float = 287.05,
) -> float:
    """Compute entropy layer thickness for a single profile.

    Args:
        eta: Wall-normal distance array.
        temp: Temperature profile.
        pressure: Pressure profile.
        temp_ref: Reference temperature.
        pressure_ref: Reference pressure.
        criterion: Entropy-layer criterion name.
        threshold: Target normalized entropy value.
        gamma: Specific heat ratio.
        gas_constant: Gas constant.

    Returns:
        Entropy layer thickness.

    Raises:
        ValueError: If the criterion is unsupported or the wall entropy difference is zero.
    """
    # build entropy-difference profile
    delta_s = entropy_difference(
        temp,
        pressure,
        temp_ref,
        pressure_ref,
        gamma=gamma,
        gas_constant=gas_constant,
    )

    # validate supported criteria
    if criterion != ENTROPY_LAYER_WALL_FRACTION:
        raise ValueError(f"unsupported entropy-layer criterion: {criterion}")

    # normalize with the wall entropy difference
    wall_delta_s = float(delta_s[0])
    if wall_delta_s == 0.0:
        raise ValueError("wall entropy difference is zero; cannot normalize entropy layer profile")

    normalized_delta_s = delta_s / wall_delta_s

    # find the first point where the entropy profile drops below the target
    delta = _find_first_downward_crossing(eta, normalized_delta_s, threshold)
    return delta


def run_entropy(
    ds: Dataset,
    stations: list[int],
    method: str = "grid_line",
    n_eta: int | None = None,
    eta_max: float | None = None,
    criterion: str = ENTROPY_LAYER_WALL_FRACTION,
    threshold: float = 0.25,
    gamma: float = 1.4,
    gas_constant: float = 287.05,
    temp_edge: float = 0.0,
    pressure_edge: float = 0.0,
) -> list[dict[str, float]]:
    """Extract profiles and compute entropy-layer thickness at selected stations.

    Args:
        ds: Dataset with StructuredGrid.
        stations: List of i-station indices.
        method: Profile extraction method name.
        n_eta: Number of interpolation samples for ``wall_normal_interp``.
        eta_max: Maximum wall-normal distance for ``wall_normal_interp``.
        criterion: Entropy-layer criterion name.
        threshold: Normalized entropy threshold.
        gamma: Specific heat ratio.
        gas_constant: Gas constant.
        temp_edge: Edge temperature. ``0`` means auto-detect from profile.
        pressure_edge: Edge pressure. ``0`` means auto-detect from profile.

    Returns:
        List of dicts with ``station`` and ``delta_entropy``.
    """
    from flow_props.profiles import extract_profiles

    # extract wall-normal profiles first
    profiles = extract_profiles(ds, stations, method=method, n_eta=n_eta, eta_max=eta_max)

    results: list[dict[str, float]] = []
    for station, profile in zip(stations, profiles):
        # read required thermodynamic fields
        temp = _get_profile_field(profile, ("temp", "temperature"))
        pressure = _get_profile_field(profile, ("pres", "pressure", "p"))
        eta = profile["eta"]

        # detect edge values from the profile if not provided explicitly
        temp_ref = temp_edge if temp_edge else float(temp[-1])
        pressure_ref = pressure_edge if pressure_edge else float(pressure[-1])

        # compute entropy-layer thickness at this station
        delta_entropy = entropy_layer_thickness(
            eta,
            temp,
            pressure,
            temp_ref,
            pressure_ref,
            criterion=criterion,
            threshold=threshold,
            gamma=gamma,
            gas_constant=gas_constant,
        )

        results.append({"station": station, "delta_entropy": delta_entropy})

    return results


# --------------------------------------------------
# helpers
# --------------------------------------------------


def _find_first_downward_crossing(
    eta: np.ndarray,
    values: np.ndarray,
    target: float,
) -> float:
    """Find the first interpolated downward crossing of a target value."""
    # handle the case where the target is already met at the wall
    if values[0] <= target:
        return float(eta[0])

    # scan neighboring points for the first crossing
    for index in range(1, len(eta)):
        previous_value = float(values[index - 1])
        current_value = float(values[index])
        crossed = previous_value > target >= current_value

        if not crossed:
            continue

        delta_value = current_value - previous_value
        if delta_value == 0.0:
            return float(eta[index])

        fraction = (target - previous_value) / delta_value
        eta_value = eta[index - 1] + fraction * (eta[index] - eta[index - 1])
        return float(eta_value)

    # return the profile limit if the threshold is not reached
    return float(eta[-1])


def _get_profile_field(
    profile: dict[str, np.ndarray],
    candidates: tuple[str, ...],
) -> np.ndarray:
    """Read the first available field from a profile dictionary."""
    # search candidate names in order
    for name in candidates:
        if name in profile:
            return profile[name]

    names = ", ".join(candidates)
    raise KeyError(f"none of the profile fields were found: {names}")
