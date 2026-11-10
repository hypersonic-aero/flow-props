"""Boundary layer integral properties: delta, delta_star, theta, H."""

from __future__ import annotations

import numpy as np
from cfd_io.dataset import Dataset

# --------------------------------------------------
# public constants
# --------------------------------------------------

BL_EDGE_VELOCITY_RATIO = "velocity_ratio"
BL_EDGE_ENTHALPY_RATIO = "enthalpy_ratio"
BL_EDGE_VELOCITY_GRADIENT = "velocity_gradient"
BL_EDGE_HTOT_DERIVATIVE = "htot_derivative"


# --------------------------------------------------
# public API
# --------------------------------------------------


def total_enthalpy_profile(
    u: np.ndarray,
    temp: np.ndarray,
    gamma: float = 1.4,
    gas_constant: float = 287.05,
) -> np.ndarray:
    """Compute total enthalpy for a wall-normal profile.

    Args:
        u: Velocity profile.
        temp: Temperature profile.
        gamma: Specific heat ratio.
        gas_constant: Gas constant.

    Returns:
        Total enthalpy profile.
    """
    # build specific heat from ideal-gas relations
    cp = gamma * gas_constant / (gamma - 1.0)

    # build total enthalpy profile
    htot = cp * temp + 0.5 * u**2
    return htot


def boundary_layer_thickness(
    eta: np.ndarray,
    u: np.ndarray,
    u_edge: float,
    threshold: float = 0.99,
) -> float:
    """Compute boundary layer thickness from a velocity-ratio criterion.

    Args:
        eta: Wall-normal distance array.
        u: Velocity profile.
        u_edge: Edge velocity.
        threshold: Fraction of edge velocity defining delta.

    Returns:
        Boundary layer thickness. Returns ``eta[-1]`` if the threshold is not reached.
    """
    # build normalized velocity profile
    ratio = u / u_edge

    # find the first threshold crossing
    delta = _find_first_crossing(eta, ratio, threshold, direction="increasing")
    return delta


def enthalpy_boundary_layer_thickness(
    eta: np.ndarray,
    u: np.ndarray,
    temp: np.ndarray,
    h_edge: float,
    threshold: float = 0.99,
    gamma: float = 1.4,
    gas_constant: float = 287.05,
) -> float:
    """Compute boundary layer thickness from a total-enthalpy ratio criterion.

    Args:
        eta: Wall-normal distance array.
        u: Velocity profile.
        temp: Temperature profile.
        h_edge: Edge total enthalpy.
        threshold: Fraction of edge total enthalpy defining delta.
        gamma: Specific heat ratio.
        gas_constant: Gas constant.

    Returns:
        Boundary layer thickness. Returns ``eta[-1]`` if the threshold is not reached.
    """
    # build normalized total enthalpy profile
    htot = total_enthalpy_profile(u, temp, gamma=gamma, gas_constant=gas_constant)
    ratio = htot / h_edge

    # find the first threshold crossing
    delta = _find_first_crossing(eta, ratio, threshold, direction="increasing")
    return delta


def htot_derivative_thickness(
    eta: np.ndarray,
    u: np.ndarray,
    temp: np.ndarray,
    gamma: float = 1.4,
    gas_constant: float = 287.05,
) -> float:
    """Compute BL thickness from the minimum of d(h_tot)/d_eta (Bertin criterion).

    Locates the wall-normal position where the total-enthalpy gradient is most
    negative.  For hypersonic body-fitted grids this is more robust than a
    fixed velocity ratio, especially when the wall is cold and the enthalpy
    profile has a strong near-wall dip.

    Args:
        eta: Wall-normal distance array.
        u: Velocity profile.
        temp: Temperature profile.
        gamma: Specific heat ratio.
        gas_constant: Gas constant [J/(kg K)].

    Returns:
        BL thickness (eta at minimum d(h_tot)/d_eta).
        Returns ``eta[-1]`` if the minimum cannot be located.
    """
    # build total enthalpy profile
    htot = total_enthalpy_profile(u, temp, gamma=gamma, gas_constant=gas_constant)

    # compute wall-normal derivative
    dh_deta = np.gradient(htot, eta)

    # find the index of the minimum derivative (most negative value)
    idx_min = int(np.argmin(dh_deta))

    # guard against the minimum being at the boundary
    if idx_min == 0 or idx_min == len(eta) - 1:
        return float(eta[-1])

    return float(eta[idx_min])


def velocity_gradient_thickness(
    eta: np.ndarray,
    u: np.ndarray,
    gradient_threshold: float,
) -> float:
    """Compute boundary layer thickness from a velocity-gradient criterion.

    Args:
        eta: Wall-normal distance array.
        u: Velocity profile.
        gradient_threshold: Threshold for ``du/deta``.

    Returns:
        Boundary layer thickness. Returns ``eta[-1]`` if the threshold is not reached.
    """
    # build velocity-gradient profile
    du_deta = np.gradient(u, eta)

    # find the first threshold crossing in the decreasing gradient profile
    delta = _find_first_crossing(
        eta,
        du_deta,
        gradient_threshold,
        direction="decreasing",
    )
    return delta


def boundary_layer_edge(
    eta: np.ndarray,
    u: np.ndarray,
    criterion: str = BL_EDGE_VELOCITY_RATIO,
    threshold: float = 0.99,
    u_edge: float | None = None,
    temp: np.ndarray | None = None,
    h_edge: float | None = None,
    gradient_threshold: float = 0.0,
    gamma: float = 1.4,
    gas_constant: float = 287.05,
) -> float:
    """Compute boundary layer thickness from a named edge criterion.

    Args:
        eta: Wall-normal distance array.
        u: Velocity profile.
        criterion: Boundary layer edge criterion name.
        threshold: Threshold for ratio-based criteria.
        u_edge: Edge velocity for the velocity-ratio criterion.
        temp: Temperature profile for the enthalpy-ratio criterion.
        h_edge: Edge total enthalpy for the enthalpy-ratio criterion.
        gradient_threshold: ``du/deta`` threshold for the gradient criterion.
        gamma: Specific heat ratio.
        gas_constant: Gas constant.

    Returns:
        Boundary layer thickness.

    Raises:
        ValueError: If the criterion is unsupported or required inputs are missing.
    """
    # select boundary layer edge criterion
    if criterion == BL_EDGE_VELOCITY_RATIO:
        edge_velocity = u_edge if u_edge is not None else float(u[-1])
        return boundary_layer_thickness(eta, u, edge_velocity, threshold=threshold)

    if criterion == BL_EDGE_ENTHALPY_RATIO:
        if temp is None:
            raise ValueError("temp is required for the enthalpy_ratio criterion")

        edge_enthalpy = h_edge
        if edge_enthalpy is None:
            h_profile = total_enthalpy_profile(u, temp, gamma=gamma, gas_constant=gas_constant)
            edge_enthalpy = float(h_profile[-1])

        return enthalpy_boundary_layer_thickness(
            eta,
            u,
            temp,
            edge_enthalpy,
            threshold=threshold,
            gamma=gamma,
            gas_constant=gas_constant,
        )

    if criterion == BL_EDGE_VELOCITY_GRADIENT:
        return velocity_gradient_thickness(eta, u, gradient_threshold=gradient_threshold)

    if criterion == BL_EDGE_HTOT_DERIVATIVE:
        if temp is None:
            raise ValueError("temp is required for the htot_derivative criterion")
        return htot_derivative_thickness(
            eta, u, temp, gamma=gamma, gas_constant=gas_constant
        )

    raise ValueError(f"unsupported boundary layer criterion: {criterion}")


def displacement_thickness(
    eta: np.ndarray,
    rho: np.ndarray,
    u: np.ndarray,
    rho_edge: float,
    u_edge: float,
) -> float:
    r"""Compute displacement thickness.

    Args:
        eta: Wall-normal distance array.
        rho: Density profile.
        u: Velocity profile.
        rho_edge: Edge density.
        u_edge: Edge velocity.

    Returns:
        Displacement thickness.
    """
    # build displacement-thickness integrand
    integrand = 1.0 - (rho * u) / (rho_edge * u_edge)

    # integrate along the profile
    delta_star = np.trapezoid(integrand, eta)
    return float(delta_star)


def momentum_thickness(
    eta: np.ndarray,
    rho: np.ndarray,
    u: np.ndarray,
    rho_edge: float,
    u_edge: float,
) -> float:
    r"""Compute momentum thickness.

    Args:
        eta: Wall-normal distance array.
        rho: Density profile.
        u: Velocity profile.
        rho_edge: Edge density.
        u_edge: Edge velocity.

    Returns:
        Momentum thickness.
    """
    # build momentum-thickness integrand
    mass_ratio = (rho * u) / (rho_edge * u_edge)
    integrand = mass_ratio * (1.0 - u / u_edge)

    # integrate along the profile
    theta = np.trapezoid(integrand, eta)
    return float(theta)


def shape_factor(delta_star: float, theta: float) -> float:
    """Compute the incompressible shape factor ``H = delta_star / theta``."""
    # protect against division by zero
    if theta == 0.0:
        return float("inf")

    shape = delta_star / theta
    return shape


def bl_properties(
    eta: np.ndarray,
    u: np.ndarray,
    rho: np.ndarray,
    u_edge: float,
    rho_edge: float,
    threshold: float = 0.99,
    criterion: str = BL_EDGE_VELOCITY_RATIO,
    temp: np.ndarray | None = None,
    h_edge: float | None = None,
    gradient_threshold: float = 0.0,
    gamma: float = 1.4,
    gas_constant: float = 287.05,
) -> dict[str, float]:
    """Compute standard boundary layer integral properties for one profile.

    ``delta_star`` and ``theta`` are integrated from the wall to ``delta`` itself (the
    profile is cut there, with an end point interpolated at ``delta``), not over the
    whole profile.

    Args:
        eta: Wall-normal distance array.
        u: Velocity profile.
        rho: Density profile.
        u_edge: Edge velocity.
        rho_edge: Edge density.
        threshold: Threshold for ratio-based criteria.
        criterion: Boundary layer edge criterion name.
        temp: Temperature profile if required by the selected criterion.
        h_edge: Edge total enthalpy if required by the selected criterion.
        gradient_threshold: ``du/deta`` threshold for the gradient criterion.
        gamma: Specific heat ratio.
        gas_constant: Gas constant.

    Returns:
        Dict with keys ``delta``, ``delta_star``, ``theta``, and ``H``.
    """
    # build boundary layer edge location
    delta = boundary_layer_edge(
        eta,
        u,
        criterion=criterion,
        threshold=threshold,
        u_edge=u_edge,
        temp=temp,
        h_edge=h_edge,
        gradient_threshold=gradient_threshold,
        gamma=gamma,
        gas_constant=gas_constant,
    )

    # integrate across the boundary layer only (0 to delta): integrating over the full
    # grid line accumulates spurious freestream contributions over the large inviscid
    # extent and can flip the sign
    eta_bl, rho_bl, u_bl = _cut_at_edge(eta, rho, u, delta)

    # build integral thickness values
    delta_star = displacement_thickness(eta_bl, rho_bl, u_bl, rho_edge, u_edge)
    theta = momentum_thickness(eta_bl, rho_bl, u_bl, rho_edge, u_edge)
    h_shape = shape_factor(delta_star, theta)

    result = {
        "delta": delta,
        "delta_star": delta_star,
        "theta": theta,
        "H": h_shape,
    }
    return result


def run_bl(
    ds: Dataset,
    stations: list[int],
    method: str = "grid_line",
    n_eta: int | None = None,
    eta_max: float | None = None,
    u_edge: float = 0.0,
    rho_edge: float = 0.0,
    threshold: float = 0.99,
    criterion: str = BL_EDGE_VELOCITY_RATIO,
    gradient_threshold: float = 0.0,
    gamma: float = 1.4,
    gas_constant: float = 287.05,
) -> list[dict[str, float]]:
    """Extract profiles and compute boundary layer properties at selected stations.

    Args:
        ds: Dataset with StructuredGrid.
        stations: List of i-station indices.
        method: Profile extraction method name.
        n_eta: Number of interpolation samples for ``wall_normal_interp``.
        eta_max: Maximum wall-normal distance for ``wall_normal_interp``.
        u_edge: Edge velocity. ``0`` means auto-detect from profile.
        rho_edge: Edge density. ``0`` means auto-detect from profile.
        threshold: Threshold for ratio-based criteria.
        criterion: Boundary layer edge criterion name.
        gradient_threshold: ``du/deta`` threshold for the gradient criterion.
        gamma: Specific heat ratio.
        gas_constant: Gas constant.

    Returns:
        List of dicts, one per station.
    """
    from flow_props.profiles import extract_profiles

    # extract wall-normal profiles first
    profiles = extract_profiles(ds, stations, method=method, n_eta=n_eta, eta_max=eta_max)

    results: list[dict[str, float]] = []
    for station, profile in zip(stations, profiles):
        # read profile variables with conservative fallbacks
        u = _get_profile_field(profile, ("u_tangent", "uvel", "u", "velocity"))
        rho = _get_profile_field(profile, ("density", "dens", "rho"), default=np.ones_like(u))
        temp = _get_profile_field(profile, ("temp", "temperature"), default=None)
        eta = profile["eta"]

        # detect edge values from the profile if not provided explicitly
        edge_velocity = u_edge if u_edge else float(u[-1])
        edge_density = rho_edge if rho_edge else float(rho[-1])

        # build edge enthalpy if the selected criterion needs it
        edge_enthalpy = None
        if criterion == BL_EDGE_ENTHALPY_RATIO:
            if temp is None:
                raise KeyError("temperature field is required for the enthalpy_ratio criterion")

            h_profile = total_enthalpy_profile(u, temp, gamma=gamma, gas_constant=gas_constant)
            edge_enthalpy = float(h_profile[-1])

        # compute boundary layer properties at this station
        props = bl_properties(
            eta,
            u,
            rho,
            edge_velocity,
            edge_density,
            threshold=threshold,
            criterion=criterion,
            temp=temp,
            h_edge=edge_enthalpy,
            gradient_threshold=gradient_threshold,
            gamma=gamma,
            gas_constant=gas_constant,
        )
        props["station"] = station
        results.append(props)

    return results


# --------------------------------------------------
# helpers
# --------------------------------------------------


def _cut_at_edge(
    eta: np.ndarray,
    rho: np.ndarray,
    u: np.ndarray,
    delta: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The profile from the wall to delta, ending with a point at delta itself.

    The grid points below delta are kept and one point at delta is added, its density
    and velocity interpolated linearly between the grid points around it. Stopping at
    the last grid point below delta would leave the integrals short by up to one grid
    spacing times the integrand there (about 1 % on a grid with 40 points across the
    layer).

    Args:
        eta: Wall-normal distance array, increasing from the wall.
        rho: Density profile.
        u: Velocity profile.
        delta: Boundary layer thickness, within the range of ``eta``.

    Returns:
        ``eta``, ``rho`` and ``u`` from the wall to delta.
    """

    # the grid points below delta
    below = eta < delta

    # the end point at delta, interpolated between the grid points around it
    rho_at_delta = float(np.interp(delta, eta, rho))
    u_at_delta = float(np.interp(delta, eta, u))

    # join them
    eta_bl = np.append(eta[below], delta)
    rho_bl = np.append(rho[below], rho_at_delta)
    u_bl = np.append(u[below], u_at_delta)

    return eta_bl, rho_bl, u_bl


def _find_first_crossing(
    eta: np.ndarray,
    values: np.ndarray,
    target: float,
    direction: str,
) -> float:
    """Find the first interpolated target crossing in a profile.

    Args:
        eta: Coordinate array.
        values: Profile values.
        target: Target value.
        direction: Either ``"increasing"`` or ``"decreasing"``.

    Returns:
        Interpolated coordinate of the first crossing. Returns ``eta[-1]`` if no crossing is found.
    """
    # handle the case where the target is already met at the wall
    if direction == "increasing" and values[0] >= target:
        return float(eta[0])
    if direction == "decreasing" and values[0] <= target:
        return float(eta[0])

    # scan neighboring points for the first crossing
    for index in range(1, len(eta)):
        previous_value = float(values[index - 1])
        current_value = float(values[index])

        if direction == "increasing":
            crossed = previous_value < target <= current_value
        elif direction == "decreasing":
            crossed = previous_value > target >= current_value
        else:
            raise ValueError(f"unsupported crossing direction: {direction}")

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
    default: np.ndarray | None = None,
) -> np.ndarray | None:
    """Read the first available field from a profile dictionary."""
    # search candidate names in order
    for name in candidates:
        if name in profile:
            return profile[name]

    # return configured default if no field matches
    if default is not None:
        return default

    names = ", ".join(candidates)
    raise KeyError(f"none of the profile fields were found: {names}")
