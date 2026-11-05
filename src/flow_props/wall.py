"""Wall quantities: skin friction, heat transfer, shear stress."""

from __future__ import annotations

import numpy as np
from cfd_io.dataset import Dataset, StructuredGrid


def wall_shear_stress(
    ds: Dataset,
    mu_wall: np.ndarray | float,
) -> np.ndarray:
    r"""Compute wall shear stress :math:`\tau_w = \mu \, \partial u / \partial \eta |_{wall}`.

    Uses a one-sided second-order finite difference at j=0.

    Args:
        ds: Dataset with StructuredGrid.  Must contain ``"uvel"`` in flow.
        mu_wall: Dynamic viscosity at the wall.  Scalar or array of shape ``(ni,)``.

    Returns:
        Array of shape ``(ni,)`` with :math:`\tau_w` at each i-station.
    """
    if not isinstance(ds.grid, StructuredGrid):
        raise TypeError("wall_shear_stress requires a StructuredGrid")

    u = ds.flow["uvel"].data[:, :, 0]  # (ni, nj)
    x = ds.grid.x[:, :, 0]
    y = ds.grid.y[:, :, 0]

    # wall-normal distances from wall (j=0) to j=1 and j=2
    h1 = np.sqrt((x[:, 1] - x[:, 0]) ** 2 + (y[:, 1] - y[:, 0]) ** 2)
    h2 = np.sqrt((x[:, 2] - x[:, 0]) ** 2 + (y[:, 2] - y[:, 0]) ** 2)

    # Lagrange derivative stencil at eta=0 (non-uniform spacing)
    c0 = -(h1 + h2) / (h1 * h2)
    c1 = h2 / (h1 * (h2 - h1))
    c2 = -h1 / (h2 * (h2 - h1))
    du_deta = c0 * u[:, 0] + c1 * u[:, 1] + c2 * u[:, 2]

    return mu_wall * du_deta


def skin_friction(
    ds: Dataset,
    mu_wall: np.ndarray | float,
    rho_inf: float,
    u_inf: float,
) -> np.ndarray:
    r"""Compute skin-friction coefficient :math:`C_f = \tau_w / (0.5 \rho_\infty u_\infty^2)`.

    Args:
        ds: Dataset with StructuredGrid.
        mu_wall: Dynamic viscosity at the wall.
        rho_inf: Freestream density.
        u_inf: Freestream velocity.

    Returns:
        Array of shape ``(ni,)`` with :math:`C_f`.
    """
    tau_w = wall_shear_stress(ds, mu_wall)
    q_inf = 0.5 * rho_inf * u_inf**2
    return tau_w / q_inf


def wall_heat_flux(
    ds: Dataset,
    k_wall: np.ndarray | float,
) -> np.ndarray:
    r"""Compute wall heat flux :math:`q_w = -k \, \partial T / \partial \eta |_{wall}`.

    Args:
        ds: Dataset with StructuredGrid.  Must contain ``"temp"`` in flow.
        k_wall: Thermal conductivity at the wall.

    Returns:
        Array of shape ``(ni,)`` with :math:`q_w`.
    """
    if not isinstance(ds.grid, StructuredGrid):
        raise TypeError("wall_heat_flux requires a StructuredGrid")

    T = ds.flow["temp"].data[:, :, 0]  # (ni, nj)
    x = ds.grid.x[:, :, 0]
    y = ds.grid.y[:, :, 0]

    h1 = np.sqrt((x[:, 1] - x[:, 0]) ** 2 + (y[:, 1] - y[:, 0]) ** 2)
    h2 = np.sqrt((x[:, 2] - x[:, 0]) ** 2 + (y[:, 2] - y[:, 0]) ** 2)

    c0 = -(h1 + h2) / (h1 * h2)
    c1 = h2 / (h1 * (h2 - h1))
    c2 = -h1 / (h2 * (h2 - h1))
    dT_deta = c0 * T[:, 0] + c1 * T[:, 1] + c2 * T[:, 2]

    return -k_wall * dT_deta


def stanton_number(
    ds: Dataset,
    k_wall: np.ndarray | float,
    rho_inf: float,
    u_inf: float,
    cp: float,
    T_wall: np.ndarray | float,
    T_inf: float,
) -> np.ndarray:
    r"""Compute Stanton number :math:`St = q_w / (\rho_\infty u_\infty c_p (T_{aw} - T_w))`.

    Note: uses (T_inf - T_wall) as the driving temperature difference.
    For adiabatic wall cases this will need adjustment.

    Args:
        ds: Dataset with StructuredGrid.
        k_wall: Thermal conductivity at the wall.
        rho_inf: Freestream density.
        u_inf: Freestream velocity.
        cp: Specific heat at constant pressure.
        T_wall: Wall temperature.
        T_inf: Freestream temperature (or recovery temperature).

    Returns:
        Array of shape ``(ni,)`` with Stanton number.
    """
    qw = wall_heat_flux(ds, k_wall)
    return qw / (rho_inf * u_inf * cp * (T_inf - T_wall))


def run_wall(
    ds: Dataset,
    mu_wall: float,
    rho_inf: float = 0.0,
    u_inf: float = 0.0,
    k_wall: float = 0.0,
    cp: float = 1005.0,
    T_wall: float = 300.0,
    T_inf: float = 300.0,
) -> dict[str, np.ndarray]:
    """Compute all applicable wall quantities for a dataset.

    Args:
        ds: Dataset with StructuredGrid.
        mu_wall: Dynamic viscosity at the wall [Pa*s].
        rho_inf: Freestream density [kg/m^3].  If nonzero with *u_inf*, Cf is computed.
        u_inf: Freestream velocity [m/s].
        k_wall: Thermal conductivity at wall [W/(m*K)].  If nonzero, heat flux is computed.
        cp: Specific heat [J/(kg*K)].
        T_wall: Wall temperature [K].
        T_inf: Freestream temperature [K].

    Returns:
        Dict of computed arrays.  Always contains ``"tau_w"``; conditionally
        contains ``"cf"``, ``"qw"``, ``"st"`` based on provided parameters.
    """
    result: dict[str, np.ndarray] = {}

    result["tau_w"] = wall_shear_stress(ds, mu_wall)

    if rho_inf and u_inf:
        result["cf"] = skin_friction(ds, mu_wall, rho_inf, u_inf)

    if k_wall:
        result["qw"] = wall_heat_flux(ds, k_wall)

        if rho_inf and u_inf:
            result["st"] = stanton_number(ds, k_wall, rho_inf, u_inf, cp, T_wall, T_inf)

    return result
