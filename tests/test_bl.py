"""Tests for flow_props.bl boundary layer integral properties."""

import numpy as np
import pytest
from cfd_io.dataset import Dataset, Field, StructuredGrid

from flow_props.bl import (
    BL_EDGE_ENTHALPY_RATIO,
    BL_EDGE_HTOT_DERIVATIVE,
    BL_EDGE_VELOCITY_GRADIENT,
    bl_properties,
    boundary_layer_edge,
    boundary_layer_thickness,
    displacement_thickness,
    enthalpy_boundary_layer_thickness,
    htot_derivative_thickness,
    momentum_thickness,
    run_bl,
    shape_factor,
    total_enthalpy_profile,
    velocity_gradient_thickness,
)
from flow_props.profiles import PROFILE_METHOD_WALL_NORMAL_INTERP


def _make_run_bl_dataset(ni: int = 6, nj: int = 40) -> Dataset:
    """Build a minimal structured dataset for run_bl integration testing."""
    x = np.zeros((ni, nj, 1))
    y = np.zeros((ni, nj, 1))
    for i in range(ni):
        for j in range(nj):
            x[i, j, 0] = float(i)
            y[i, j, 0] = 0.001 * float(j)
    z = np.zeros_like(x)
    grid = StructuredGrid(x=x, y=y, z=z)

    uvel = np.zeros((ni, nj, 1))
    dens = np.zeros((ni, nj, 1))
    temp = np.zeros((ni, nj, 1))
    for j in range(nj):
        eta_fraction = float(j) / float(nj - 1)
        uvel[:, j, 0] = eta_fraction
        dens[:, j, 0] = 1.0 - 0.2 * eta_fraction
        temp[:, j, 0] = 300.0 + 10.0 * eta_fraction

    flow = {
        "uvel": Field(data=uvel, association="node"),
        "dens": Field(data=dens, association="node"),
        "temp": Field(data=temp, association="node"),
    }
    return Dataset(grid=grid, flow=flow, attrs={"format": "test"})


def _make_rotated_run_bl_dataset(ni: int = 6, nj: int = 40, angle_deg: float = 35.0) -> Dataset:
    """Build a rotated Dataset whose BL result should match the reference case."""
    angle_rad = np.deg2rad(angle_deg)
    tangent = np.array([np.cos(angle_rad), np.sin(angle_rad)], dtype=float)
    normal = np.array([-np.sin(angle_rad), np.cos(angle_rad)], dtype=float)

    streamwise_spacing = 1.0
    normal_spacing = 0.001

    x = np.zeros((ni, nj, 1))
    y = np.zeros((ni, nj, 1))
    uvel = np.zeros((ni, nj, 1))
    vvel = np.zeros((ni, nj, 1))
    dens = np.zeros((ni, nj, 1))
    temp = np.zeros((ni, nj, 1))

    for i in range(ni):
        for j in range(nj):
            eta_value = normal_spacing * float(j)
            eta_fraction = float(j) / float(nj - 1)

            # build rotated grid coordinates from the local tangent and normal basis
            position = float(i) * streamwise_spacing * tangent + eta_value * normal
            x[i, j, 0] = position[0]
            y[i, j, 0] = position[1]

            # build the physical tangential and normal velocity components
            u_tangent = eta_fraction
            u_normal = 0.2 * eta_fraction**2
            velocity = u_tangent * tangent + u_normal * normal
            uvel[i, j, 0] = velocity[0]
            vvel[i, j, 0] = velocity[1]
            dens[i, j, 0] = 1.0 - 0.2 * eta_fraction
            temp[i, j, 0] = 300.0 + 10.0 * eta_fraction

    z = np.zeros_like(x)
    grid = StructuredGrid(x=x, y=y, z=z)
    flow = {
        "uvel": Field(data=uvel, association="node"),
        "vvel": Field(data=vvel, association="node"),
        "dens": Field(data=dens, association="node"),
        "temp": Field(data=temp, association="node"),
    }
    return Dataset(grid=grid, flow=flow, attrs={"format": "test"})


# --------------------------------------------------
# boundary_layer_thickness tests
# --------------------------------------------------


def test_bl_thickness_linear():
    """Linear velocity profile u = eta: delta should be at u_edge * threshold."""
    eta = np.linspace(0, 1, 200)
    u = eta.copy()
    u_edge = 1.0
    delta = boundary_layer_thickness(eta, u, u_edge, threshold=0.99)
    assert delta == pytest.approx(0.99, rel=1e-3)


def test_bl_thickness_returns_last_if_not_reached():
    """If u never hits 99% of edge, return eta[-1]."""
    eta = np.linspace(0, 1, 100)
    u = 0.5 * eta  # max is 0.5, threshold = 0.99
    delta = boundary_layer_thickness(eta, u, u_edge=1.0, threshold=0.99)
    assert delta == pytest.approx(eta[-1])


def test_total_enthalpy_profile():
    u = np.array([0.0, 2.0])
    temp = np.array([1.0, 1.0])
    htot = total_enthalpy_profile(u, temp, gamma=2.0, gas_constant=1.0)
    np.testing.assert_allclose(htot, np.array([2.0, 4.0]))


def test_enthalpy_boundary_layer_thickness_linear_profile():
    eta = np.linspace(0.0, 1.0, 200)
    u = np.zeros_like(eta)
    temp = 0.5 + 0.5 * eta
    delta = enthalpy_boundary_layer_thickness(
        eta,
        u,
        temp,
        h_edge=2.0,
        threshold=0.9,
        gamma=2.0,
        gas_constant=1.0,
    )
    assert delta == pytest.approx(0.8, rel=1e-3)


def test_htot_derivative_thickness_locates_interior_minimum():
    """htot_derivative_thickness returns the interior eta of steepest enthalpy descent."""
    eta = np.linspace(0, 0.01, 200)
    u = np.tanh(eta / 0.003) * 100.0
    temp = 300.0 - 100.0 * np.tanh((eta - 0.004) / 0.0005)
    delta = htot_derivative_thickness(eta, u, temp, gamma=1.4, gas_constant=287.05)
    assert 0.0 < delta < 0.01
    # the constructed profile has its steepest h_tot descent at eta = 0.004
    assert abs(delta - 0.004) < 5.0e-4


def test_htot_derivative_in_boundary_layer_edge():
    """boundary_layer_edge dispatches to htot_derivative when the criterion matches."""
    eta = np.linspace(0, 0.01, 200)
    u = np.tanh(eta / 0.003) * 100.0
    temp = 300.0 - 100.0 * np.tanh((eta - 0.004) / 0.0005)
    delta = boundary_layer_edge(
        eta, u, criterion=BL_EDGE_HTOT_DERIVATIVE, temp=temp, gamma=1.4, gas_constant=287.05
    )
    assert 0.0 < delta < 0.01


def test_htot_derivative_wall_minimum_falls_back_to_outer_edge():
    """A near-wall enthalpy spike puts the gradient minimum at the wall: return eta[-1]."""
    eta = np.linspace(0, 0.01, 200)
    u = np.tanh(eta / 0.002) * 876.0
    temp = 300.0 + 200.0 * np.exp(-eta / 0.001)
    delta = htot_derivative_thickness(eta, u, temp, gamma=1.4, gas_constant=287.05)
    assert delta == eta[-1]


def test_htot_derivative_requires_temperature():
    """boundary_layer_edge raises when temp is missing for the htot_derivative criterion."""
    eta = np.linspace(0, 0.01, 50)
    u = np.tanh(eta / 0.003) * 100.0
    with pytest.raises(ValueError):
        boundary_layer_edge(eta, u, criterion=BL_EDGE_HTOT_DERIVATIVE)


def test_velocity_gradient_thickness():
    eta = np.linspace(0.0, 1.0, 500)
    u = 2.0 * eta - eta**2
    delta = velocity_gradient_thickness(eta, u, gradient_threshold=0.2)
    assert delta == pytest.approx(0.9, rel=1e-3)


def test_boundary_layer_edge_dispatches_gradient_criterion():
    eta = np.linspace(0.0, 1.0, 500)
    u = 2.0 * eta - eta**2
    delta = boundary_layer_edge(
        eta,
        u,
        criterion=BL_EDGE_VELOCITY_GRADIENT,
        gradient_threshold=0.2,
    )
    assert delta == pytest.approx(0.9, rel=1e-3)


def test_bl_properties_enthalpy_ratio():
    eta = np.linspace(0.0, 1.0, 200)
    u = eta.copy()
    rho = np.ones_like(eta)
    temp = 0.5 + 0.5 * eta
    result = bl_properties(
        eta,
        u,
        rho,
        u_edge=1.0,
        rho_edge=1.0,
        criterion=BL_EDGE_ENTHALPY_RATIO,
        threshold=0.95,
        temp=temp,
        gamma=2.0,
        gas_constant=1.0,
    )
    assert result["delta"] > 0.0
    assert result["delta_star"] > 0.0


# --------------------------------------------------
# displacement_thickness tests
# --------------------------------------------------


def test_displacement_thickness_blasius():
    """For incompressible flow with constant rho, delta_star = integral(1 - u/u_e) d_eta."""
    eta = np.linspace(0, 1, 1000)
    # build simple parabolic profile
    u_edge = 1.0
    u = 2 * eta - eta**2
    rho = np.ones_like(eta)
    ds = displacement_thickness(eta, rho, u, rho_edge=1.0, u_edge=u_edge)
    # check analytical value for the parabolic profile
    assert ds == pytest.approx(1 / 3, rel=1e-3)


# --------------------------------------------------
# momentum_thickness tests
# --------------------------------------------------


def test_momentum_thickness_parabolic():
    """Momentum thickness for parabolic profile u = 2*eta - eta^2."""
    eta = np.linspace(0, 1, 1000)
    u_edge = 1.0
    u = 2 * eta - eta**2
    rho = np.ones_like(eta)
    theta = momentum_thickness(eta, rho, u, rho_edge=1.0, u_edge=u_edge)
    # check analytical value for the parabolic profile
    assert theta == pytest.approx(2 / 15, rel=1e-3)


# --------------------------------------------------
# shape_factor tests
# --------------------------------------------------


def test_shape_factor():
    assert shape_factor(1.0, 0.5) == pytest.approx(2.0)


def test_shape_factor_zero_theta():
    assert shape_factor(1.0, 0.0) == float("inf")


# --------------------------------------------------
# bl_properties integration tests
# --------------------------------------------------


def test_bl_properties_returns_all_keys():
    eta = np.linspace(0, 1, 200)
    u = eta.copy()
    rho = np.ones_like(eta)
    result = bl_properties(eta, u, rho, u_edge=1.0, rho_edge=1.0)
    assert set(result.keys()) == {"delta", "delta_star", "theta", "H"}
    assert result["delta"] > 0
    assert result["theta"] > 0
    assert result["H"] == pytest.approx(result["delta_star"] / result["theta"])


def test_bl_properties_integrates_up_to_delta_itself():
    """A linear profile on a coarse grid: delta = 0.99 lies between grid points, and the
    displacement thickness is the exact integral of 1 - u from 0 to 0.99, with nothing
    from the profile beyond delta."""

    # u = eta up to 1, then freestream out to 3, on steps of 0.1
    eta = np.linspace(0.0, 3.0, 31)
    u = np.minimum(eta, 1.0)
    rho = np.ones_like(eta)
    result = bl_properties(eta, u, rho, u_edge=1.0, rho_edge=1.0)

    # check: delta interpolated, delta* exact (trapezoid on a linear integrand)
    assert result["delta"] == pytest.approx(0.99, rel=1e-12)
    assert result["delta_star"] == pytest.approx(0.99 - 0.99**2 / 2.0, rel=1e-12)


def test_run_bl_accepts_dens_field_name():
    ds = _make_run_bl_dataset()
    result = run_bl(ds, stations=[2], threshold=0.99)
    assert len(result) == 1
    assert result[0]["station"] == 2
    assert result[0]["delta"] > 0.0
    assert result[0]["delta_star"] > 0.0
    assert result[0]["theta"] > 0.0


def test_run_bl_accepts_wall_normal_interp_method():
    ds = _make_run_bl_dataset()
    result = run_bl(
        ds,
        stations=[2],
        method=PROFILE_METHOD_WALL_NORMAL_INTERP,
        n_eta=40,
        eta_max=0.039,
        threshold=0.99,
    )
    assert len(result) == 1
    assert result[0]["delta"] > 0.0


def test_run_bl_prefers_tangential_velocity_component_on_rotated_grid():
    reference_ds = _make_run_bl_dataset()
    rotated_ds = _make_rotated_run_bl_dataset()

    reference_result = run_bl(reference_ds, stations=[2], threshold=0.99)
    rotated_result = run_bl(rotated_ds, stations=[2], threshold=0.99)

    assert rotated_result[0]["delta"] == pytest.approx(reference_result[0]["delta"], rel=1.0e-6)
    assert rotated_result[0]["delta_star"] == pytest.approx(reference_result[0]["delta_star"], rel=1.0e-6)
    assert rotated_result[0]["theta"] == pytest.approx(reference_result[0]["theta"], rel=1.0e-6)
