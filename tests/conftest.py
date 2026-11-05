"""Shared synthetic solutions for the tests: small structured datasets with known answers."""

# --------------------------------------------------
# load necessary modules
# --------------------------------------------------
from __future__ import annotations

import numpy as np
from cfd_io.dataset import Dataset, Field, StructuredGrid

# --------------------------------------------------
# constants
# --------------------------------------------------

# freestream pressure of the shock solution [Pa]
FREESTREAM_PRESSURE = 1000.0


# --------------------------------------------------
# builders
# --------------------------------------------------
def boundary_layer_solution(ni: int = 6, nj: int = 40) -> Dataset:
    """Flat wall with a linear boundary-layer profile (no shock): velocity 0 to 1.

    Args:
        ni: Grid lines along the wall.
        nj: Points per grid line.

    Returns:
        Dataset with uvel, vvel, dens, temp and pres on a uniform grid.
    """

    # grid: wall at y = 0, straight j-lines
    x = np.zeros((ni, nj, 1))
    y = np.zeros((ni, nj, 1))
    for i in range(ni):
        for j in range(nj):
            x[i, j, 0] = 0.1 * float(i)
            y[i, j, 0] = 0.001 * float(j)

    # flow: linear velocity, hot wall, constant pressure
    uvel = np.zeros((ni, nj, 1))
    dens = np.zeros((ni, nj, 1))
    temp = np.zeros((ni, nj, 1))
    for j in range(nj):
        eta_fraction = float(j) / float(nj - 1)
        uvel[:, j, 0] = eta_fraction
        dens[:, j, 0] = 1.0 - 0.2 * eta_fraction
        temp[:, j, 0] = 300.0 - 240.0 * eta_fraction
    flow = {
        "uvel": Field(data=uvel, association="node"),
        "vvel": Field(data=np.zeros_like(uvel), association="node"),
        "dens": Field(data=dens, association="node"),
        "temp": Field(data=temp, association="node"),
        "pres": Field(data=np.full_like(uvel, 2000.0), association="node"),
    }

    return Dataset(grid=StructuredGrid(x=x, y=y, z=np.zeros_like(x)), flow=flow, attrs={})


def shock_solution(ni: int = 21, nj: int = 81, width: float = 1.0) -> tuple[Dataset, np.ndarray]:
    """Flat wall with a captured shock at a known fractional j on every grid line.

    The j-lines are straight and stretched toward the wall. The pressure jumps from
    freestream (outer end) to a post-shock value over about 2 * width cells, with a
    weak post-shock gradient and a jump that weakens downstream (40 to 0.5 times
    freestream), like a bow shock seen from the nose to far downstream.

    Args:
        ni: Grid lines along the wall.
        nj: Points per grid line.
        width: Shock width parameter of the tanh profile, in cells.

    Returns:
        The dataset, and the true fractional j of the shock on each line.
    """

    # grid: wall at y = 0, j-lines straight up, stretched toward the wall
    wall_x = np.linspace(0.0, 0.4, ni)
    s = np.linspace(0.0, 1.0, nj)
    eta = 0.05 * (np.exp(2.0 * s) - 1.0) / (np.exp(2.0) - 1.0)
    x = np.repeat(wall_x[:, None], nj, axis=1)[:, :, None]
    y = np.repeat(eta[None, :], ni, axis=0)[:, :, None]

    # shock: true position moves out by 10 cells along the wall
    j_true = 50.3 + 10.0 * np.linspace(0.0, 1.0, ni)
    j_index = np.arange(nj, dtype=float)
    pres = np.zeros((ni, nj, 1))
    for i in range(ni):
        jump = 40.0 * np.exp(-8.0 * wall_x[i]) + 0.5
        post_shock = 1.0 + jump + 0.004 * jump * (j_true[i] - j_index)
        shape = 0.5 * (1.0 - np.tanh((j_index - j_true[i]) / width))
        pres[i, :, 0] = FREESTREAM_PRESSURE * (1.0 + shape * (post_shock - 1.0))

    # velocity: a thin boundary layer at the wall (cfd-io finds the wall by it)
    profile = np.minimum(eta / 0.002, 1.0) * 900.0
    uvel = np.repeat(profile[None, :], ni, axis=0)[:, :, None]
    flow = {
        "pres": Field(data=pres, association="node"),
        "uvel": Field(data=uvel, association="node"),
        "vvel": Field(data=np.zeros_like(uvel), association="node"),
    }
    dataset = Dataset(grid=StructuredGrid(x=x, y=y, z=np.zeros_like(x)), flow=flow, attrs={})

    return dataset, j_true
