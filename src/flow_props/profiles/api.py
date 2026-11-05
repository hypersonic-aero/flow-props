"""Public profile extraction API."""

from __future__ import annotations

import numpy as np
from cfd_io.dataset import Dataset, StructuredGrid

from flow_props.profiles.grid_line import extract_profiles_grid_line
from flow_props.profiles.wall_normal import extract_profiles_wall_normal_interp

# --------------------------------------------------
# profile extraction methods
# --------------------------------------------------

PROFILE_METHOD_GRID_LINE = "grid_line"
PROFILE_METHOD_WALL_NORMAL_INTERP = "wall_normal_interp"


# --------------------------------------------------
# public API
# --------------------------------------------------


def extract_profiles(
    ds: Dataset,
    stations: list[int],
    method: str = PROFILE_METHOD_GRID_LINE,
    n_eta: int | None = None,
    eta_max: float | None = None,
) -> list[dict[str, np.ndarray]]:
    """Extract wall-normal profiles at specified i-stations."""
    # validate inputs
    if not isinstance(ds.grid, StructuredGrid):
        raise TypeError("extract_profiles requires a StructuredGrid")

    # dispatch to the requested extraction method
    if method == PROFILE_METHOD_GRID_LINE:
        return extract_profiles_grid_line(ds, stations)
    if method == PROFILE_METHOD_WALL_NORMAL_INTERP:
        return extract_profiles_wall_normal_interp(ds, stations, n_eta, eta_max)

    raise ValueError(f"unsupported profile extraction method: {method}")
