"""Tests for flow_props.output result writers."""

from pathlib import Path

import numpy as np
import pytest
from cfd_io.dataset import Dataset, Field, StructuredGrid

from flow_props.output import write_profiles_output, write_station_output, write_tecplot_table


def _make_station_dataset(ni: int = 4, nj: int = 6) -> Dataset:
    """Build a minimal structured dataset for output testing."""
    x = np.zeros((ni, nj, 1))
    y = np.zeros((ni, nj, 1))
    for i in range(ni):
        for j in range(nj):
            x[i, j, 0] = 0.1 * float(i)
            y[i, j, 0] = 0.01 * float(j)

    z = np.zeros_like(x)
    grid = StructuredGrid(x=x, y=y, z=z)
    flow = {"uvel": Field(data=np.ones((ni, nj, 1)), association="node")}
    return Dataset(grid=grid, flow=flow, attrs={"format": "test"})


def test_write_profiles_output_dat(tmp_path: Path):
    profiles = [
        {
            "x": np.array([0.0, 0.0, 0.0]),
            "y": np.array([0.0, 0.1, 0.2]),
            "eta": np.array([0.0, 0.1, 0.2]),
            "uvel": np.array([0.0, 0.5, 1.0]),
        },
        {
            "x": np.array([1.0, 1.0, 1.0]),
            "y": np.array([0.0, 0.1, 0.2]),
            "eta": np.array([0.0, 0.1, 0.2]),
            "uvel": np.array([0.0, 0.4, 0.9]),
        },
    ]

    output_path = tmp_path / "profiles.dat"
    write_profiles_output(output_path, profiles, [10, 20])

    text = output_path.read_text()
    assert 'TITLE = "flow-props profiles"' in text
    assert 'VARIABLES = "x", "y", "z", "eta", "station", "uvel"' in text
    assert 'ZONE T="x_0.0000_m_k_00001", I=3, J=1, K=1, F=POINT' in text
    assert 'ZONE T="x_1.0000_m_k_00001", I=3, J=1, K=1, F=POINT' in text


def test_write_station_output_dat(tmp_path: Path):
    ds = _make_station_dataset()
    results = [
        {"station": 1, "delta": 0.01, "theta": 0.002},
        {"station": 3, "delta": 0.02, "theta": 0.003},
    ]

    output_path = tmp_path / "bl.dat"
    write_station_output(
        output_path,
        ds,
        [1, 3],
        results,
        title="flow-props boundary layer",
        zone_title="boundary-layer",
    )

    text = output_path.read_text()
    assert 'TITLE = "flow-props boundary layer"' in text
    assert 'VARIABLES = "x", "y", "z", "station", "delta", "theta"' in text
    assert 'ZONE T="boundary-layer", I=2, J=1, K=1, F=POINT' in text


def test_write_tecplot_table_dat(tmp_path: Path):
    """One I-ordered zone, columns in dict order (x, y first for a line in the plane)."""
    columns = {"x": np.array([0.0, 1.0]), "y": np.array([0.5, 0.6]), "j": np.array([3.2, 3.4])}

    output_path = write_tecplot_table(tmp_path / "shock.dat", columns, title="t", zone_title="z")

    lines = output_path.read_text().splitlines()
    assert lines[1] == 'VARIABLES = "x", "y", "j"'
    assert lines[2] == 'ZONE T = "z", I = 2, F = POINT'
    np.testing.assert_allclose(np.loadtxt(lines[3:]), np.column_stack(list(columns.values())))

    with pytest.raises(ValueError, match="rows"):
        write_tecplot_table(
            tmp_path / "bad.dat", {"x": np.zeros(2), "y": np.zeros(3)}, title="t", zone_title="z"
        )
