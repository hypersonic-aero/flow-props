"""Tests for flow_props.cli.common station selection."""

import numpy as np
import pytest
import typer
from cfd_io.dataset import Dataset, Field, StructuredGrid

from flow_props.cli.common import resolve_station_list
from flow_props.pipeline import _resolve_stations
from flow_props.schema import StationsConfig

# common.resolve_station_list raises typer.Exit; catch that exact class
Exit = typer.Exit


def _make_dataset(ni: int = 8, nj: int = 4) -> Dataset:
    """Build a small structured Dataset with monotonic wall x coordinates."""
    x = np.zeros((ni, nj, 1))
    y = np.zeros((ni, nj, 1))
    for i in range(ni):
        for j in range(nj):
            x[i, j, 0] = 0.1 * float(i)
            y[i, j, 0] = 0.01 * float(j)

    z = np.zeros_like(x)
    grid = StructuredGrid(x=x, y=y, z=z)
    flow = {"uvel": Field(data=np.zeros((ni, nj, 1)), association="node")}
    return Dataset(grid=grid, flow=flow, attrs={"format": "test"})


def test_pipeline_empty_station_selection_uses_all_grid_indices():
    ds = _make_dataset(ni=8)

    stations = _resolve_stations(ds, StationsConfig())

    assert stations == list(range(8))


def test_pipeline_explicit_station_range_still_limits_selection():
    ds = _make_dataset(ni=8)

    stations = _resolve_stations(ds, StationsConfig(i_s=2, i_e=6, di=2))

    assert stations == [2, 4, 6]


def test_resolve_station_list_explicit_values():
    ds = _make_dataset()
    stations = resolve_station_list(ds, cfg={}, stations_text="1,3,5")
    assert stations == [1, 3, 5]


def test_resolve_station_list_i_range():
    ds = _make_dataset()
    stations = resolve_station_list(ds, cfg={}, i_s=1, i_e=5, di=2)
    assert stations == [1, 3, 5]


def test_resolve_station_list_x_range():
    ds = _make_dataset()
    stations = resolve_station_list(ds, cfg={}, x_s=0.1, x_e=0.5, dx=0.2)
    assert stations == [1, 3, 5]


def test_resolve_station_list_reads_config_x_range():
    ds = _make_dataset()
    stations = resolve_station_list(ds, cfg={"x_s": 0.1, "x_e": 0.5, "dx": 0.2})
    assert stations == [1, 3, 5]


def test_resolve_station_list_rejects_multiple_modes():
    ds = _make_dataset()
    with pytest.raises(Exit):
        resolve_station_list(ds, cfg={"stations": [1, 3], "x_s": 0.1, "x_e": 0.5, "dx": 0.2})


def test_resolve_station_list_requires_station_mode():
    ds = _make_dataset()
    with pytest.raises(Exit):
        resolve_station_list(ds, cfg={})
