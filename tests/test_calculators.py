"""Tests for the calculations' configs: templates, examples, loading and the shared steps."""

from importlib.resources import files
from pathlib import Path

import numpy as np
import pytest
from cfd_io.dataset import Dataset, Field, StructuredGrid

from flow_props.calculators import CALCULATORS
from flow_props.calculators.common import (
    AT_CELL_CENTRES,
    AT_VERTICES,
    GasConfig,
    StationsConfig,
    load_config,
    resolve_gas_properties,
    resolve_stations,
    solution_points,
)
from flow_props.cli.examples_command import list_examples


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


# --------------------------------------------------
# templates and examples
# --------------------------------------------------
@pytest.mark.parametrize("name", list(CALCULATORS))
def test_template_validates(name, tmp_path):
    """What init writes is a valid config of its calculation."""
    path = tmp_path / f"{name}.toml"
    path.write_text(CALCULATORS[name].template, encoding="utf-8")
    load_config(path, CALCULATORS[name].config_model)


@pytest.mark.parametrize("name", list(CALCULATORS))
def test_every_template_key_has_a_comment_above(name):
    """Every key (set or commented out) is documented on the line above it."""
    lines = CALCULATORS[name].template.splitlines()
    for index, line in enumerate(lines):
        # a key line: "key = value", possibly commented out ("# key = value")
        text = line.lstrip("# ").strip()
        is_key = (
            "=" in text and not text.startswith("[") and text.split("=")[0].strip().isidentifier()
        )
        if not is_key or line.strip().startswith("# Mode"):
            continue
        # the line above is a comment (description or another commented-out key of a group)
        assert lines[index - 1].strip().startswith("#"), f"{name}: no comment above {line!r}"


@pytest.mark.parametrize("name", list(CALCULATORS))
def test_bundled_examples_validate(name):
    """Every bundled example config is a valid config of its calculation."""
    for example in list_examples(name):
        resource = files("flow_props") / "examples" / name / example / f"{name}.toml"
        with resource.open("rb"):
            pass
        load_config(Path(str(resource)), CALCULATORS[name].config_model)


def test_load_config_reports_bad_keys_on_one_line(tmp_path):
    """A misspelled key and a bad value both show up in one error line."""
    path = tmp_path / "shock.toml"
    path.write_text('fname = "a.h5"\n[shock]\nlevle = 0.5\nlevel = 2.0\n', encoding="utf-8")
    with pytest.raises(ValueError) as error:
        load_config(path, CALCULATORS["shock"].config_model)
    message = str(error.value)
    assert "\n" not in message
    assert "shock.levle" in message
    assert "shock.level" in message


def test_load_config_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.toml", CALCULATORS["bl"].config_model)


# --------------------------------------------------
# shared run steps
# --------------------------------------------------
def test_empty_station_selection_uses_all_grid_indices():
    ds = _make_dataset(ni=8)

    stations = resolve_stations(ds, StationsConfig())

    assert stations == list(range(8))


def test_explicit_station_range_still_limits_selection():
    ds = _make_dataset(ni=8)

    stations = resolve_stations(ds, StationsConfig(i_s=2, i_e=6, di=2))

    assert stations == [2, 4, 6]


def test_x_range_selection_maps_to_nearest_stations():
    ds = _make_dataset(ni=8)

    stations = resolve_stations(ds, StationsConfig(x_s=0.1, x_e=0.5, dx=0.2))

    assert stations == [1, 3, 5]


def test_station_modes_are_exclusive():
    with pytest.raises(ValueError, match="only one selection mode"):
        StationsConfig(list=[1, 3], x_s=0.1, x_e=0.5, dx=0.2)


def test_gas_metadata_prefers_overrides_then_hdf5_then_json(tmp_path):
    """Unit-tagged HDF5 attributes and flow-conditions JSON supply gas values."""
    input_path = tmp_path / "base_flow.hdf5"
    (tmp_path / "flow_conditions.json").write_text(
        '{"gamma": [1.3, "-"], "r_gas": [300.0, "J/(kg*K)"]}',
        encoding="utf-8",
    )

    gamma, gas_constant = resolve_gas_properties(
        GasConfig(), {"gamma": "[1.4, '-']"}, input_path, required=True, needed_for="test"
    )
    assert (gamma, gas_constant) == (1.4, 300.0)

    gamma, gas_constant = resolve_gas_properties(
        GasConfig(gas_constant=287.05), {}, input_path, required=True, needed_for="test"
    )
    assert (gamma, gas_constant) == (1.3, 287.05)


def test_gas_metadata_missing_names_who_needs_it(tmp_path):
    with pytest.raises(ValueError, match="the entropy layer needs gamma"):
        resolve_gas_properties(
            GasConfig(), {}, tmp_path / "a.h5", required=True, needed_for="the entropy layer"
        )


# --------------------------------------------------
# solutions at the vertices or at the cell centres
# --------------------------------------------------
def test_vertex_solution_is_used_as_it_is():
    ds = _make_dataset()

    points, location = solution_points(ds)

    assert points is ds
    assert location == AT_VERTICES


def test_cell_solution_moves_to_a_grid_of_cell_centres():
    """Cell values on a vertex grid: the grid becomes the cell centres (corner means)."""
    ds = _make_dataset(ni=8, nj=4)
    cell_values = np.arange(7 * 3, dtype=float).reshape(7, 3, 1)
    ds.flow = {"pres": Field(data=cell_values, association="cell")}

    points, location = solution_points(ds)

    assert location == AT_CELL_CENTRES
    assert points.grid.shape == (7, 3, 1)
    np.testing.assert_allclose(points.grid.x[:, 0, 0], 0.1 * (np.arange(7) + 0.5))
    np.testing.assert_allclose(points.grid.y[0, :, 0], 0.01 * (np.arange(3) + 0.5))
    np.testing.assert_array_equal(points.flow["pres"].data, cell_values)
    assert points.flow["pres"].association == "node"


def test_fields_in_different_places_are_refused():
    ds = _make_dataset(ni=8, nj=4)
    ds.flow["pres"] = Field(data=np.zeros((7, 3, 1)), association="cell")

    with pytest.raises(ValueError, match="different places"):
        solution_points(ds)
