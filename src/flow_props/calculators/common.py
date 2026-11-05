"""Shared pieces of every calculation: the input files, station selection and gas data.

Each calculation (bl, entropy-layer, profiles, shock) has its own config file. They all
name the CFD solution the same way (``fname``, ``gname``), most select stations with a
``[stations]`` table, and the ones that need gas constants read them the same way. The
pydantic models, the template blocks and the run steps for those shared parts live here.
"""

# --------------------------------------------------
# load necessary modules
# --------------------------------------------------
from __future__ import annotations

import ast
import builtins
import json
import logging
import tomllib
from pathlib import Path
from typing import TypeVar

import numpy as np
from cfd_io.dataset import Dataset, StructuredGrid
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

logger = logging.getLogger(__name__)

# any calculation's config model (load_config returns the model it is given)
ConfigModel = TypeVar("ConfigModel", bound=BaseModel)

# --------------------------------------------------
# config models (input boundary)
# --------------------------------------------------


class CaseConfig(BaseModel):
    """Input shared by every calculation: the CFD solution (and its grid file)."""

    model_config = ConfigDict(extra="forbid")

    fname: str = Field(min_length=1, description="Path to CFD solution file")
    gname: str = Field(default="", description="Grid file for split-format CFD++")


class StationsConfig(BaseModel):
    """Station selection: no mode means all stations; otherwise choose one."""

    model_config = ConfigDict(extra="forbid")

    # -- mode 1: x-range --
    x_s: float | None = Field(default=None, description="Start wall-x [m]")
    x_e: float | None = Field(default=None, description="End wall-x [m]")
    dx: float | None = Field(default=None, gt=0, description="Step in x [m]")

    # -- mode 2: i-range --
    i_s: int | None = Field(default=None, description="Start i-index (inclusive)")
    i_e: int | None = Field(default=None, description="End i-index (inclusive)")
    di: int | None = Field(default=None, gt=0, description="Step in i")

    # -- mode 3: explicit list --
    list: builtins.list[int] | None = Field(default=None, description="Explicit i-station list")

    @model_validator(mode="after")
    def _check_exactly_one_mode(self) -> StationsConfig:
        """Validate that exactly one selection mode is fully specified."""
        # check which modes are (at least partially) filled
        mode1 = any(v is not None for v in [self.x_s, self.x_e, self.dx])
        mode2 = any(v is not None for v in [self.i_s, self.i_e, self.di])
        mode3 = self.list is not None
        active = sum([mode1, mode2, mode3])
        if active > 1:
            raise ValueError("stations: only one selection mode is allowed at a time")
        return self


class GasConfig(BaseModel):
    """Optional gas-property overrides for calculations that require them."""

    model_config = ConfigDict(extra="forbid")

    gamma: float | None = Field(default=None, gt=1.0, description="Specific heat ratio")
    gas_constant: float | None = Field(default=None, gt=0.0, description="Gas constant [J/(kg K)]")


# --------------------------------------------------
# template blocks (every key has a comment line above it)
# --------------------------------------------------

INPUT_BLOCK = """\
#--------------------------------------------------
# input
#--------------------------------------------------
# CFD solution file.
fname = "base_flow.hdf5"
# For split file formats, uncomment and set the separate grid file.
# gname = "grid.s8"
"""

STATIONS_BLOCK = """\
#--------------------------------------------------
# station selection
#--------------------------------------------------
# Leave this table empty to process all i-stations (0 through Nx - 1, step 1).
# To select fewer stations, uncomment exactly one mode below.
[stations]
# Mode 1: sample wall x-coordinates from x_s to x_e, inclusive.
# Start x [m].
# x_s = 0.05
# End x [m].
# x_e = 0.50
# Spacing between requested x-coordinates [m].
# dx  = 0.05

# Mode 2: select i-indices from i_s to i_e, inclusive, with step di.
# i_s = 10
# i_e = 500
# di  = 10

# Mode 3: select only these i-indices.
# list = [10, 50, 100, 200]
"""

GAS_BLOCK = """\
#--------------------------------------------------
# optional gas overrides (otherwise read HDF5 attrs or flow_conditions.json)
#--------------------------------------------------
# [gas]
# Specific heat ratio.
# gamma        = 1.4
# Specific gas constant [J/(kg K)].
# gas_constant = 287.05
"""


def build_template(name: str, *blocks: str) -> str:
    """Join a calculation's header and its blocks into one config template.

    Args:
        name: Calculation name, as typed on the command line (e.g. "bl").
        *blocks: TOML blocks, in the order they appear in the file.

    Returns:
        Template text, ending in one newline.
    """

    # header: what the file is and how to run it
    header = (
        "#--------------------------------------------------\n"
        f"# flow-props {name} configuration\n"
        f"# generated by: flow-props {name} init\n"
        f"# run with:     flow-props {name} run\n"
        "#--------------------------------------------------\n"
    )

    # build: header first, blocks separated by one blank line
    parts = [header.strip()]
    for block in blocks:
        parts.append(block.strip())
    template = "\n\n".join(parts) + "\n"

    return template


# --------------------------------------------------
# public API: run steps shared by the calculations
# --------------------------------------------------


def load_config(config_path: Path, model: type[ConfigModel]) -> ConfigModel:
    """Load and validate a calculation's TOML config file.

    Args:
        config_path: Path to the TOML config file.
        model: Pydantic model of the calculation's config.

    Returns:
        Validated config.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file isn't valid TOML or fails validation (one line naming
            every bad key).
    """
    # validate path
    if not config_path.is_file():
        raise FileNotFoundError(f"config file not found: {config_path}")

    # read TOML
    with open(config_path, "rb") as f:
        data = tomllib.load(f)

    # build and validate, reporting every problem on one line
    try:
        config = model.model_validate(data)
    except ValidationError as error:
        problems = []
        for detail in error.errors():
            location = ".".join(str(part) for part in detail["loc"])
            problems.append(f"{location}: {detail['msg']}" if location else detail["msg"])
        raise ValueError(f"{config_path}: {'; '.join(problems)}") from None

    return config


def load_dataset(fname: Path, gname: Path | None) -> Dataset:
    """Load a CFD dataset, passing a grid file for split formats.

    Args:
        fname: Path to the primary solution file.
        gname: Path to the grid file for split formats, or ``None``.

    Returns:
        Loaded Dataset.

    Raises:
        FileNotFoundError: If the solution file does not exist.
    """
    from cfd_io import read_file

    # validate the primary input path
    if not fname.is_file():
        raise FileNotFoundError(f"input file not found: {fname}")

    # debug output for devs
    logger.debug("loading dataset: %s (grid file: %s)", fname, gname)

    # dispatch by whether a grid file is configured
    if gname is not None:
        return read_file(fname, grid_file=gname)
    return read_file(fname)


def input_paths(config: CaseConfig, config_dir: Path) -> tuple[Path, Path | None]:
    """Resolve the solution and grid file paths relative to the config file's folder.

    Args:
        config: Validated config of any calculation.
        config_dir: Directory containing the config file.

    Returns:
        Solution file path, and the grid file path (``None`` without one).
    """

    # -- resolve file paths relative to config directory --
    fname = config_dir / config.fname
    gname = (config_dir / config.gname) if config.gname else None

    return fname, gname


def resolve_stations(ds: Dataset, stations_cfg: StationsConfig) -> list[int]:
    """Resolve the configured station selection into a list of i-indices.

    Args:
        ds: Dataset with a StructuredGrid.
        stations_cfg: Validated StationsConfig instance.

    Returns:
        List of i-station indices.

    Raises:
        TypeError: If the dataset is not structured.
        ValueError: If the selection is incomplete or out of range.
    """
    # station selection requires a structured grid
    if not isinstance(ds.grid, StructuredGrid):
        raise TypeError("station selection requires a StructuredGrid dataset")

    ni = ds.grid.shape[0]

    # mode 3: explicit list
    if stations_cfg.list is not None:
        return _validate_indices(ni, stations_cfg.list)

    # mode 2: i-index range
    if stations_cfg.i_s is not None or stations_cfg.i_e is not None or stations_cfg.di is not None:
        return _resolve_i_range(ni, stations_cfg.i_s, stations_cfg.i_e, stations_cfg.di)

    # mode 1: x-range along the wall surface
    if stations_cfg.x_s is not None or stations_cfg.x_e is not None or stations_cfg.dx is not None:
        return _resolve_x_range(ds, stations_cfg.x_s, stations_cfg.x_e, stations_cfg.dx)

    # without an explicit selection, process every streamwise grid station
    return list(range(ni))


def resolve_gas_properties(
    gas_cfg: GasConfig, attrs: dict, fname: Path, *, required: bool, needed_for: str
) -> tuple[float, float]:
    """Resolve gas constants for calculations that need them.

    The values come from the [gas] overrides first, then the dataset's HDF5 attributes
    (``gamma``, ``r_gas``), then a ``flow_conditions.json`` next to the solution file.

    Args:
        gas_cfg: Validated [gas] table.
        attrs: Dataset attributes.
        fname: Solution file (its folder may hold flow_conditions.json).
        required: Whether the calculation needs the values at all; without them it
            gets air's (1.4, 287.05), which it doesn't use.
        needed_for: What needs them, for the error message (e.g. "the entropy layer").

    Returns:
        Specific heat ratio and gas constant [J/(kg K)].

    Raises:
        ValueError: If a required value is missing everywhere, or out of range.
    """
    if not required:
        return 1.4, 287.05

    # read HDF5 attributes first; the sibling flow-conditions file fills gaps
    values = {
        "gamma": _metadata_number(attrs.get("gamma")),
        "gas_constant": _metadata_number(attrs.get("r_gas")),
    }
    conditions_path = fname.parent / "flow_conditions.json"
    if None in values.values() and conditions_path.is_file():
        with conditions_path.open(encoding="utf-8") as stream:
            conditions = json.load(stream)
        for key, source_key in (("gamma", "gamma"), ("gas_constant", "r_gas")):
            if values[key] is None:
                values[key] = _metadata_number(conditions.get(source_key))

    gamma = gas_cfg.gamma if gas_cfg.gamma is not None else values["gamma"]
    gas_constant = (
        gas_cfg.gas_constant if gas_cfg.gas_constant is not None else values["gas_constant"]
    )
    if gamma is None or gas_constant is None:
        raise ValueError(
            f"{needed_for} needs gamma and r_gas: set them in the HDF5 attributes or "
            "flow_conditions.json, or as [gas] overrides"
        )
    if gamma <= 1.0 or gas_constant <= 0.0:
        raise ValueError("gamma must exceed 1 and gas constant must be positive")

    # debug output for devs
    logger.debug("gas: gamma = %g, gas constant = %g J/(kg K)", gamma, gas_constant)

    return gamma, gas_constant


# --------------------------------------------------
# helpers
# --------------------------------------------------


def _metadata_number(value: object) -> float | None:
    """Read a scalar or unit-tagged value from cfd-io attributes or JSON."""
    if value is None:
        return None
    if isinstance(value, str):
        value = ast.literal_eval(value)
    if isinstance(value, (list, tuple)):
        value = value[0]
    return float(value)


def _validate_indices(ni: int, indices: list[int]) -> list[int]:
    """Validate explicit i-station indices against the grid bounds."""
    # bounds-check each requested station
    for station in indices:
        if station < 0 or station >= ni:
            raise ValueError(f"station i={station} out of range [0, {ni})")
    return list(indices)


def _resolve_i_range(ni: int, i_s: int | None, i_e: int | None, di: int | None) -> list[int]:
    """Build a station list from an inclusive i-index range, clamped to the grid."""
    # require both endpoints
    if i_s is None or i_e is None:
        raise ValueError("i-range selection requires both i_s and i_e")

    step = 1 if di is None else di

    # clamp the inclusive end to the last valid index
    start = max(0, i_s)
    end = min(ni - 1, i_e)
    if start > end:
        raise ValueError(f"i-range selection produced no stations in [0, {ni})")

    stations = list(range(start, end + 1, step))
    if not stations:
        raise ValueError("i-range selection produced no stations")
    return stations


def _resolve_x_range(
    ds: Dataset, x_s: float | None, x_e: float | None, dx: float | None
) -> list[int]:
    """Map a wall x-range to the nearest i-station indices along the wall line."""
    # require both endpoints
    if x_s is None or x_e is None:
        raise ValueError("x-range selection requires both x_s and x_e")

    # read wall x-coordinates along the first wall line (j=0, k=0)
    wall_x = np.asarray(ds.grid.x[:, 0, 0], dtype=float)
    ni = ds.grid.shape[0]

    # build the inclusive target x values
    step = 0.0 if dx is None else dx
    targets = _build_targets(float(x_s), float(x_e), float(step))

    # map each target to the nearest wall i-index, preserving order and uniqueness
    stations: list[int] = []
    for target in targets:
        idx = int(np.argmin(np.abs(wall_x - target)))
        if 0 <= idx < ni and idx not in stations:
            stations.append(idx)

    if not stations:
        raise ValueError("x-range selection produced no stations")
    return stations


def _build_targets(x_s: float, x_e: float, dx: float) -> list[float]:
    """Build inclusive, evenly spaced target x-values."""
    # single-target case
    if dx <= 0.0 or x_s == x_e:
        return [x_s]

    direction = 1.0 if x_e >= x_s else -1.0
    step = direction * dx
    tolerance = abs(dx) * 1.0e-9 + 1.0e-12

    targets: list[float] = []
    current = x_s
    while (direction > 0.0 and current <= x_e + tolerance) or (
        direction < 0.0 and current >= x_e - tolerance
    ):
        targets.append(float(current))
        current += step
    return targets
