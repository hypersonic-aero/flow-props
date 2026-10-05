"""Main BL extraction pipeline: load dataset -> extract profiles -> compute BL properties."""

from __future__ import annotations

import ast
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from cfd_io.dataset import Dataset, StructuredGrid

log = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Output of a complete pipeline run."""

    # number of stations processed
    n_stations: int
    # path to the written output file
    output_path: Path
    # table of per-station results (dict of 1-D numpy arrays, one entry per column)
    table: dict[str, np.ndarray]


def run_pipeline(cfg, config_dir: Path, mode: str = "all") -> PipelineResult:
    """Run the BL extraction pipeline.

    Args:
        cfg: Validated FlowPropsConfig instance.
        config_dir: Directory containing the config file (used to resolve
                    relative input/output paths).
        mode: ``"bl"`` runs BL extraction only; ``"all"`` runs all sections
              present in the config.  Currently only ``"bl"`` is implemented;
              ``"all"`` behaves identically until entropy/wall sections are added.

    Returns:
        PipelineResult with output path and result table.
    """
    from flow_props.output import write_bl_properties, write_profiles_output
    from flow_props.profiles import extract_profiles

    # -- resolve file paths relative to config directory --
    fname = config_dir / cfg.fname
    gname = (config_dir / cfg.gname) if cfg.gname else None

    log.info("loading dataset: %s", fname)

    # -- load CFD dataset via cfd_io --
    ds = _load_dataset(fname, gname)

    # -- resolve station list --
    stations = _resolve_stations(ds, cfg.stations)
    log.info("processing %d stations", len(stations))

    # -- extract wall-normal profiles --
    profiles = extract_profiles(ds, stations, method="grid_line")
    if cfg.bl.profiles_output:
        write_profiles_output(config_dir / cfg.bl.profiles_output, profiles, stations)

    # -- compute BL properties per station --
    gamma, gas_constant = _resolve_gas_properties(
        cfg.gas, ds.attrs, fname, required=cfg.bl.criterion == "htot_derivative"
    )
    table = _compute_bl_table(profiles, cfg.bl, gamma, gas_constant)

    # -- write output --
    output_path = config_dir / cfg.bl.output
    write_bl_properties(table, output_path)

    log.info("written: %s", output_path)
    return PipelineResult(n_stations=len(stations), output_path=output_path, table=table)


# --------------------------------------------------
# helpers
# --------------------------------------------------


def _resolve_gas_properties(
    gas_cfg, attrs: dict, fname: Path, *, required: bool
) -> tuple[float, float]:
    """Resolve gas constants for criteria that need total enthalpy."""
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
            "htot_derivative requires gamma and r_gas in HDF5 attributes or "
            "flow_conditions.json, or explicit [gas] overrides"
        )
    if gamma <= 1.0 or gas_constant <= 0.0:
        raise ValueError("gamma must exceed 1 and gas constant must be positive")
    return gamma, gas_constant


def _metadata_number(value: object) -> float | None:
    """Read a scalar or unit-tagged value from cfd-io attributes or JSON."""
    if value is None:
        return None
    if isinstance(value, str):
        value = ast.literal_eval(value)
    if isinstance(value, (list, tuple)):
        value = value[0]
    return float(value)


def _load_dataset(fname: Path, gname: Path | None) -> Dataset:
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

    # dispatch by whether a grid file is configured
    if gname is not None:
        return read_file(fname, grid_file=gname)
    return read_file(fname)


def _resolve_stations(ds: Dataset, stations_cfg) -> list[int]:
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


def _compute_bl_table(
    profiles: list[dict[str, np.ndarray]], bl_cfg, gamma: float, gas_constant: float
) -> dict[str, np.ndarray]:
    """Compute the per-station BL property table.

    Args:
        profiles: Extracted wall-normal profiles in station order.
        bl_cfg: Validated BLConfig instance.
        gamma: Heat capacity ratio for the enthalpy criterion.
        gas_constant: Specific gas constant for the enthalpy criterion.

    Returns:
        Dict mapping each output column name to a 1-D array over stations.
    """
    from flow_props.bl import (
        boundary_layer_edge,
        displacement_thickness,
        momentum_thickness,
        shape_factor,
    )

    # accumulate one value per station for every output column
    columns: dict[str, list[float]] = {
        name: []
        for name in (
            "x",
            "delta",
            "delta_star",
            "theta",
            "H",
            "uvel_edge",
            "temp_edge",
            "dens_edge",
            "mach_edge",
            "twte",
        )
    }

    for profile in profiles:
        eta = np.asarray(profile["eta"], dtype=float)
        u = _profile_field(profile, ("u_tangent", "uvel", "u", "velocity"))
        rho = _profile_field(profile, ("dens", "density", "rho"))
        temp = _profile_field(profile, ("temp", "temperature"))
        mach = _profile_field(profile, ("mach",), optional=True)

        # edge state from the outermost profile point
        u_edge = float(u[-1])
        temp_edge = float(temp[-1])
        dens_edge = float(rho[-1])
        mach_edge = float(mach[-1]) if mach is not None else float("nan")

        # boundary-layer edge thickness from the configured criterion
        delta = boundary_layer_edge(
            eta,
            u,
            criterion=bl_cfg.criterion,
            threshold=bl_cfg.threshold,
            u_edge=u_edge,
            temp=temp,
            gamma=gamma,
            gas_constant=gas_constant,
        )

        # integrate integral thicknesses across the boundary layer only (0 to delta);
        # integrating over the full grid line accumulates spurious freestream
        # contributions over the large inviscid extent and can flip the sign.
        mask = eta <= delta
        if int(np.count_nonzero(mask)) < 2:
            mask = np.zeros_like(eta, dtype=bool)
            mask[:2] = True
        eta_bl, rho_bl, u_bl = eta[mask], rho[mask], u[mask]

        # compressible integral thicknesses and shape factor
        delta_star = displacement_thickness(eta_bl, rho_bl, u_bl, dens_edge, u_edge)
        theta = momentum_thickness(eta_bl, rho_bl, u_bl, dens_edge, u_edge)
        h_shape = shape_factor(delta_star, theta)

        # wall-to-edge temperature ratio from the first profile point
        twte = float(temp[0]) / temp_edge if temp_edge != 0.0 else float("nan")

        columns["x"].append(float(np.asarray(profile["x"]).flat[0]))
        columns["delta"].append(delta)
        columns["delta_star"].append(delta_star)
        columns["theta"].append(theta)
        columns["H"].append(h_shape)
        columns["uvel_edge"].append(u_edge)
        columns["temp_edge"].append(temp_edge)
        columns["dens_edge"].append(dens_edge)
        columns["mach_edge"].append(mach_edge)
        columns["twte"].append(twte)

    return {name: np.asarray(values, dtype=float) for name, values in columns.items()}


def _profile_field(
    profile: dict[str, np.ndarray],
    candidates: tuple[str, ...],
    optional: bool = False,
) -> np.ndarray | None:
    """Return the first matching profile field as a float array."""
    # search candidate names in order
    for name in candidates:
        if name in profile:
            return np.asarray(profile[name], dtype=float)

    # optional fields may be absent
    if optional:
        return None

    names = ", ".join(candidates)
    raise KeyError(f"none of the profile fields were found: {names}")
