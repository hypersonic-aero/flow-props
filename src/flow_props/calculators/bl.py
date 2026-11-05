"""The bl calculation: boundary-layer edge and integral thicknesses per station.

``flow-props bl init`` writes the template below, ``flow-props bl run`` reads it,
extracts the wall-normal grid-line profiles at the selected stations and writes the
boundary-layer property table.
"""

# --------------------------------------------------
# load necessary modules
# --------------------------------------------------
from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from flow_props.calculators.common import (
    GAS_BLOCK,
    INPUT_BLOCK,
    STATIONS_BLOCK,
    CaseConfig,
    GasConfig,
    StationsConfig,
    build_template,
    input_paths,
    load_dataset,
    resolve_gas_properties,
    resolve_stations,
)

logger = logging.getLogger(__name__)

# --------------------------------------------------
# config models (input boundary)
# --------------------------------------------------


class BLConfig(BaseModel):
    """Boundary-layer extraction options."""

    model_config = ConfigDict(extra="forbid")

    criterion: Literal["velocity_ratio", "htot_derivative"] = Field(
        default="velocity_ratio",
        description="BL edge detection criterion",
    )
    threshold: float = Field(
        default=0.99,
        gt=0.0,
        le=1.0,
        description="Velocity ratio threshold (velocity_ratio criterion only)",
    )
    output: str = Field(
        default="bl_properties.dat",
        description="Output Tecplot ASCII file path",
    )


class BLCaseConfig(CaseConfig):
    """Config file of the bl calculation."""

    stations: StationsConfig = Field(default_factory=StationsConfig)
    bl: BLConfig = Field(default_factory=BLConfig)
    gas: GasConfig = Field(default_factory=GasConfig)


# --------------------------------------------------
# template (written by: flow-props bl init)
# --------------------------------------------------

BL_BLOCK = """\
#--------------------------------------------------
# boundary layer extraction
#--------------------------------------------------
[bl]
# Edge criterion: velocity_ratio or htot_derivative.
criterion = "velocity_ratio"
# Fraction of edge velocity for velocity_ratio.
threshold = 0.99
# Tecplot ASCII BL property table.
output = "bl_properties.dat"
"""

TEMPLATE = build_template("bl", INPUT_BLOCK, STATIONS_BLOCK, BL_BLOCK, GAS_BLOCK)


# --------------------------------------------------
# public API
# --------------------------------------------------


def run_case(cfg: BLCaseConfig, config_dir: Path) -> list[str]:
    """Run the BL extraction pipeline.

    Args:
        cfg: Validated BLCaseConfig instance.
        config_dir: Directory containing the config file (used to resolve
                    relative input/output paths).

    Returns:
        Report lines for the command line.
    """
    from flow_props.output import write_bl_properties
    from flow_props.profiles import extract_profiles

    # -- resolve file paths relative to config directory --
    fname, gname = input_paths(cfg, config_dir)

    # -- load CFD dataset via cfd_io --
    ds = load_dataset(fname, gname)

    # -- resolve station list --
    stations = resolve_stations(ds, cfg.stations)

    # debug output for devs
    logger.debug("processing %d stations", len(stations))

    # -- extract wall-normal profiles --
    profiles = extract_profiles(ds, stations, method="grid_line")

    # -- compute BL properties per station --
    gamma, gas_constant = resolve_gas_properties(
        cfg.gas,
        ds.attrs,
        fname,
        required=cfg.bl.criterion == "htot_derivative",
        needed_for="the htot_derivative criterion",
    )
    table = _compute_bl_table(profiles, cfg.bl, gamma, gas_constant)

    # -- write output --
    output_path = config_dir / cfg.bl.output
    write_bl_properties(table, output_path)

    # debug output for devs
    logger.debug("written: %s", output_path)

    # report
    report = [
        f"Boundary layer at {len(stations)} stations ({cfg.bl.criterion} edge criterion)",
        f"Wrote: {output_path}",
    ]

    return report


# --------------------------------------------------
# helpers
# --------------------------------------------------


def _compute_bl_table(
    profiles: list[dict[str, np.ndarray]], bl_cfg: BLConfig, gamma: float, gas_constant: float
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
