"""The profiles calculation: wall-normal profiles of every flow variable per station.

``flow-props profiles init`` writes the template below, ``flow-props profiles run``
reads it and writes the profiles at the selected stations as a multi-zone Tecplot ASCII
file (one zone per station), e.g. to check a boundary-layer edge or to plot profiles.
"""

# --------------------------------------------------
# load necessary modules
# --------------------------------------------------
from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from flow_props.calculators.common import (
    INPUT_BLOCK,
    STATIONS_BLOCK,
    CaseConfig,
    StationsConfig,
    build_template,
    input_paths,
    load_dataset,
    resolve_stations,
)

logger = logging.getLogger(__name__)

# --------------------------------------------------
# config models (input boundary)
# --------------------------------------------------


class ProfilesConfig(BaseModel):
    """Profile extraction options."""

    model_config = ConfigDict(extra="forbid")

    method: Literal["grid_line", "wall_normal_interp"] = Field(
        default="grid_line",
        description="Along the j grid lines, or along straight wall normals (interpolated)",
    )
    n_eta: int | None = Field(
        default=None,
        ge=2,
        description="wall_normal_interp: points per profile; default: the grid line's count",
    )
    eta_max: float | None = Field(
        default=None,
        gt=0.0,
        description="wall_normal_interp: profile length [m]; default: the grid line's length",
    )
    output: str = Field(
        default="profiles.dat",
        description="Output Tecplot ASCII file path",
    )


class ProfilesCaseConfig(CaseConfig):
    """Config file of the profiles calculation."""

    stations: StationsConfig = Field(default_factory=StationsConfig)
    profiles: ProfilesConfig = Field(default_factory=ProfilesConfig)


# --------------------------------------------------
# template (written by: flow-props profiles init)
# --------------------------------------------------

PROFILES_BLOCK = """\
#--------------------------------------------------
# profile extraction
#--------------------------------------------------
[profiles]
# How profiles are taken: grid_line (along the j grid lines) or wall_normal_interp
# (along straight wall normals, interpolated from the grid).
method = "grid_line"
# wall_normal_interp only: points per profile; uncomment to override the grid line's count.
# n_eta = 200
# wall_normal_interp only: profile length [m]; uncomment to override the grid line's length.
# eta_max = 0.01
# Tecplot ASCII file with one zone per station.
output = "profiles.dat"
"""

TEMPLATE = build_template("profiles", INPUT_BLOCK, STATIONS_BLOCK, PROFILES_BLOCK)


# --------------------------------------------------
# public API
# --------------------------------------------------


def run_case(cfg: ProfilesCaseConfig, config_dir: Path) -> list[str]:
    """Extract the profiles at the configured stations and write them.

    Args:
        cfg: Validated ProfilesCaseConfig instance.
        config_dir: Directory containing the config file (used to resolve
            relative input/output paths).

    Returns:
        Report lines for the command line.
    """
    from flow_props.output import write_profiles_output
    from flow_props.profiles import extract_profiles

    # resolve file paths relative to the config directory, then load the dataset
    fname, gname = input_paths(cfg, config_dir)
    ds = load_dataset(fname, gname)

    # resolve the stations
    stations = resolve_stations(ds, cfg.stations)

    # debug output for devs
    logger.debug("extracting %d profiles (%s)", len(stations), cfg.profiles.method)

    # extract
    options = cfg.profiles
    profiles = extract_profiles(
        ds, stations, method=options.method, n_eta=options.n_eta, eta_max=options.eta_max
    )

    # write
    output_path = config_dir / options.output
    write_profiles_output(output_path, profiles, stations)

    # report
    report = [
        f"Profiles at {len(stations)} stations ({options.method})",
        f"Wrote: {output_path}",
    ]

    return report
