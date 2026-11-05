"""The shock calculation: the shock curve of a CFD solution, and how it sits on the grid.

``flow-props shock init`` writes the template below, ``flow-props shock run`` reads it,
finds the shock on every wall-normal grid line (see flow_props.shock) and writes:

- the shock curve as an x, y CSV, the file grid-generator's ``[shock] curve`` reads, so
  a precursor solution's shock can become the next grid's shock line (smoothed along the
  wall first, unless ``smoothing_length = 0``, since it becomes a grid line);
- a Tecplot ASCII report whose x, y are the shock points (load it over the flow field),
  with per line the wall point, distance from the wall, captured thickness in cells and,
  on a shock-line grid, the offset from the grid line meant to hold the shock.

Detection (``method``): walk (the first rise of min_rise over freestream, from outside),
indicator (the first point whose WENO smoothness indicator reaches indicator_threshold)
or hybrid (the outermost jump either finds); the point is where the quantity crosses
`level` of the jump, whichever detector found it (see flow_props.shock).
"""

# --------------------------------------------------
# load necessary modules
# --------------------------------------------------
from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator

from flow_props.calculators.common import (
    INPUT_BLOCK,
    CaseConfig,
    build_template,
    input_paths,
    load_dataset,
)

logger = logging.getLogger(__name__)

# --------------------------------------------------
# constants
# --------------------------------------------------

# within a quarter cell of its grid line, the shock counts as sitting on it
ALIGNED_OFFSET_CELLS = 0.25

# --------------------------------------------------
# config models (input boundary)
# --------------------------------------------------


class ShockConfig(BaseModel):
    """Shock extraction options."""

    model_config = ConfigDict(extra="forbid")

    quantity: Literal["pressure", "density"] = Field(
        default="pressure",
        description="Flow quantity whose jump marks the shock",
    )
    level: float = Field(
        default=0.5,
        gt=0.0,
        lt=1.0,
        description="Where in the jump the shock point sits (0.5 = middle)",
    )
    method: Literal["walk", "indicator", "hybrid"] = Field(
        default="walk",
        description="Detection of the jump: threshold walk, smoothness indicator, or both",
    )
    min_rise: float = Field(
        default=0.05,
        gt=0.0,
        description="walk: rise over the freestream value that marks the shock (0.05 = 5 %)",
    )
    indicator_threshold: float = Field(
        default=0.0025,
        gt=0.0,
        description="indicator: smoothness indicator value that marks the shock",
    )
    max_thickness: float = Field(
        default=20.0,
        gt=0.0,
        description="Largest captured shock thickness in cells (10-90 % of the jump)",
    )
    shock_line_j: int | None = Field(
        default=None,
        ge=1,
        description="Grid line meant to hold the shock; default: the solution's attributes",
    )
    smoothing_length: float = Field(
        default=0.5,
        ge=0.0,
        description="Length of each smoothing fit, in local wall-to-shock distances; 0: none",
    )
    output: str = Field(
        default="shock_cfd.csv",
        description="Shock curve, x, y CSV",
    )
    report: str = Field(
        default="shock_report.dat",
        description="Per-line report, Tecplot ASCII",
    )

    @field_validator("output")
    @classmethod
    def _check_csv(cls, value: str) -> str:
        """The curve is a CSV (what grid-generator's [shock] curve reads)."""
        if not value.lower().endswith(".csv"):
            raise ValueError(
                f"the shock curve is the CSV grid-generator reads; use a .csv name, not {value!r} "
                "(the Tecplot file of the shock is [shock] report)"
            )
        return value

    @field_validator("report")
    @classmethod
    def _check_dat(cls, value: str) -> str:
        """The report is a Tecplot ASCII table."""
        if not value.lower().endswith(".dat"):
            raise ValueError(f"the report is Tecplot ASCII; use a .dat name, not {value!r}")
        return value


class ShockCaseConfig(CaseConfig):
    """Config file of the shock calculation."""

    shock: ShockConfig = Field(default_factory=ShockConfig)


# --------------------------------------------------
# template (written by: flow-props shock init)
# --------------------------------------------------

SHOCK_BLOCK = """\
#--------------------------------------------------
# shock extraction (every wall-normal grid line; the outer boundary must be freestream)
# a detector finds the jump on each line, walking in from its outer end (see method);
# the shock point is where the quantity crosses `level` of the jump, interpolated
# between the two grid points around the crossing, whichever detector found it
#--------------------------------------------------
[shock]
# Flow quantity whose jump marks the shock: pressure or density.
quantity = "pressure"
# Where in the captured jump the shock point sits, from the freestream value (0) to the
# post-shock value (1); 0.5 is the middle of the jump, where a grid line should sit.
level = 0.5
# Detection of the jump: walk (the first rise of min_rise over freestream), indicator
# (the first point whose smoothness indicator reaches indicator_threshold) or hybrid
# (the outermost jump either finds; the report says which one found each line).
method = "walk"
# walk: rise over the freestream value that marks the shock (0.05 = 5 % above freestream).
min_rise = 0.05
# indicator: smoothness indicator value that marks the shock. The indicator is the central
# five-point smoothness measure of ninth-order WENO divided by 5040 q_inf^2: on a smooth
# ramp it is (rise per cell / q_inf)^2, so 0.0025 is a rise of 5 % of freestream per cell;
# sharper jumps score higher, weak smeared ones lower (lower it if lines are missed).
indicator_threshold = 0.0025
# Largest captured shock thickness in cells (10-90 % of the jump); a wider rise is a
# smooth compression, not a shock. Raise it for solvers that smear the shock more.
max_thickness = 20.0
# Shock-line grids: the grid line meant to hold the shock (grid-generator's shock_line_j);
# uncomment to compare with it, if the solution file doesn't carry it as an attribute.
# shock_line_j = 160
# Smoothing of the curve that is written (it becomes a grid line): each point moves, along
# its grid line, onto a quadratic fitted through the shock points within a stretch of the
# shock this many local wall-to-shock distances long (tight at a blunt nose, long far
# downstream, as the shock bends). 0 writes the points as found.
smoothing_length = 0.5
# Shock curve as an x, y CSV, for grid-generator's [shock] curve (smoothed, if set above).
output = "shock_cfd.csv"
# Tecplot ASCII of the shock location: x, y are the shock points as found (load it over
# the flow field), with x_smooth, y_smooth and per-line details; one row per grid line.
report = "shock_report.dat"
"""

TEMPLATE = build_template("shock", INPUT_BLOCK, SHOCK_BLOCK)


# --------------------------------------------------
# public API
# --------------------------------------------------


def run_case(cfg: ShockCaseConfig, config_dir: Path) -> list[str]:
    """Extract the shock, compare it with the grid's shock line, and write curve and report.

    Args:
        cfg: Validated ShockCaseConfig instance.
        config_dir: Directory containing the config file (used to resolve
            relative input/output paths).

    Returns:
        Report lines for the command line.

    Raises:
        ValueError: If no grid line has a shock, or shock_line_j lies outside the grid.
    """
    from flow_props.output import write_tecplot_table
    from flow_props.shock import (
        DETECTED_BY_INDICATOR,
        DETECTED_BY_WALK,
        extract_shock,
        smooth_shock,
        write_shock_curve,
    )

    # resolve file paths relative to the config directory, then load the dataset
    fname, gname = input_paths(cfg, config_dir)
    ds = load_dataset(fname, gname)
    options = cfg.shock

    # extract: one shock point per grid line
    extraction = extract_shock(
        ds,
        quantity=options.quantity,
        level=options.level,
        min_rise=options.min_rise,
        max_thickness=options.max_thickness,
        method=options.method,
        indicator_threshold=options.indicator_threshold,
    )
    ni = ds.grid.shape[0]
    nj = ds.grid.shape[1]

    # the grid line meant to hold the shock: config first, then the solution's attributes
    shock_line_j = options.shock_line_j
    source = "[shock] shock_line_j"
    if shock_line_j is None and ds.attrs.get("shock_line_j") is not None:
        shock_line_j = int(ds.attrs["shock_line_j"])
        source = "the solution file"
    if shock_line_j is not None and shock_line_j >= nj:
        raise ValueError(f"shock_line_j = {shock_line_j} lies outside the grid (j = 0 to {nj - 1})")

    # debug output for devs
    logger.debug("shock line: %s (from %s)", shock_line_j, source)

    # build the report table: one row per grid line with a shock, the shock point first
    # (as x, y, so Tecplot plots the shock over the flow field)
    table = {
        "x": extraction.x,
        "y": extraction.y,
        "x_wall": ds.grid.x[extraction.station, 0, 0],
        "y_wall": ds.grid.y[extraction.station, 0, 0],
        "station": extraction.station.astype(float),
        "j_shock": extraction.j,
        "distance": extraction.distance,
        "thickness_cells": extraction.thickness_cells,
    }
    if options.method == "hybrid":
        # 1: walk, 2: indicator, 3: both
        table["detected_by"] = extraction.detected_by.astype(float)
    offsets = None
    if shock_line_j is not None:
        # positive: the solution's shock lies outside the grid's shock line
        offsets = extraction.j - shock_line_j
        table["offset_cells"] = offsets

    # smooth: the curve that is written becomes a grid line
    curve = extraction
    moved = None
    if options.smoothing_length > 0.0:
        curve = smooth_shock(ds, extraction, options.smoothing_length)
        moved = curve.j - extraction.j
        table["x_smooth"] = curve.x
        table["y_smooth"] = curve.y
        table["j_smooth"] = curve.j
        table["smoothing_cells"] = moved

    # write: the curve, then the report
    curve_path = write_shock_curve(config_dir / options.output, curve)
    report_path = write_tecplot_table(
        config_dir / options.report, table, title="flow-props shock", zone_title="shock"
    )

    # report: where the shock was found and how thick it is
    n_found = extraction.station.size
    detection = {
        "walk": f"walk (first rise of {options.min_rise:.0%} over freestream)",
        "indicator": f"indicator (WENO smoothness indicator >= {options.indicator_threshold:g})",
        "hybrid": (
            f"hybrid (outermost jump of the walk, {options.min_rise:.0%} rise, and the "
            f"indicator, >= {options.indicator_threshold:g})"
        ),
    }
    report = [
        f"Shock on {n_found} of {ni} grid lines ({options.quantity}; method "
        f"{detection[options.method]}; point at level {options.level:g} of the jump)"
    ]
    if options.method == "hybrid":
        by_walk = int(np.count_nonzero(extraction.detected_by & DETECTED_BY_WALK))
        by_indicator = int(np.count_nonzero(extraction.detected_by & DETECTED_BY_INDICATOR))
        both = int(
            np.count_nonzero(extraction.detected_by == DETECTED_BY_WALK + DETECTED_BY_INDICATOR)
        )
        report.append(
            f"  found by the walk on {by_walk} lines, by the indicator on {by_indicator} "
            f"(both on {both})"
        )
    if extraction.stations_without_shock:
        report.append(f"  no shock on i = {_index_ranges(extraction.stations_without_shock)}")
    thickness = extraction.thickness_cells
    report.append(
        f"  captured thickness (10-90 % of the jump): {thickness.min():.1f} to "
        f"{thickness.max():.1f} cells (median {np.median(thickness):.1f})"
    )

    # report: how the shock sits on the grid's shock line
    if offsets is None:
        report.append("  no shock line to compare with (set [shock] shock_line_j)")
    else:
        worst = int(np.argmax(np.abs(offsets)))
        worst_x = float(ds.grid.x[extraction.station[worst], 0, 0])
        n_off = int(np.count_nonzero(np.abs(offsets) > ALIGNED_OFFSET_CELLS))
        report.append(
            f"  offset from shock line j = {shock_line_j} (from {source}): "
            f"{offsets.min():+.2f} to {offsets.max():+.2f} cells "
            f"(largest at wall x = {worst_x:.6g} m)"
        )
        report.append(f"  {n_off} of {n_found} lines more than a quarter cell off the shock line")

    # report: how far the smoothing moved the points
    if moved is None:
        report.append("  curve written as found (smoothing_length = 0)")
    else:
        rms = float(np.sqrt(np.mean(moved**2)))
        report.append(
            f"  smoothed over {options.smoothing_length:g} shock distances: points moved by up to "
            f"{np.abs(moved).max():.2f} cells (rms {rms:.2f})"
        )

    # report: written files
    report.append(f"Wrote: {curve_path}")
    report.append(f"Wrote: {report_path}")

    return report


# --------------------------------------------------
# helpers
# --------------------------------------------------


def _index_ranges(indices: list[int]) -> str:
    """Compact text for sorted indices: [0, 1, 2, 7, 9, 10] -> "0-2, 7, 9-10"."""

    # group consecutive indices into runs
    runs: list[list[int]] = []
    for index in indices:
        if runs and index == runs[-1][-1] + 1:
            runs[-1].append(index)
        else:
            runs.append([index])

    # format each run
    parts = []
    for run in runs:
        if len(run) == 1:
            parts.append(str(run[0]))
        else:
            parts.append(f"{run[0]}-{run[-1]}")
    text = ", ".join(parts)

    return text
