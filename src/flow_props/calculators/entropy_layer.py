"""The entropy-layer calculation: entropy-layer thickness per station.

``flow-props entropy-layer init`` writes the template below, ``flow-props entropy-layer
run`` reads it, extracts the wall-normal grid-line profiles at the selected stations and
writes the entropy-layer thickness of each (see flow_props.entropy_layer).
"""

# --------------------------------------------------
# load necessary modules
# --------------------------------------------------
from __future__ import annotations

import logging
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from flow_props.calculators.common import (
    AT_CELL_CENTRES,
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
    solution_points,
)

logger = logging.getLogger(__name__)

# --------------------------------------------------
# config models (input boundary)
# --------------------------------------------------


class EntropyLayerConfig(BaseModel):
    """Entropy-layer extraction options."""

    model_config = ConfigDict(extra="forbid")

    threshold: float = Field(
        default=0.25,
        gt=0.0,
        lt=1.0,
        description="Fraction of the wall entropy rise that marks the entropy-layer edge",
    )
    temp_edge: float | None = Field(
        default=None,
        gt=0.0,
        description="Reference temperature [K]; default: the profile's outermost point",
    )
    pressure_edge: float | None = Field(
        default=None,
        gt=0.0,
        description="Reference pressure [Pa]; default: the profile's outermost point",
    )
    output: str = Field(
        default="entropy_layer.dat",
        description="Output Tecplot ASCII file path",
    )


class EntropyLayerCaseConfig(CaseConfig):
    """Config file of the entropy-layer calculation."""

    stations: StationsConfig = Field(default_factory=StationsConfig)
    entropy_layer: EntropyLayerConfig = Field(default_factory=EntropyLayerConfig)
    gas: GasConfig = Field(default_factory=GasConfig)


# --------------------------------------------------
# template (written by: flow-props entropy-layer init)
# --------------------------------------------------

ENTROPY_LAYER_BLOCK = """\
#--------------------------------------------------
# entropy layer extraction
#--------------------------------------------------
[entropy_layer]
# Edge of the entropy layer: where the entropy rise over the reference state falls
# to this fraction of its value at the wall.
threshold = 0.25
# Reference temperature [K]; uncomment to override the profile's outermost point.
# temp_edge = 220.0
# Reference pressure [Pa]; uncomment to override the profile's outermost point.
# pressure_edge = 2000.0
# Tecplot ASCII table of the entropy-layer thickness per station.
output = "entropy_layer.dat"
"""

TEMPLATE = build_template(
    "entropy-layer", INPUT_BLOCK, STATIONS_BLOCK, ENTROPY_LAYER_BLOCK, GAS_BLOCK
)


# --------------------------------------------------
# public API
# --------------------------------------------------


def run_case(cfg: EntropyLayerCaseConfig, config_dir: Path) -> list[str]:
    """Extract the entropy-layer thickness at the configured stations and write it.

    Args:
        cfg: Validated EntropyLayerCaseConfig instance.
        config_dir: Directory containing the config file (used to resolve
            relative input/output paths).

    Returns:
        Report lines for the command line.
    """
    from flow_props.entropy_layer import run_entropy
    from flow_props.output import write_station_output

    # resolve file paths relative to the config directory, then load the dataset (cell
    # values go on a grid of the cell centres)
    fname, gname = input_paths(cfg, config_dir)
    ds, location = solution_points(load_dataset(fname, gname))

    # resolve the stations and the gas constants (the entropy rise needs both)
    stations = resolve_stations(ds, cfg.stations)
    gamma, gas_constant = resolve_gas_properties(
        cfg.gas, ds.attrs, fname, required=True, needed_for="the entropy layer"
    )

    # debug output for devs
    logger.debug("processing %d stations", len(stations))

    # compute: 0 asks run_entropy for the profile's outermost point as reference
    options = cfg.entropy_layer
    results = run_entropy(
        ds,
        stations,
        method="grid_line",
        threshold=options.threshold,
        gamma=gamma,
        gas_constant=gas_constant,
        temp_edge=options.temp_edge or 0.0,
        pressure_edge=options.pressure_edge or 0.0,
    )

    # write: wall x, y, z of each station plus its thickness
    output_path = config_dir / options.output
    write_station_output(
        output_path,
        ds,
        stations,
        results,
        title="flow-props entropy layer",
        zone_title="entropy_layer",
    )

    # report
    report = [
        f"Entropy layer at {len(stations)} stations (edge at {options.threshold:g} "
        "of the wall entropy rise)",
        f"Wrote: {output_path}",
    ]

    if location == AT_CELL_CENTRES:
        report.insert(
            1,
            "  solution at cell centres: profiles run through them, from the first cell "
            "centre (half a cell off the wall)",
        )

    return report
