"""Registry of the calculations: one command group each (``flow-props <name> init/run``).

Each calculation is one module in this package with its config model, its template
(what ``init`` writes) and ``run_case`` (what ``run`` does). Adding a calculation means
adding a module and one entry below; the CLI builds its commands from this registry.
"""

# --------------------------------------------------
# load necessary modules
# --------------------------------------------------
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel

from flow_props.calculators import bl, entropy_layer, profiles, shock


# --------------------------------------------------
# registry
# --------------------------------------------------
@dataclass(frozen=True, slots=True)
class Calculator:
    """What the CLI needs to know about one calculation."""

    # one-line help of the command group
    help: str
    # config file written by init
    template: str
    # pydantic model that validates the config file
    config_model: type[BaseModel]
    # run(config, config_dir) -> report lines
    run: Callable[[BaseModel, Path], list[str]]


# command name -> calculation (order of registration is order of appearance in the CLI help)
CALCULATORS: dict[str, Calculator] = {
    "bl": Calculator(
        help="Boundary-layer edge and integral thicknesses per station.",
        template=bl.TEMPLATE,
        config_model=bl.BLCaseConfig,
        run=bl.run_case,
    ),
    "entropy-layer": Calculator(
        help="Entropy-layer thickness per station.",
        template=entropy_layer.TEMPLATE,
        config_model=entropy_layer.EntropyLayerCaseConfig,
        run=entropy_layer.run_case,
    ),
    "profiles": Calculator(
        help="Wall-normal profiles of every flow variable per station.",
        template=profiles.TEMPLATE,
        config_model=profiles.ProfilesCaseConfig,
        run=profiles.run_case,
    ),
    "shock": Calculator(
        help="Shock curve of a solution, and its offset from a shock-line grid's shock line.",
        template=shock.TEMPLATE,
        config_model=shock.ShockCaseConfig,
        run=shock.run_case,
    ),
}

__all__ = ["CALCULATORS", "Calculator"]
