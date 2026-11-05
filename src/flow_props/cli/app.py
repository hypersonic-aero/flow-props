"""Root application for the flow-props CLI."""

# --------------------------------------------------
# load necessary modules
# --------------------------------------------------
from __future__ import annotations

from typing import Annotated

import typer

from flow_props.calculators import CALCULATORS
from flow_props.cli.callbacks import verbose_callback, version_callback
from flow_props.cli.cmd_calculator import build_calculator_app

# --------------------------------------------------
# root application
# --------------------------------------------------
app = typer.Typer(
    name="flow-props",
    help="flow field analysis from CFD data.",
    no_args_is_help=True,
    add_completion=False,
)


# --------------------------------------------------
# register callbacks
# --------------------------------------------------
@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            "-V",
            callback=version_callback,
            is_eager=True,
            help="Show version and exit.",
        ),
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-v",
            callback=verbose_callback,
            help="Enable verbose (debug) output.",
        ),
    ] = False,
) -> None:
    """Flow field analysis from CFD data: one command group per calculation."""

    del version, verbose


# --------------------------------------------------
# register subcommands: one command group per calculation (bl, entropy-layer, profiles, shock)
# --------------------------------------------------
for calculator_name in CALCULATORS:
    app.add_typer(build_calculator_app(calculator_name), name=calculator_name)
