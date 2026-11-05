"""Per-calculation command groups: `flow-props <name> init`, `run` and `examples`."""

# --------------------------------------------------
# load necessary modules
# --------------------------------------------------
from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from flow_props.calculators import CALCULATORS
from flow_props.calculators.common import load_config
from flow_props.cli.examples_command import build_examples_command, list_examples


# --------------------------------------------------
# command group factory
# --------------------------------------------------
def build_calculator_app(name: str) -> typer.Typer:
    """Build the `init` / `run` (/ `examples`) command group for one calculation.

    Every calculation gets the same commands; only the template that `init` writes, the
    config model `run` validates and the run step differ. The commands close over name.

    Args:
        name: Key in CALCULATORS, e.g. "bl".

    Returns:
        Typer app to register as `flow-props <name>`.
    """

    # command group
    calculator = CALCULATORS[name]
    calculator_app = typer.Typer(name=name, help=calculator.help, no_args_is_help=True)
    default_config = Path(f"{name}.toml")

    # init: write the starter configuration
    @calculator_app.command("init")
    def cmd_init(
        output: Annotated[
            Path,
            typer.Option("--output", "-o", help="Configuration file to create."),
        ] = default_config,
        force: Annotated[
            bool,
            typer.Option("--force", "-f", help="Overwrite an existing configuration."),
        ] = False,
    ) -> None:
        """Create a starter configuration (every key documented)."""

        # protect existing user configuration unless overwrite was requested
        if output.exists() and not force:
            typer.echo(f"error: {output} already exists. Use --force to overwrite.", err=True)
            raise typer.Exit(code=1)

        # write template
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(calculator.template, encoding="utf-8")

        typer.echo(f"Created: {output}")
        run_hint = f"flow-props {name} run"
        if output != default_config:
            run_hint = f"{run_hint} -c {output}"
        typer.echo(f"Edit the file, then run: {run_hint}")

    # run: read the configuration, run the calculation, report
    @calculator_app.command("run")
    def cmd_run(
        config: Annotated[
            Path,
            typer.Option("--config", "-c", help="Configuration file."),
        ] = default_config,
    ) -> None:
        """Run the calculation of a configuration."""

        # check: a configuration to run (with a hint how to get one)
        if not config.is_file():
            typer.echo(
                f"error: config file not found: {config} (create one with: flow-props {name} init)",
                err=True,
            )
            raise typer.Exit(code=1)

        # read, validate and run (pydantic's ValidationError and TOML errors are ValueErrors)
        try:
            case_config = load_config(config, calculator.config_model)
            report = calculator.run(case_config, config.parent)
        except (OSError, ValueError, TypeError, IndexError) as error:
            typer.echo(f"error: {error}", err=True)
            raise typer.Exit(code=1) from None
        except KeyError as error:
            # a KeyError's text is its quoted argument; print the message itself
            message = error.args[0] if error.args else str(error)
            typer.echo(f"error: {message}", err=True)
            raise typer.Exit(code=1) from None

        # report
        for line in report:
            typer.echo(line)

    # examples: bundled cases, copied into the current folder to run (only if there are some)
    if list_examples(name):
        calculator_app.command("examples")(build_examples_command(name))

    return calculator_app
