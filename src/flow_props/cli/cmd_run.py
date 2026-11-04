"""CLI subcommand: flow-props run"""

from __future__ import annotations

from pathlib import Path

import typer


def cmd_run(
    bl: bool = typer.Option(
        False,
        "--bl",
        help="Run BL extraction only; default config is bl.toml.",
    ),
    config: Path | None = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to TOML config file.  Defaults to bl.toml (--bl) or flow_props.toml.",
    ),
) -> None:
    """Extract boundary-layer properties from a CFD dataset.

    Without flags: reads flow_props.toml and runs all sections present.
    With --bl:     reads bl.toml and runs BL extraction only.
    An explicit config path always overrides the default filename.
    """
    # import here to keep startup fast
    from flow_props.pipeline import run_pipeline
    from flow_props.schema import load_config

    # resolve default config filename from mode
    default_name = "bl.toml" if bl else "flow_props.toml"
    config_path = config if config is not None else Path(default_name)

    # validate config path exists
    if not config_path.is_file():
        hint = "--bl " if bl else ""
        typer.echo(
            f"error: config file not found: {config_path}  "
            f"(hint: run 'flow-props init {hint}' to create it)",
            err=True,
        )
        raise typer.Exit(code=1)

    # load and validate config
    cfg = load_config(config_path)

    # run the pipeline; pass mode so pipeline can skip irrelevant sections
    result = run_pipeline(cfg, config_dir=config_path.parent, mode="bl" if bl else "all")

    typer.echo(f"Written {result.output_path}  ({result.n_stations} stations)")
