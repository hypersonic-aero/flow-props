"""CLI subcommand: flow-props examples"""

from __future__ import annotations

from pathlib import Path

import typer


def cmd_examples(
    name: str | None = typer.Argument(
        None,
        help="Name of the example to copy into the current directory.",
    ),
) -> None:
    """List bundled example configs, or copy one into the current directory.

    Without an argument: print the available examples.
    With a name:         copy <name>.toml into the current directory.
    """
    from flow_props.io.examples import (
        available_examples,
        get_example_text,
    )

    # no name: list all available examples
    if name is None:
        typer.echo("Available examples:")
        for example in available_examples():
            typer.echo(f"  {example.name:24s}  {example.description}")
        typer.echo("")
        typer.echo("Copy one with:  flow-props examples <name>")
        return

    # resolve the example text, reporting unknown names clearly
    try:
        text = get_example_text(name)
    except KeyError:
        valid = ", ".join(ex.name for ex in available_examples())
        typer.echo(
            f"error: unknown example: {name}  (available: {valid})",
            err=True,
        )
        raise typer.Exit(code=1) from None

    # refuse to overwrite an existing file
    dest = Path(f"{name}.toml")
    if dest.exists():
        typer.echo(f"error: {dest} already exists.", err=True)
        raise typer.Exit(code=1)

    # write the example config
    dest.write_text(text, encoding="utf-8")
    typer.echo(f"Written {dest}")
    typer.echo(f"Run with:  flow-props run -c {dest}")
