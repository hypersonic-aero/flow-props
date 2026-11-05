"""Bundled example cases and the `flow-props <calculation> examples` command.

Each example is a folder under ``flow_props/examples/<calculation>/`` holding the
files of one case, with the config named like run's default (e.g. ``bl.toml``).
The folder name is the example's name and the first text line of its config is
the description, so the folders are the registry: adding an example means
adding a folder.
"""

# --------------------------------------------------
# load necessary modules
# --------------------------------------------------
from __future__ import annotations

from collections.abc import Callable
from importlib.resources import files
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import Annotated

import typer


# --------------------------------------------------
# helpers
# --------------------------------------------------
def _calculation_root(name: str) -> Traversable:
    """Folder of one calculation's examples inside the installed package."""
    return files("flow_props") / "examples" / name


def _description(config_text: str) -> str:
    """First text line of a config's header comment (dividers skipped)."""

    # walk the leading comment block
    for line in config_text.splitlines():
        stripped = line.strip()

        # the header ends at the first line that isn't a comment
        if stripped and not stripped.startswith("#"):
            break

        # skip blank comment lines and dividers like "# -----"
        text = stripped.lstrip("#").strip()
        if text and set(text) - set("-=#"):
            return text

    return ""


# --------------------------------------------------
# public API
# --------------------------------------------------
def list_examples(name: str) -> dict[str, str]:
    """Return an example name -> description mapping of one calculation's examples.

    Args:
        name: Calculation, e.g. "bl".

    Returns:
        Dict of {example name: description}, sorted by name (empty without examples).
    """

    # a calculation without examples has no folder
    root = _calculation_root(name)
    if not root.is_dir():
        return {}

    # every folder with the calculation's config is an example
    examples: dict[str, str] = {}
    config_name = f"{name}.toml"
    for folder in sorted(root.iterdir(), key=lambda item: item.name):
        config = folder / config_name
        if folder.is_dir() and config.is_file():
            examples[folder.name] = _description(config.read_text(encoding="utf-8"))

    return examples


def copy_example(
    name: str, example: str, destination: str | Path, force: bool = False
) -> list[Path]:
    """Copy one example's files into a folder.

    Args:
        name: Calculation, e.g. "bl".
        example: Example name (see list_examples).
        destination: Folder to copy the files into (created if missing).
        force: Overwrite files of the same name in the folder.

    Returns:
        The paths of the written files.

    Raises:
        KeyError: If the calculation has no example of that name.
        FileExistsError: If a file of the example already exists there and force
            is False (nothing is written then).
    """

    # check: a known example
    if example not in list_examples(name):
        raise KeyError(example)

    # the example's files
    source = _calculation_root(name) / example
    items = sorted(
        (entry for entry in source.iterdir() if entry.is_file()),
        key=lambda entry: entry.name,
    )

    # check: don't overwrite the user's files unless asked to (before writing any)
    destination = Path(destination)
    existing = [item.name for item in items if (destination / item.name).exists()]
    if existing and not force:
        raise FileExistsError(f"would overwrite {', '.join(existing)} (use --force to overwrite)")

    # copy every file of the example folder
    destination.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for item in items:
        target = destination / item.name
        target.write_bytes(item.read_bytes())
        written.append(target)

    return written


def build_examples_command(name: str) -> Callable[..., None]:
    """Build the `examples` subcommand of one calculation.

    Args:
        name: Calculation, e.g. "bl" (also the examples folder).

    Returns:
        The command function to register as ``<calculation_app>.command("examples")``.
    """

    def cmd_examples(
        example: Annotated[
            str | None,
            typer.Argument(
                help="Example to copy into the current folder. Omit to list the examples."
            ),
        ] = None,
        force: Annotated[
            bool,
            typer.Option("--force", "-f", help="Overwrite files of the same name in this folder."),
        ] = False,
    ) -> None:
        """List the bundled examples, or copy one into the current folder to run."""

        # list: name and description of every example
        examples = list_examples(name)
        if example is None:
            width = max(len(example_name) for example_name in [*examples, "<name>"])
            typer.echo("")
            typer.echo(f"  {'<name>':<{width}}  description")
            typer.echo(f"  {'-' * width}  {'-' * 11}")
            for example_name, description in examples.items():
                typer.echo(f"  {example_name:<{width}}  {description}")
            typer.echo("")
            typer.echo("command sequence:")
            typer.echo("  - check out the example")
            typer.echo(f"     flow-props {name} examples <name>")
            typer.echo("  - run the example")
            typer.echo(f"     flow-props {name} run")
            typer.echo("")
            return

        # copy the example into the current folder
        try:
            written = copy_example(name, example, Path("."), force=force)
        except KeyError:
            known = ", ".join(examples)
            typer.echo(f"error: unknown example {example!r}; available: {known}", err=True)
            raise typer.Exit(1) from None
        except FileExistsError as error:
            typer.echo(f"error: {error}", err=True)
            raise typer.Exit(1) from None

        # report the files and the next step
        for path in written:
            typer.echo(f"wrote {path}")
        typer.echo(f"then: flow-props {name} run")

    return cmd_examples
