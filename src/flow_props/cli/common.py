"""Shared CLI helpers for flow-props."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import typer
from cfd_io.dataset import Dataset, StructuredGrid

from flow_props import __version__
from flow_props.config import load_config

# --------------------------------------------------
# config file defaults
# --------------------------------------------------

DEFAULT_CONFIG_NAME = "flow_props.toml"

ROOT_TEMPLATE = """\
fname = "solution.vtu"         # path to CFD data file
gname = ""                     # grid file for split formats only
"""


# --------------------------------------------------
# public helpers
# --------------------------------------------------


def version_callback(value: bool) -> None:
    """Print the package version and exit when requested."""
    if value:
        typer.echo(f"flow-props {__version__}")
        raise typer.Exit()


def verbose_echo(ctx: typer.Context | None, message: str) -> None:
    """Print an informational message when verbose mode is enabled."""
    # check whether verbose output is enabled
    if ctx is None:
        return
    if not ctx.obj:
        return
    if not ctx.obj.get("verbose", False):
        return

    # write informational output for the user
    typer.echo(f"[info] {message}")


def load_cli_config(config_path: Path | None, section: str) -> tuple[dict, Path | None]:
    """Load config values and return the resolved config path.

    Args:
        config_path: Explicit config path from the CLI, or ``None``.
        section: Section name to merge with the root values.

    Returns:
        Tuple of merged config values and the resolved config path.
    """
    # resolve the config path from the CLI or default file name
    resolved_path = runtime_config_path(config_path)

    # default empty dicts for optional arguments
    if resolved_path is None:
        return {}, None

    # read config values from disk
    try:
        cfg = load_config(resolved_path, section)
    except FileNotFoundError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    return cfg, resolved_path


def runtime_config_path(config_path: Path | None) -> Path | None:
    """Return the runtime config path if one is available."""
    # prefer the explicit config path when one is provided
    if config_path is not None:
        return config_path

    # fall back to the default config file in the current directory
    default_path = Path(DEFAULT_CONFIG_NAME)
    if default_path.is_file():
        return default_path

    return None


def init_config_path(config_path: Path | None) -> Path:
    """Return the config path to use during initialization."""
    # prefer the explicit config path when one is provided
    if config_path is not None:
        return config_path

    # use the default config file name in the current directory
    return Path(DEFAULT_CONFIG_NAME)


def config_path_value(
    raw_value: str | Path | None,
    config_path: Path | None,
    allow_empty: bool = False,
) -> Path | None:
    """Resolve a path value from config text."""
    # default empty dicts for optional arguments
    if raw_value is None:
        return None

    # check for empty path text before building a Path object
    raw_text = str(raw_value).strip()
    if raw_text == "":
        return None

    # convert to Path object
    path_value = Path(raw_value)

    # allow empty path values for optional config entries
    if allow_empty and str(path_value).strip() == "":
        return None

    # resolve relative config values against the config file directory
    if config_path is not None and not path_value.is_absolute():
        return config_path.parent / path_value

    return path_value


def require_input_path(
    input_path: Path | None,
    cli_input: Path | None,
    config_path: Path | None,
) -> Path:
    """Validate the runtime input path and raise a useful CLI error.

    Args:
        input_path: Resolved runtime input path from CLI or config.
        cli_input: Explicit CLI input path, if one was provided.
        config_path: Resolved config path, if one was used.

    Returns:
        Validated input path.

    Raises:
        typer.Exit: If the input path cannot be resolved or does not exist.
    """
    # check the explicit CLI input first
    if cli_input is not None:
        if cli_input.is_file():
            return cli_input

        typer.echo(f"error: input file not found: {cli_input}", err=True)
        raise typer.Exit(code=1)

    # report a missing default config before falling through to generic input errors
    if config_path is None and input_path is None:
        hint_message = _default_config_hint()
        typer.echo(
            "error: no input file was provided and no default config file was found. "
            f"Expected {DEFAULT_CONFIG_NAME} in the current directory. "
            "Use --config <file> or --input <file>."
            f"{hint_message}",
            err=True,
        )
        raise typer.Exit(code=1)

    # report a config file that exists but does not define fname
    if input_path is None:
        typer.echo(
            "error: no input file was configured. "
            f"Set 'fname' in {config_path} or pass --input <file>.",
            err=True,
        )
        raise typer.Exit(code=1)

    # report a missing file path with config context when available
    if not input_path.is_file():
        if config_path is not None:
            typer.echo(
                f"error: input file not found: {input_path} (from {config_path})",
                err=True,
            )
            raise typer.Exit(code=1)

        typer.echo(f"error: input file not found: {input_path}", err=True)
        raise typer.Exit(code=1)

    return input_path


def resolve_station_list(
    ds: Dataset,
    cfg: dict,
    stations_text: str | None = None,
    i_s: int | None = None,
    i_e: int | None = None,
    di: int | None = None,
    x_s: float | None = None,
    x_e: float | None = None,
    dx: float | None = None,
) -> list[int]:
    """Resolve station selection from explicit, index-range, or x-range inputs.

    Args:
        ds: Dataset used for bounds checks and x-space selection.
        cfg: Merged CLI config section.
        stations_text: Comma-separated station indices from the CLI.
        i_s: Inclusive starting i-index.
        i_e: Inclusive ending i-index.
        di: Positive i-index step size.
        x_s: Inclusive starting wall x-location.
        x_e: Inclusive ending wall x-location.
        dx: Positive wall x step size.

    Returns:
        Resolved list of station indices.

    Raises:
        typer.Exit: If the station selection is invalid or ambiguous.
    """
    # validate dataset type for any station-selection mode
    if not isinstance(ds.grid, StructuredGrid):
        typer.echo("error: station selection requires a StructuredGrid dataset.", err=True)
        raise typer.Exit(code=1)

    # build the combined explicit station list from CLI or config values
    if stations_text is not None:
        station_values = _parse_station_text(stations_text)
    else:
        station_values = cfg.get("stations")

    # build the combined i-range selection from CLI or config values
    i_s_value = i_s if i_s is not None else cfg.get("i_s")
    i_e_value = i_e if i_e is not None else cfg.get("i_e")
    di_value = di if di is not None else cfg.get("di")

    # build the combined x-range selection from CLI or config values
    x_s_value = x_s if x_s is not None else cfg.get("x_s")
    x_e_value = x_e if x_e is not None else cfg.get("x_e")
    dx_value = dx if dx is not None else cfg.get("dx")

    # select exactly one station-selection mode
    stations_active = station_values is not None
    i_range_active = i_s_value is not None or i_e_value is not None or di_value is not None
    x_range_active = x_s_value is not None or x_e_value is not None or dx_value is not None
    active_modes = int(stations_active) + int(i_range_active) + int(x_range_active)

    if active_modes == 0:
        typer.echo(
            "error: no stations specified. Use 'stations', an i-range, or an x-range selection.",
            err=True,
        )
        raise typer.Exit(code=1)

    if active_modes > 1:
        typer.echo(
            "error: multiple station selection modes were configured. Use only one of: "
            "stations, i_s/i_e/di, or x_s/x_e/dx.",
            err=True,
        )
        raise typer.Exit(code=1)

    # validate and return the explicit station list
    if stations_active:
        return _validate_station_indices(ds.grid.shape[0], station_values)

    # validate and build the i-range station list
    if i_range_active:
        station_list = _select_i_range(ds.grid.shape[0], i_s_value, i_e_value, di_value)
        return station_list

    # validate and build the x-range station list
    station_list = _select_x_range(ds, x_s_value, x_e_value, dx_value)
    return station_list


# --------------------------------------------------
# private helpers
# --------------------------------------------------


def _default_config_hint() -> str:
    """Build a hint string for alternate TOML files in the current directory."""
    # search for likely config files near the current working directory
    toml_paths = sorted(Path.cwd().glob("*.toml"))
    other_names = [path.name for path in toml_paths if path.name != DEFAULT_CONFIG_NAME]

    # return an empty suffix when there are no nearby TOML files to suggest
    if not other_names:
        return ""

    # build a compact hint using the first few nearby TOML files
    shown_names = ", ".join(other_names[:3])
    return f" Found TOML file(s) here: {shown_names}."


def _parse_station_text(stations_text: str) -> list[int]:
    """Parse a comma-separated CLI station list."""
    # split and strip the raw CLI text into integer tokens
    tokens = [token.strip() for token in stations_text.split(",") if token.strip()]
    if not tokens:
        typer.echo("error: empty station list provided.", err=True)
        raise typer.Exit(code=1)

    try:
        station_list = [int(token) for token in tokens]
    except ValueError as exc:
        typer.echo("error: stations must be comma-separated integers.", err=True)
        raise typer.Exit(code=1) from exc

    return station_list


def _validate_station_indices(ni: int, station_list: list[int]) -> list[int]:
    """Validate an explicit station list against the grid bounds."""
    # validate station bounds in input order
    for station in station_list:
        if station < 0 or station >= ni:
            typer.echo(f"error: station i={station} out of range [0, {ni}).", err=True)
            raise typer.Exit(code=1)

    return station_list


def _select_i_range(
    ni: int,
    i_s: int | None,
    i_e: int | None,
    di: int | None,
) -> list[int]:
    """Build a station list from an inclusive i-index range."""
    # validate required range endpoints
    if i_s is None or i_e is None:
        typer.echo("error: i-range selection requires both i_s and i_e.", err=True)
        raise typer.Exit(code=1)

    # default the step size to one station when it is not provided
    step_size = 1 if di is None else di
    if step_size <= 0:
        typer.echo("error: di must be positive.", err=True)
        raise typer.Exit(code=1)

    # validate inclusive range bounds against the streamwise grid size
    if i_s < 0 or i_s >= ni:
        typer.echo(f"error: i_s={i_s} out of range [0, {ni}).", err=True)
        raise typer.Exit(code=1)
    if i_e < 0 or i_e >= ni:
        typer.echo(f"error: i_e={i_e} out of range [0, {ni}).", err=True)
        raise typer.Exit(code=1)

    # build an inclusive range that supports either direction
    direction = 1 if i_e >= i_s else -1
    stop_value = i_e + direction
    station_list = list(range(i_s, stop_value, direction * step_size))

    if not station_list:
        typer.echo("error: i-range selection produced no stations.", err=True)
        raise typer.Exit(code=1)

    return station_list


def _select_x_range(
    ds: Dataset,
    x_s: float | None,
    x_e: float | None,
    dx: float | None,
) -> list[int]:
    """Build a station list by sampling the wall x-coordinate and mapping to i-indices."""
    # validate required x-range inputs
    if x_s is None or x_e is None:
        typer.echo("error: x-range selection requires both x_s and x_e.", err=True)
        raise typer.Exit(code=1)

    # default the x increment to a single target value when it is not provided
    step_size = 0.0 if dx is None else dx
    if step_size < 0.0:
        typer.echo("error: dx must be non-negative.", err=True)
        raise typer.Exit(code=1)

    # read wall x coordinates along the first wall line
    wall_x = np.asarray(ds.grid.x[:, 0, 0], dtype=float)
    ni = ds.grid.shape[0]

    # build target x values between the inclusive endpoints
    target_x = _build_target_x_values(float(x_s), float(x_e), float(step_size))
    station_list: list[int] = []
    for target_value in target_x:
        nearest_index = int(np.argmin(np.abs(wall_x - target_value)))
        if nearest_index < 0 or nearest_index >= ni:
            typer.echo(f"error: x target {target_value} mapped out of range.", err=True)
            raise typer.Exit(code=1)
        if nearest_index not in station_list:
            station_list.append(nearest_index)

    if not station_list:
        typer.echo("error: x-range selection produced no stations.", err=True)
        raise typer.Exit(code=1)

    return station_list


def _build_target_x_values(x_s: float, x_e: float, dx: float) -> list[float]:
    """Build inclusive target x values for wall sampling."""
    # handle the single-target case directly
    if dx == 0.0 or x_s == x_e:
        return [x_s]

    # build evenly spaced targets in the requested direction
    direction = 1.0 if x_e >= x_s else -1.0
    step_size = direction * dx
    if step_size == 0.0:
        return [x_s]

    target_values: list[float] = []
    current_value = x_s
    tolerance = abs(dx) * 1.0e-9 + 1.0e-12
    while (direction > 0.0 and current_value <= x_e + tolerance) or (
        direction < 0.0 and current_value >= x_e - tolerance
    ):
        target_values.append(float(current_value))
        current_value += step_size

    return target_values
