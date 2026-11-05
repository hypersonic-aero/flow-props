"""Output helpers for flow-props result files."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from cfd_io.dataset import Dataset, Field, StructuredGrid

# --------------------------------------------------
# output format constants
# --------------------------------------------------

OUTPUT_FORMAT_TECPLOT_ASCII = "tecplot_ascii"

# column order for the BL properties output file
BL_PROPERTIES_COLUMNS = (
    "x",
    "delta",
    "delta_star",
    "theta",
    "H",
    "uvel_edge",
    "temp_edge",
    "dens_edge",
    "mach_edge",
    "twte",
)


# --------------------------------------------------
# public API
# --------------------------------------------------


def write_bl_properties(table: dict[str, np.ndarray], output_path: str | Path) -> Path:
    """Write the BL properties table as a Tecplot ASCII file.

    Args:
        table: Dict mapping each column name in ``BL_PROPERTIES_COLUMNS`` to a
            1-D array over stations.
        output_path: Output file path (``.dat``).

    Returns:
        Path to the created file.

    Raises:
        KeyError: If a required column is missing from *table*.
        ValueError: If the output suffix is unsupported.
    """
    # convert to Path object and validate suffix
    output_path = Path(output_path)
    detect_output_format(output_path)

    # validate that every required column is present
    missing = [name for name in BL_PROPERTIES_COLUMNS if name not in table]
    if missing:
        raise KeyError(f"missing output columns: {', '.join(missing)}")

    # stack columns in fixed order
    arrays = [np.asarray(table[name], dtype=float) for name in BL_PROPERTIES_COLUMNS]
    n_rows = len(arrays[0])

    # write Tecplot header and one data row per station
    with open(output_path, "w") as fobj:
        fobj.write('TITLE = "boundary_layer_properties"\n')
        header = "\n".join(f'"{name}"' for name in BL_PROPERTIES_COLUMNS)
        fobj.write(f"VARIABLES = {header}\n")
        fobj.write('ZONE T = "bl_properties"\n')
        for row_index in range(n_rows):
            row = " ".join(f"{arr[row_index]:.8E}" for arr in arrays)
            fobj.write(row + "\n")

    return output_path


def write_tecplot_table(
    output_path: str | Path,
    columns: dict[str, np.ndarray],
    *,
    title: str,
    zone_title: str,
) -> Path:
    """Write columns of equal length as one I-ordered Tecplot ASCII zone.

    The columns are written in dict order; name the first two "x" and "y" and Tecplot
    plots the rows as a line in the (x, y) plane, e.g. over a loaded flow field.

    Args:
        output_path: Output file path (``.dat``).
        columns: Column name -> 1-D array, all of the same length.
        title: Tecplot TITLE string.
        zone_title: Tecplot zone title.

    Returns:
        Path to the created file.

    Raises:
        ValueError: If the suffix is unsupported or the columns differ in length.
    """
    # convert to Path object and validate suffix
    output_path = Path(output_path)
    detect_output_format(output_path)

    # validate: one length for every column
    arrays = [np.asarray(values, dtype=float) for values in columns.values()]
    n_rows = arrays[0].size
    for name, values in zip(columns, arrays):
        if values.size != n_rows:
            raise ValueError(f"column {name} has {values.size} rows, expected {n_rows}")

    # write Tecplot header and one data row per entry
    with open(output_path, "w") as fobj:
        fobj.write(f'TITLE = "{title}"\n')
        variable_text = ", ".join(f'"{name}"' for name in columns)
        fobj.write(f"VARIABLES = {variable_text}\n")
        fobj.write(f'ZONE T = "{zone_title}", I = {n_rows}, F = POINT\n')
        for row_index in range(n_rows):
            row = " ".join(f"{values[row_index]:.10E}" for values in arrays)
            fobj.write(row + "\n")

    return output_path


def write_profiles_output(
    output_path: str | Path,
    profiles: list[dict[str, np.ndarray]],
    stations: list[int],
) -> Path:
    """Write extracted profiles to disk as a multi-zone Tecplot ASCII file.

    Args:
        output_path: Output file path.
        profiles: Extracted profile dictionaries.
        stations: Station indices for each profile.

    Returns:
        Path to the created file.

    Raises:
        ValueError: If the output format is unsupported.
    """
    # convert to Path object
    output_path = Path(output_path)

    # validate output format from the filename suffix
    detect_output_format(output_path)

    # write Tecplot ASCII output
    return write_profiles_tecplot_ascii(output_path, profiles, stations)


def write_station_output(
    output_path: str | Path,
    ds: Dataset,
    stations: list[int],
    results: list[dict[str, float]] | dict[str, np.ndarray],
    *,
    title: str,
    zone_title: str,
) -> Path:
    """Write station-based reduction results to disk as a Tecplot ASCII file.

    Args:
        output_path: Output file path.
        ds: Source CFD dataset.
        stations: Station indices in output order.
        results: Station-based results as a list of dicts or dict of arrays.
        title: Tecplot TITLE string.
        zone_title: Tecplot zone title.

    Returns:
        Path to the created file.

    Raises:
        ValueError: If the output format is unsupported.
    """
    # convert to Path object
    output_path = Path(output_path)

    # validate output format from the filename suffix
    detect_output_format(output_path)

    # write Tecplot ASCII output
    dataset = station_results_to_dataset(ds, stations, results, title=title)
    from cfd_io.writers.tecplot_ascii import write_tecplot_ascii

    return write_tecplot_ascii(
        output_path,
        dataset,
        title=title,
        zone_title=zone_title,
    )


def detect_output_format(output_path: str | Path) -> str:
    """Infer the output format from a file suffix.

    Args:
        output_path: Output file path.

    Returns:
        Output format name.

    Raises:
        ValueError: If the file suffix is unsupported.
    """
    # convert to Path object
    output_path = Path(output_path)
    suffix = output_path.suffix.lower()

    # map known suffixes to output formats
    if suffix == ".dat":
        return OUTPUT_FORMAT_TECPLOT_ASCII
    if suffix == ".plt":
        raise ValueError(
            "Tecplot binary output is not implemented yet. Use .dat for Tecplot ASCII."
        )

    raise ValueError(f"unsupported output suffix: {suffix}. Use .dat for Tecplot ASCII.")


# --------------------------------------------------
# dataset builders
# --------------------------------------------------


def write_profiles_tecplot_ascii(
    output_path: str | Path,
    profiles: list[dict[str, np.ndarray]],
    stations: list[int],
) -> Path:
    """Write profiles as a multi-zone Tecplot ASCII file.

    Args:
        output_path: Output file path.
        profiles: Extracted profile dictionaries.
        stations: Station indices in output order.

    Returns:
        Path to the created file.

    Raises:
        ValueError: If the profiles list is empty or inconsistent.
    """
    # validate inputs
    output_path = Path(output_path)
    if not profiles:
        raise ValueError("cannot write Tecplot profiles output with no profiles")
    if len(profiles) != len(stations):
        raise ValueError("profile count does not match station count")

    # build the output variable order from the first profile
    first_profile = profiles[0]
    variable_names = ["x", "y", "z", "eta", "station"]
    for name in first_profile:
        if name in {"x", "y", "eta"}:
            continue
        variable_names.append(name)

    # write Tecplot header and one zone per profile
    with open(output_path, "w") as fobj:
        fobj.write('TITLE = "flow-props profiles"\n')
        variable_text = ", ".join(f'"{name}"' for name in variable_names)
        fobj.write(f"VARIABLES = {variable_text}\n")

        for station, profile in zip(stations, profiles):
            n_points = len(profile["eta"])
            # build zone title from wall x-location and k-index (1-based), matching legacy format
            x_wall = float(np.asarray(profile["x"])[0])
            k_index = 1
            zone_title = f"x_{x_wall:.4f}_m_k_{k_index:05d}"
            fobj.write(f'ZONE T="{zone_title}", I={n_points}, J=1, K=1, F=POINT\n')

            station_values = np.full(n_points, float(station))
            z_values = np.zeros(n_points)
            field_map: dict[str, np.ndarray] = {
                "x": np.asarray(profile["x"], dtype=float),
                "y": np.asarray(profile["y"], dtype=float),
                "z": z_values,
                "eta": np.asarray(profile["eta"], dtype=float),
                "station": station_values,
            }
            for name in first_profile:
                if name in {"x", "y", "eta"}:
                    continue
                field_map[name] = np.asarray(profile[name], dtype=float)

            for point_index in range(n_points):
                row_values = []
                for name in variable_names:
                    row_values.append(f"{field_map[name][point_index]:16.8E}")
                fobj.write(" ".join(row_values) + "\n")

    return output_path


def station_results_to_dataset(
    ds: Dataset,
    stations: list[int],
    results: list[dict[str, float]] | dict[str, np.ndarray],
    *,
    title: str,
) -> Dataset:
    """Convert station-based results to a 1-D structured Dataset.

    Args:
        ds: Source CFD dataset.
        stations: Station indices in output order.
        results: Station-based results as a list of dicts or dict of arrays.
        title: Dataset title.

    Returns:
        Structured Dataset for Tecplot export.
    """
    # build wall coordinate arrays for the requested stations
    n_stations = len(stations)
    x = np.zeros((n_stations, 1, 1))
    y = np.zeros((n_stations, 1, 1))
    z = np.zeros((n_stations, 1, 1))
    for output_index, station in enumerate(stations):
        x[output_index, 0, 0] = ds.grid.x[station, 0, 0]
        y[output_index, 0, 0] = ds.grid.y[station, 0, 0]
        z[output_index, 0, 0] = ds.grid.z[station, 0, 0]

    grid = StructuredGrid(x=x, y=y, z=z)

    # normalize results to a dict of 1-D arrays
    flow_arrays = normalize_station_results(stations, results)

    # build field objects for Tecplot export
    flow: dict[str, Field] = {}
    for name, values in flow_arrays.items():
        flow[name] = Field(data=values.reshape((n_stations, 1, 1)), association="node")

    return Dataset(grid=grid, flow=flow, attrs={"title": title})


def normalize_station_results(
    stations: list[int],
    results: list[dict[str, float]] | dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Normalize station-based results to arrays.

    Args:
        stations: Station indices in output order.
        results: Station-based results as a list of dicts or dict of arrays.

    Returns:
        Dict of 1-D arrays.
    """
    # use dict-of-arrays results directly after normalizing dtypes
    if isinstance(results, dict):
        flow_arrays: dict[str, np.ndarray] = {}
        for name, values in results.items():
            flow_arrays[name] = np.asarray(values, dtype=float)
        return flow_arrays

    # build arrays from a list of per-station dicts
    flow_arrays = {"station": np.asarray(stations, dtype=float)}
    if not results:
        return flow_arrays

    first_result = results[0]
    for name in first_result:
        flow_arrays[name] = np.asarray([result[name] for result in results], dtype=float)

    return flow_arrays
