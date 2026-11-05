# Architecture

`flow_props` follows a narrow pipeline:

1. `cfd-io` reads a CFD dataset
2. `flow_props.profiles.extract_profiles` slices wall-normal profiles at chosen stations
3. reduction modules compute derived quantities from those profiles

## Module Layout

- `profiles/`: wall-normal profile extraction (along grid lines or interpolated normals)
- `bl.py`: boundary-layer edge detection and integral thicknesses
- `entropy_layer.py`: entropy-layer thickness
- `shock.py`: shock point per grid line and the shock curve
- `wall.py`: wall shear and thermal quantities
- `output.py`: Tecplot ASCII writers
- `calculators/`: one module per CLI calculation (`bl`, `entropy_layer`, `profiles`,
  `shock`) with its pydantic config model, its `init` template and its `run_case`;
  `common.py` holds the shared parts (input files, `[stations]`, `[gas]`), and
  `__init__.py` the registry the CLI is built from
- `cli/`: Typer app (`app.py`), callbacks (`--version`, `--verbose`), the command-group
  factory (`cmd_calculator.py`) and the `examples` command
- `examples/<calculation>/<name>/`: bundled cases; the folders are the registry

## Adding a Calculation

Write `calculators/<name>.py` with a config model, a `TEMPLATE` and
`run_case(config, config_dir) -> list[str]`, and add one entry to `CALCULATORS` in
`calculators/__init__.py`. The CLI then has `flow-props <name> init` and `run`.

## Design Intent

The package is deliberately lightweight.

- it does not own CFD file I/O
- it assumes `cfd-io` has already constructed a dataset
- it focuses on profile-based property extraction rather than general field operations

That separation keeps `flow_props` easy to reason about and consistent with the rest of the workspace.
