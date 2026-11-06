# Basic Usage

## Read A Dataset

`flow_props` expects a dataset that has already been read by `cfd-io`.

```python
import cfd_io

dataset = cfd_io.read_file("solution.vtu")
```

For split solution/grid formats, provide the separate grid file through `cfd-io`.

## Extract Profiles

```python
from flow_props import extract_profiles

profiles = extract_profiles(dataset, stations=[10, 50, 100])
print(profiles[0].keys())
```

Each extracted profile includes:

- `x`, `y`
- `eta`
- one array per flow field available in the input dataset

For config-driven extraction, `flow-props bl init` creates `bl.toml`. Set
`fname` to a structured HDF5 file, then run `flow-props bl run`. To look at the
profiles behind the table, `flow-props profiles init` / `run` writes them, for the
same stations, as multi-zone Tecplot ASCII.
`cfd-io` handles file reading; `flow-props` does not convert CFD formats.
Leave `[stations]` empty to process all i-indices from 0 through `Nx - 1` with
step 1, or choose one station-selection mode to limit the run.

Only the `htot_derivative` criterion requires gas constants. The tool reads
`gamma` and `r_gas` from HDF5 attributes, fills missing values from an adjacent
`flow_conditions.json`, and allows optional `[gas]` overrides. The default
`velocity_ratio` criterion does not require those values.

## Compute Boundary-Layer Properties

```python
from flow_props import run_bl

bl_results = run_bl(
    dataset,
    stations=[10, 50, 100],
    criterion="velocity_ratio",
    threshold=0.99,
)
```

Available edge criteria currently include:

- `velocity_ratio`
- `enthalpy_ratio`
- `velocity_gradient`

## Compute Entropy-Layer Thickness

```python
from flow_props import run_entropy

entropy_results = run_entropy(
    dataset,
    stations=[10, 50, 100],
    threshold=0.25,
)
```

## Extract the Shock

```python
from flow_props import extract_shock, write_shock_curve

extraction = extract_shock(dataset, quantity="pressure", level=0.5)
write_shock_curve("shock_cfd.csv", extraction)
```

`extraction.j` holds the fractional grid index of the shock on each line, for comparing
with a shock-aligned grid's shock line.

## Compute Wall Quantities

```python
from flow_props import run_wall

wall_results = run_wall(
    dataset,
    mu_wall=1.8e-5,
    rho_inf=0.12,
    u_inf=2500.0,
    k_wall=0.03,
    cp=1005.0,
    T_wall=300.0,
    T_inf=220.0,
)
```

## Typical Workflow

In practice, a common pattern is:

```python
import cfd_io
from flow_props import run_bl, run_entropy, run_wall

dataset = cfd_io.read_file("solution.vtu")

bl_results = run_bl(dataset, stations=[20, 40, 60])
entropy_results = run_entropy(dataset, stations=[20, 40, 60])
wall_results = run_wall(dataset, mu_wall=1.8e-5)
```

That gives you station-wise layer quantities plus surface distributions from the same CFD input.
