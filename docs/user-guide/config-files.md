# Config Files

Each calculation reads its own TOML file. `flow-props <calculation> init` writes a
template with every key documented; edit it, then run `flow-props <calculation> run`.

## Shared Parts

Every config names the CFD solution at the top level:

```toml
# CFD solution file.
fname = "base_flow.hdf5"
# For split file formats, uncomment and set the separate grid file.
# gname = "grid.s8"
```

`bl`, `entropy-layer` and `profiles` select stations (i-indices along the wall) with a
`[stations]` table: leave it empty for every station, or use exactly one mode.

```toml
[stations]
# Mode 1: wall x from x_s to x_e, step dx (nearest station to each x)
x_s = 0.05
x_e = 0.50
dx  = 0.05
# Mode 2: i_s, i_e, di (inclusive)
# Mode 3: list = [10, 50, 100]
```

A solution with cell values (a finite-volume solver's, read by cfd-io as fields at the
cell centres of the vertex grid) is handled by every calculation: the profiles run
through the cell centres, starting at the first one, half a cell off the wall.

`bl` (with the `htot_derivative` criterion) and `entropy-layer` need $\gamma$ and $R$.
They come from the solution's HDF5 attributes (`gamma`, `r_gas`), then a
`flow_conditions.json` next to the solution file, and can be overridden:

```toml
[gas]
gamma        = 1.4
gas_constant = 287.05
```

## Calculation Tables

Each calculation has one table named after it:

=== "bl.toml"

	```toml
	[bl]
	criterion = "velocity_ratio"   # or htot_derivative
	threshold = 0.99
	output = "bl_properties.dat"
	```

=== "entropy-layer.toml"

	```toml
	[entropy_layer]
	threshold = 0.25
	# temp_edge = 220.0        # reference state; default: the profile's outermost point
	# pressure_edge = 2000.0
	output = "entropy_layer.dat"
	```

=== "profiles.toml"

	```toml
	[profiles]
	method = "grid_line"           # or wall_normal_interp
	# n_eta = 200                  # wall_normal_interp only
	# eta_max = 0.01               # wall_normal_interp only
	output = "profiles.dat"
	```

=== "shock.toml"

	```toml
	[shock]
	quantity = "pressure"          # or density
	level = 0.5                    # middle of the captured jump
	method = "hybrid"              # hybrid (both), walk, or indicator (WENO smoothness)
	min_rise = 0.05                # walk: 5 % above freestream marks the shock
	indicator_threshold = 0.0025   # indicator: (rise per cell / q_inf)^2 on a ramp
	max_thickness = 20.0           # wider rises are smooth compressions, not shocks
	# shock_line_j = 160           # shock-line grids: compare with this grid line
	smoothing_length = 0.5         # fit length in wall-to-shock distances; 0: as found
	output = "shock_cfd.csv"
	report = "shock_report.dat"
	```

Unknown keys are errors, so a misspelled key doesn't silently fall back to a default.
