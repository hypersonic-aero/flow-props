# Output Formats

Tables are written as Tecplot ASCII (`.dat`), one row per station; the shock curve is a
CSV.

## bl

`bl_properties.dat`: one row per station with

- `x` (wall), `delta`, `delta_star`, `theta`, `H`
- `uvel_edge`, `temp_edge`, `dens_edge`, `mach_edge`, `twte` (wall-to-edge temperature ratio)

## entropy-layer

`entropy_layer.dat`: the wall point `x`, `y`, `z` of each station, `station` and
`delta_entropy`.

## profiles

`profiles.dat`: one Tecplot zone per station with `x`, `y`, `z`, `eta`, `station` and one
column per flow variable of the solution (plus `u_tangent` and `u_normal`, the velocity
in the local wall frame).

## shock

- `shock_cfd.csv`: `x,y` header, then one shock point per grid line (smoothed, unless
  `smoothing_length = 0`). This is the file grid-generator's `[shock] curve` reads. On a
  symmetry line (grid line `i = 0` on the axis `y = 0`), a curve from cell values starts
  half a cell above the axis; it is continued there by symmetry ($x = a + b y^2$ through
  its first points), so the file starts with the point `(a, 0)`.
- `shock_report.dat`: the shock location for Tecplot, one I-ordered zone with one row per
  grid line with a shock. `x`, `y` are the shock points as found, so loading the file
  over the flow field draws the shock; then the wall point `x_wall`, `y_wall`, `station`,
  the shock point's fractional grid index `j_shock`, its outer edge `j_foot` (where the
  quantity crosses 10 % of the jump), the `distance` from the wall along
  the grid line, the captured `thickness_cells`, `detected_by` (which detector found
  the line's shock: 1 walk, 2 indicator, 3 both), the point as written to the curve
  `x_smooth`, `y_smooth`, `j_smooth` and how far smoothing moved it, `smoothing_cells`
  (0 with `smoothing_length = 0`), and last, when a shock line is known, `offset_cells`
  (`j_shock - shock_line_j`; positive means the solution's shock lies outside the grid's
  shock line). The columns keep this order whatever the options; new ones go at the
  end. Scripts should still read them by the `VARIABLES` names.
