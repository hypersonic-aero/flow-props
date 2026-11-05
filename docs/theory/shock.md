# Shock Extraction

`flow-props shock` finds the shock of a CFD solution on every wall-normal grid line. Each
line runs from the wall ($j = 0$) to the outer boundary, which must lie in the freestream.
A shock-capturing solver smears the shock over a few cells, so along the line the
quantity $q$ (pressure by default, or density) climbs from its freestream value to the
post-shock value over a short stretch.

## Finding the Jump

Walking inward from the line's outer end, where $q_\infty$ is $q$ at the outer end, a
detector proposes where a jump starts (`method`):

- **walk** (default): the first point where $q > (1 + \text{min\_rise})\, q_\infty$;
- **indicator**: the first point whose smoothness indicator reaches
  `indicator_threshold` (see [Smoothness Indicator](#smoothness-indicator));
- **hybrid**: both detectors, and the outermost jump either finds, since the shock is
  the first jump from outside. The report says which detector found each line.

From the proposed start, the steepest cell within `max_thickness` cells inward is the
jump's core. Walking on inward, the rise per cell shrinks across the jump's rounded
shoulder; once past the jump's center (the rise below half the steepest), the jump ends
where the rise shrinks by less than 1 % of the steepest from one cell to the next: at the
knee into the post-shock plateau or the top of an overshoot. $q_t$ is the post-shock
value there. The profile is flat
there, so $q_t$ barely changes when the shock drifts by a cell from one grid line to the
next. (Ending the jump on the shoulder instead makes $q_t$, and with it the shock point,
jump at every such drift: a staircase along the shock.)

Because the walk starts outside, the first jump it meets is the shock: the boundary layer,
the entropy layer and any inner compression are never reached.

## Placing the Shock Point

The shock point is where $q$ crosses

$$
q_\text{level} = q_\infty + \text{level}\,(q_t - q_\infty),
$$

interpolated linearly between the two grid points around the crossing. `level = 0.5`
puts it in the middle of the captured jump, which is where a grid line should sit; values
toward 0 move it to the freestream side, toward 1 to the post-shock side.

Only this crossing places the point; the foot and tail just bracket the jump. The point
therefore doesn't depend on `min_rise`, and is placed the same way for a shock of 40 times
freestream pressure at the nose and a weak one far downstream. On synthetic captured
shocks (jumps 0.3 to 40 times freestream) it lands within 0.1 cell of the jump's center
for shocks captured over up to about 5 cells; with a post-shock gradient, a shock smeared
over more cells comes out inward by up to about 2 % of its thickness. As a shock drifts
across the cells from one grid line to the next, its point moves smoothly; what remains
is a small error that repeats with the shock's position inside a cell (about 0.03 to
0.07 cell for shocks captured over 1 to 2 cells), which the smoothing removes.

## Thickness and Smooth Compressions

The captured thickness is the distance, in cells, between the crossings of 10 % and 90 %
of the jump. A rise spread over more than `max_thickness` cells (default 20) is a smooth
compression, not a shock: for example the pressure rise toward the stagnation point on a
line that ends inside the shock layer. Such lines are reported as having no shock.

## Comparison with a Shock-Line Grid

On a shock-line grid the shock is meant to lie on grid line `shock_line_j`. The report
gives each line's offset $j_\text{shock} - j_s$ in cells, and the run summary the range of
offsets and the number of lines more than a quarter cell off.

## Checks in the Report

Both detectors run on every line, whichever the method, so each result can be checked
against the other. The run summary warns when

- the shock's outer edge (where $q$ crosses 10 % of the jump) lies within 5 cells of the
  outer boundary: too little freestream outside it to read $q_\infty$ from, or a
  disturbance ahead of the shock was taken for it;
- the walk and the indicator differ by more than 3 cells on a line: one of them found
  something else. A smooth compression ahead of the shock (e.g. freestream compressed
  against a bent symmetry line) is met first by the walk, while the indicator, which
  only flags sharp jumps, finds the shock.

It also counts how often the shock's outer edge changes grid row along the body and, on
a shock-line grid, how often it crosses the shock line `shock_line_j`. A captured shock
that moves between rows makes a solver's wall quantities wave; on a converged
shock-line grid the edge stays on one side of the shock line.

## Solutions at Cell Centres

A finite-volume solution holds cell values. Read through cfd-io as fields at the cell
centres of the vertex grid (association `cell`: one point fewer than the grid in each
direction), they are put on a grid of the cell centres, each the mean of its cell's
corners, and the shock is found along the lines through them. Those lines lie half a
cell from the vertex grid's lines, so:

- the shock line `shock_line_j` counts vertices, and cell $j$ lies at $j + 0.5$ on the
  vertex grid; the report's `j_shock` and the offsets use that, and the run summary
  says the solution was at cell centres;
- the first line runs half a cell off the symmetry axis, so the curve starts there;
  grid-generator continues a curve that starts slightly off the axis to it.

A solution whose grid is the cell centres themselves (stored as an ordinary grid)
can't be told from vertex data; write cell values on the vertex grid instead.

## Smoothing the Curve

The extracted curve becomes a grid line of the next grid, so line-to-line jitter of the
points would become wiggles in the grid. With `smoothing_length = a` (default 0.5), each
point is moved onto a quadratic fitted, by least squares, through the shock points within
a stretch of the shock $a\,d$ long, with $d$ the local wall-to-shock distance, and the
curve that is written is the smoothed one.

The stretch scales with $d$ because the shock bends on scales of $d$: tightly around a
blunt nose, gently far downstream. A fixed number of grid lines fails one way or the
other: a precursor grid's lines are dense on the body compared with a shock-line grid's
cells, so a short window leaves wiggles a few millimetres long that the new grid resolves
(its shock line zig-zags), and a long one flattens the nose.

The fit is made in the shock's own local frame: the tangent along the chord of the window
and the normal to it. The point then moves along its own grid line onto the fitted
quadratic, so there is still one point per grid line. Fitting the distance along the grid
lines instead would be wrong where the body's curvature jumps: at a nose-body tangency
the wall normals stop turning, and the distance along them bends although the shock is
smooth. A straight shock over such lines stays exactly where it is.

The report keeps the raw and the smoothed points, and the run summary says how far the
smoothing moved them, in cells; `smoothing_length = 0` writes the points as found.

## Smoothness Indicator

A WENO smoothness indicator measures how far a stencil is from smooth. flow-props uses the
central five-point measure of ninth-order WENO (Balsara and Shu, J. Comput. Phys. 160,
2000) divided by $5040\, q_\infty^2$, which on a smooth ramp equals
$(\Delta q_\text{cell} / q_\infty)^2$, with $\Delta q_\text{cell}$ the rise per cell; a jump
captured over fewer cells scores higher. The default threshold 0.0025 is thus a rise of
5 % of freestream per cell. Dividing by $q_\infty^2$ makes one threshold work for any
freestream pressure.

Its peak scales with $(\text{jump} / \text{thickness})^2$, so the threshold that finds
the shock depends on how strong and how smeared it is: on synthetic captured shocks a
jump of $0.3\, q_\infty$ peaks at about 0.06 when captured over 2 cells and at about
0.0015 over 9 cells. Along a body, from the strong nose shock to a weak, smeared shock
downstream, a single threshold can miss part of the shock; lower it, or use the walk (its
`min_rise` is a total rise, whatever the width) or the hybrid. Where the walk and the
indicator find the same jump, the shock point is the same: the level crossing places it.
