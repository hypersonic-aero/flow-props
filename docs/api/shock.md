# Shock

Module: `flow_props.shock`

## Purpose

This module finds the shock on each wall-normal grid line of a structured solution and
writes the shock curve (see [Shock Extraction](../theory/shock.md) for the method).

## Main Functions

- `find_shock_on_line(values, level=0.5, min_rise=0.05, max_thickness=20.0, method="walk",
  indicator_threshold=0.0025)`: the shock on one line, as a `ShockCrossing` (fractional
  `j`, `j_tail`, the outer edge `j_foot`, `thickness_cells`, `detected_by`, and each
  detector's own point `j_walk`, `j_indicator`, whatever the method), or `None`
- `extract_shock(ds, quantity="pressure", ..., method="walk", indicator_threshold=0.0025)`:
  every line of a dataset, as a `ShockExtraction` (station, x, y, j, distance,
  thickness_cells, detected_by, and the stations without a shock)
- `smoothness_indicator(values, reference)`: the WENO smoothness indicator along a line,
  divided by 5040 reference^2
- `smooth_shock(ds, extraction, length)`: the same points moved, along their grid lines,
  onto local quadratics fitted in the shock's frame over `length` wall-to-shock distances
  (see
  [Smoothing the Curve](../theory/shock.md#smoothing-the-curve))
- `write_shock_curve(path, extraction)`: the x, y CSV that grid-generator's `[shock] curve`
  reads

## Example

```python
import cfd_io
from flow_props import extract_shock, smooth_shock, write_shock_curve

dataset = cfd_io.read_file("solution.h5")
extraction = extract_shock(dataset)
smoothed = smooth_shock(dataset, extraction, length=0.5)
write_shock_curve("shock_cfd.csv", smoothed)

print(extraction.thickness_cells.max(), extraction.stations_without_shock)
```

## Notes

- the dataset must use a `StructuredGrid` with $j = 0$ at the wall (cfd-io's canonical
  orientation) and the outer boundary in the freestream
- the quantity's field is looked up as `pres`, `pressure`, `p` (or `dens`, `density`, `rho`)
