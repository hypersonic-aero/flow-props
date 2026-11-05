"""Shock extraction from a CFD solution: one shock point per wall-normal grid line.

Each grid line runs from the wall (j = 0) to the outer boundary, which lies in the
freestream. A shock-capturing solver smears the shock over a few cells, so along the
line the flow quantity q (pressure or density) climbs from its freestream value q_inf
(q at the line's outer end) to the post-shock value over a short stretch.

1. Detection: walking inward from the outer end, a detector proposes where a jump starts.
   - walk: the first point where q exceeds (1 + min_rise) q_inf;
   - indicator: the first point whose smoothness indicator reaches indicator_threshold.
     The indicator is the central five-point smoothness measure of ninth-order WENO
     (Balsara and Shu, J. Comput. Phys. 160, 2000) divided by 5040 q_inf^2; on a smooth
     ramp it equals (rise per cell / q_inf)^2, and a jump captured over fewer cells
     scores higher;
   - hybrid: both, and the outermost jump either finds (the shock is the first jump
     from outside).
2. Jump: from the proposed start, the steepest cell within max_thickness cells inward is
   the jump's core. Walking on inward, the rise per cell shrinks across the jump's
   rounded shoulder; once past the jump's center (the rise below half the steepest), the
   jump ends (its tail) where the rise shrinks by less than 1 % of the steepest from one
   cell to the next: at the knee into the post-shock plateau, or the top of an
   overshoot. q_tail is the post-shock value there. The profile has leveled off there,
   so q_tail barely changes when the shock drifts by a cell from one grid line to the
   next; a tail on the shoulder would make the shock point jump at every such drift
   (a staircase along the shock).
3. Shock point: where q crosses q_inf + level (q_tail - q_inf), interpolated linearly
   between the two grid points around the crossing. level = 0.5 puts the point in the
   middle of the captured jump, which is where a grid line should sit.

Only the level crossing places the point; the detectors just find the jump. Where both
find the same jump, the point is the same, and it is placed the same way for a shock 40
times freestream pressure at the nose and a weak one far downstream.

The captured shock's thickness is the distance, in cells, between the crossings of 10 %
and 90 % of the jump. A shock is captured over a few cells; a rise spread over more than
max_thickness cells is a smooth compression (e.g. toward the stagnation point on a line
that ends inside the shock layer), not a shock. Pressure is the default quantity: it is
nearly constant across the boundary layer and the entropy layer, so neither can be
mistaken for the shock (and the detection stops at the first jump from outside anyway).

Before the curve becomes a grid line it can be smoothed (smooth_shock): each point moves,
along its own grid line, onto a quadratic fitted, in the shock's own frame, through the
shock points within a stretch of a given number of local wall-to-shock distances.
"""

# --------------------------------------------------
# load necessary modules
# --------------------------------------------------
from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from cfd_io.dataset import Dataset, StructuredGrid
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# --------------------------------------------------
# constants
# --------------------------------------------------

# past the jump's center: the rise per cell has fallen below this fraction of the steepest
CENTER_FRACTION = 0.5

# the jump has leveled off where its rise shrinks by less than this fraction of the
# steepest rise from one cell to the next
FLATTENING = 0.01

# levels whose crossings measure the captured shock's thickness
THICKNESS_LEVELS = (0.1, 0.9)

# field names of each quantity, in lookup order (cfd-io's canonical name first)
QUANTITY_FIELDS: dict[str, tuple[str, ...]] = {
    "pressure": ("pres", "pressure", "p"),
    "density": ("dens", "density", "rho"),
}

# which detector found a line's shock (codes add up when both found it)
DETECTED_BY_WALK = 1
DETECTED_BY_INDICATOR = 2

# fewest shock points in one smoothing fit (a quadratic through 3 would not smooth)
MIN_FIT_POINTS = 5

# scale of the smoothness measure's integer coefficients (a ramp of slope s gives 5040 s^2)
INDICATOR_SCALE = 5040.0


# --------------------------------------------------
# results
# --------------------------------------------------


@dataclass(slots=True, frozen=True)
class ShockCrossing:
    """The shock on one grid line, in grid indices along the line (j = 0 at the wall)."""

    # fractional j of the shock point (at the requested level)
    j: float
    # last grid point of the jump on the post-shock side (q there is the post-shock value)
    j_tail: int
    # the shock's outer edge: fractional j where q crosses 10 % of the jump
    j_foot: float
    # captured shock thickness: from 10 % to 90 % of the jump, in cells
    thickness_cells: float
    # which detector found it: DETECTED_BY_WALK, DETECTED_BY_INDICATOR, or their sum
    detected_by: int
    # each detector's own shock point, whatever the method (nan: it found none); a
    # cross-check: far apart, one of them took something else for the shock
    j_walk: float = float("nan")
    j_indicator: float = float("nan")


@dataclass(slots=True)
class ShockExtraction:
    """The shock on every grid line where one was found, in i order."""

    # i of each grid line with a shock
    station: NDArray[np.int64]
    # shock point [m]
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    # fractional j of the shock point
    j: NDArray[np.float64]
    # distance from the wall along the grid line [m]
    distance: NDArray[np.float64]
    # captured shock thickness (10 % to 90 % of the jump) in cells
    thickness_cells: NDArray[np.float64]
    # which detector found each line's shock (see ShockCrossing.detected_by)
    detected_by: NDArray[np.int64]
    # the shock's outer edge (10 % of the jump), fractional j
    j_foot: NDArray[np.float64]
    # each detector's own shock point, fractional j (nan: it found none on that line)
    j_walk: NDArray[np.float64]
    j_indicator: NDArray[np.float64]
    # i of the grid lines without a shock
    stations_without_shock: list[int]


@dataclass(slots=True, frozen=True)
class _Detection:
    """What one detector proposes on a line."""

    # grid points where a jump may start, from outside in
    starts: list[int]
    # least post-shock value that makes a jump the detector's shock
    least_tail_value: float
    # code of the detector (DETECTED_BY_WALK or DETECTED_BY_INDICATOR)
    code: int


# --------------------------------------------------
# public API
# --------------------------------------------------


def smoothness_indicator(values: NDArray[np.float64], reference: float) -> NDArray[np.float64]:
    """Smoothness indicator along a line: how far each five-point stencil is from smooth.

    The central five-point smoothness measure of ninth-order WENO (Balsara and Shu,
    J. Comput. Phys. 160, 2000), divided by 5040 reference^2. On a smooth ramp it equals
    (rise per cell / reference)^2; a jump captured over fewer cells scores higher.

    Args:
        values: Pressure or density along the line.
        reference: Value to normalize by (the freestream value).

    Returns:
        The indicator at every point (0 at the two points at each end, which have no
        centered stencil).
    """

    # the five stencil values around every interior point
    values = np.asarray(values, dtype=float)
    f_m2 = values[:-4]
    f_m1 = values[1:-3]
    f_0 = values[2:-2]
    f_p1 = values[3:-1]
    f_p2 = values[4:]

    # smoothness measure of the centered stencil (integer coefficients, scale 5040)
    row_m2 = f_m2 * (
        6908.0 * f_m2 - 51001.0 * f_m1 + 67923.0 * f_0 - 38947.0 * f_p1 + 8209.0 * f_p2
    )
    row_m1 = f_m1 * (104963.0 * f_m1 - 299076.0 * f_0 + 179098.0 * f_p1 - 38947.0 * f_p2)
    row_0 = f_0 * (231153.0 * f_0 - 299076.0 * f_p1 + 67923.0 * f_p2)
    row_p1 = f_p1 * (104963.0 * f_p1 - 51001.0 * f_p2)
    row_p2 = 6908.0 * f_p2 * f_p2
    measure = row_m2 + row_m1 + row_0 + row_p1 + row_p2

    # normalize; the end points have no centered stencil
    indicator = np.zeros(values.size)
    indicator[2:-2] = measure / (INDICATOR_SCALE * reference**2)

    return indicator


def find_shock_on_line(
    values: NDArray[np.float64],
    level: float = 0.5,
    min_rise: float = 0.05,
    max_thickness: float = 20.0,
    method: str = "walk",
    indicator_threshold: float = 0.0025,
) -> ShockCrossing | None:
    """Find the shock along one grid line.

    Args:
        values: Pressure or density along the line, from the wall (index 0) to the
            outer boundary (last index, in the freestream).
        level: Where in the jump the shock point sits: 0.5 is the middle; toward 0 is
            the freestream side, toward 1 the post-shock side.
        min_rise: walk: rise over the freestream value that marks the shock (0.05 = 5 %).
        max_thickness: Largest captured thickness (10 % to 90 % of the jump) in cells;
            a wider rise is a smooth compression, not a shock.
        method: "walk", "indicator" or "hybrid" (see the module docstring).
        indicator_threshold: indicator: smoothness indicator value that marks the shock.

    Returns:
        The crossing, or None if the method finds no jump that is a shock.

    Raises:
        ValueError: If the line has fewer than five points, the freestream value isn't
            positive, level isn't between 0 and 1, or the method is unknown.
    """

    # validate inputs
    values = np.asarray(values, dtype=float)
    if values.size < 5:
        raise ValueError("a grid line needs at least five points to find a shock")
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must lie between 0 and 1, got {level}")
    if method not in METHODS:
        raise ValueError(f"method must be one of: {', '.join(METHODS)}, got {method!r}")

    # freestream: the value at the line's outer end
    q_inf = float(values[-1])
    if not q_inf > 0.0:
        raise ValueError(f"the freestream value (outer end of the line) is {q_inf}; expected > 0")

    # every detector: its first jump from outside that is a shock (the method picks
    # among them; the others are kept as a cross-check)
    found: dict[str, ShockCrossing] = {}
    for name, detector in DETECTORS.items():
        detection = detector(values, q_inf, min_rise, indicator_threshold)
        crossing = _first_shock(values, detection, q_inf, level, max_thickness)
        if crossing is not None:
            found[name] = crossing

    # the method's detectors
    crossings = [found[name] for name in METHODS[method] if name in found]
    if not crossings:
        return None

    # the shock is the outermost jump; detectors that found the same jump share it
    outermost = max(crossings, key=lambda crossing: crossing.j)
    detected_by = 0
    for crossing in crossings:
        if abs(crossing.j - outermost.j) < 1.0:
            detected_by += crossing.detected_by

    shock = ShockCrossing(
        j=outermost.j,
        j_tail=outermost.j_tail,
        j_foot=outermost.j_foot,
        thickness_cells=outermost.thickness_cells,
        detected_by=detected_by,
        j_walk=found["walk"].j if "walk" in found else float("nan"),
        j_indicator=found["indicator"].j if "indicator" in found else float("nan"),
    )

    return shock


def extract_shock(
    ds: Dataset,
    quantity: str = "pressure",
    level: float = 0.5,
    min_rise: float = 0.05,
    max_thickness: float = 20.0,
    method: str = "walk",
    indicator_threshold: float = 0.0025,
) -> ShockExtraction:
    """Find the shock on every wall-normal grid line of a structured solution.

    Args:
        ds: Dataset with a StructuredGrid; j = 0 is the wall and the last j-line the
            outer boundary, in the freestream (cfd-io's canonical orientation).
        quantity: "pressure" or "density".
        level: Where in the jump the shock point sits (see find_shock_on_line).
        min_rise: walk: rise over the freestream value that marks the shock (0.05 = 5 %).
        max_thickness: Largest captured thickness in cells (see find_shock_on_line).
        method: "walk", "indicator" or "hybrid" (see the module docstring).
        indicator_threshold: indicator: smoothness indicator value that marks the shock.

    Returns:
        The shock points of every line that has one.

    Raises:
        TypeError: If the grid isn't structured.
        ValueError: If the quantity is unknown or missing, or no line has a shock.
    """
    from flow_props.profiles import extract_profiles

    # validate inputs
    if not isinstance(ds.grid, StructuredGrid):
        raise TypeError("shock extraction requires a StructuredGrid dataset")
    if quantity not in QUANTITY_FIELDS:
        raise ValueError(f"quantity must be one of: {', '.join(QUANTITY_FIELDS)}")

    # find the quantity's field
    candidates = QUANTITY_FIELDS[quantity]
    field_name = None
    for name in candidates:
        if name in ds.flow:
            field_name = name
            break
    if field_name is None:
        raise ValueError(
            f"the solution has no {quantity} field (looked for: {', '.join(candidates)})"
        )

    # every grid line, wall to outer boundary
    ni = ds.grid.shape[0]
    stations = list(range(ni))
    profiles = extract_profiles(ds, stations, method="grid_line")

    # initialize one list per output column
    found_stations: list[int] = []
    x_values: list[float] = []
    y_values: list[float] = []
    j_values: list[float] = []
    distance_values: list[float] = []
    thickness_values: list[float] = []
    detected_values: list[int] = []
    foot_values: list[float] = []
    walk_values: list[float] = []
    indicator_values: list[float] = []
    without_shock: list[int] = []

    # find the shock on each line
    for station, profile in zip(stations, profiles):
        crossing = find_shock_on_line(
            profile[field_name],
            level=level,
            min_rise=min_rise,
            max_thickness=max_thickness,
            method=method,
            indicator_threshold=indicator_threshold,
        )
        if crossing is None:
            without_shock.append(station)
            continue

        # interpolate the point between the two grid points around the crossing
        # (the crossing lies in [j, j + 1) with j at most the second-to-last point)
        j_below = int(crossing.j)
        fraction = crossing.j - j_below
        x_point = _interpolate(profile["x"], j_below, fraction)
        y_point = _interpolate(profile["y"], j_below, fraction)
        distance = _interpolate(profile["eta"], j_below, fraction)

        # collect
        found_stations.append(station)
        x_values.append(x_point)
        y_values.append(y_point)
        j_values.append(crossing.j)
        distance_values.append(distance)
        thickness_values.append(crossing.thickness_cells)
        detected_values.append(crossing.detected_by)
        foot_values.append(crossing.j_foot)
        walk_values.append(crossing.j_walk)
        indicator_values.append(crossing.j_indicator)

    # check: at least one line crosses the shock
    if not found_stations:
        raise ValueError(
            f"no shock found on any grid line (method {method}, {quantity}): no jump "
            f"narrower than {max_thickness:g} cells (does the outer boundary lie in the "
            "freestream?)"
        )

    # debug output for devs
    logger.debug("shock on %d of %d grid lines (method %s)", len(found_stations), ni, method)

    extraction = ShockExtraction(
        station=np.asarray(found_stations, dtype=np.int64),
        x=np.asarray(x_values, dtype=float),
        y=np.asarray(y_values, dtype=float),
        j=np.asarray(j_values, dtype=float),
        distance=np.asarray(distance_values, dtype=float),
        thickness_cells=np.asarray(thickness_values, dtype=float),
        detected_by=np.asarray(detected_values, dtype=np.int64),
        j_foot=np.asarray(foot_values, dtype=float),
        j_walk=np.asarray(walk_values, dtype=float),
        j_indicator=np.asarray(indicator_values, dtype=float),
        stations_without_shock=without_shock,
    )

    return extraction


def smooth_shock(ds: Dataset, extraction: ShockExtraction, length: float) -> ShockExtraction:
    """Fit a smooth curve through the shock points, keeping one point per grid line.

    Around each shock point, a quadratic is fitted by least squares through the shock
    points within a stretch of the shock `length` local wall-to-shock distances long
    (centered on the point, shifted inward at the curve's two ends, at least 5 points), in
    the shock's own local frame: the tangent along the chord of the stretch and the
    normal to it. The point then moves along its own grid line onto the fitted quadratic.

    The stretch scales with the wall-to-shock distance d because the shock bends on
    scales of d: tight around a blunt nose (d small), gently far downstream (d large). A
    fixed number of grid lines would either keep the downstream wiggles (lines are dense
    there compared with a new grid's cells) or flatten the nose.

    The fit uses the shock's frame, not the distance along the grid lines, because that
    distance bends where the body's curvature jumps (at a nose-body tangency the wall
    normals stop turning) even though the shock itself is smooth there.

    Args:
        ds: The dataset the shock was extracted from (for the grid lines).
        extraction: Result of extract_shock.
        length: Length of each fit along the shock, in local wall-to-shock distances.

    Returns:
        The smoothed shock: same stations and thickness, smoothed x, y, j and distance.

    Raises:
        ValueError: If the length isn't positive, or there are fewer than 5 shock points.
    """
    from flow_props.profiles.geometry import build_eta_from_path

    # validate inputs
    n_points = extraction.station.size
    if not length > 0.0:
        raise ValueError(f"the smoothing length must be positive, got {length}")
    if n_points < MIN_FIT_POINTS:
        raise ValueError(
            f"smoothing needs at least {MIN_FIT_POINTS} lines with a shock; found {n_points}"
        )

    # arc length along the shock, through the points as found
    arc = build_eta_from_path(extraction.x, extraction.y)

    # fit around each point, then move it along its grid line onto the fit
    smoothed_distance = np.zeros(n_points)
    for k in range(n_points):
        # stretch around k: length * d long, shifted inward at the ends, >= 5 points
        start, stop = _fit_stretch(arc, k, length * extraction.distance[k])

        # local frame: tangent along the stretch's chord, normal to it
        chord_x = extraction.x[stop - 1] - extraction.x[start]
        chord_y = extraction.y[stop - 1] - extraction.y[start]
        chord_length = np.hypot(chord_x, chord_y)
        tangent = np.array([chord_x, chord_y]) / chord_length
        normal = np.array([-tangent[1], tangent[0]])

        # stretch points in the local frame, origin at point k
        dx = extraction.x[start:stop] - extraction.x[k]
        dy = extraction.y[start:stop] - extraction.y[k]
        along_chord = dx * tangent[0] + dy * tangent[1]
        across_chord = dx * normal[0] + dy * normal[1]

        # fit: across = c0 + c1 along + c2 along^2
        c2, c1, c0 = np.polyfit(along_chord, across_chord, 2)

        # direction of the grid line at the point (the segment that holds it)
        station = extraction.station[k]
        j_below = int(extraction.j[k])
        segment_x = ds.grid.x[station, j_below + 1, 0] - ds.grid.x[station, j_below, 0]
        segment_y = ds.grid.y[station, j_below + 1, 0] - ds.grid.y[station, j_below, 0]
        segment_length = np.hypot(segment_x, segment_y)
        line_along = (segment_x * tangent[0] + segment_y * tangent[1]) / segment_length
        line_across = (segment_x * normal[0] + segment_y * normal[1]) / segment_length

        # move: the step along the grid line that lands on the fitted quadratic
        step = _step_onto_quadratic(c0, c1, c2, line_along, line_across)
        smoothed_distance[k] = extraction.distance[k] + step

    # put each smoothed point back on its grid line
    nj = ds.grid.shape[1]
    j_index = np.arange(nj, dtype=float)
    x_values = np.zeros(n_points)
    y_values = np.zeros(n_points)
    j_values = np.zeros(n_points)
    for k, station in enumerate(extraction.station):
        line_x = ds.grid.x[station, :, 0]
        line_y = ds.grid.y[station, :, 0]
        line_eta = build_eta_from_path(line_x, line_y)
        j_values[k] = np.interp(smoothed_distance[k], line_eta, j_index)
        x_values[k] = np.interp(smoothed_distance[k], line_eta, line_x)
        y_values[k] = np.interp(smoothed_distance[k], line_eta, line_y)

    # debug output for devs
    moved = np.abs(j_values - extraction.j)
    logger.debug("smoothed over %g shock distances: moved up to %.3f cells", length, moved.max())

    smoothed = ShockExtraction(
        station=extraction.station.copy(),
        x=x_values,
        y=y_values,
        j=j_values,
        distance=smoothed_distance,
        thickness_cells=extraction.thickness_cells.copy(),
        detected_by=extraction.detected_by.copy(),
        j_foot=extraction.j_foot.copy(),
        j_walk=extraction.j_walk.copy(),
        j_indicator=extraction.j_indicator.copy(),
        stations_without_shock=list(extraction.stations_without_shock),
    )

    return smoothed


def write_shock_curve(path: str | Path, extraction: ShockExtraction) -> Path:
    """Write the shock points as an x, y CSV (the curve file grid-generator's [shock] reads).

    Args:
        path: Output file (.csv).
        extraction: Result of extract_shock.

    Returns:
        Path to the written file.

    Raises:
        ValueError: If the file name doesn't end in .csv.
    """

    # convert to Path object and validate the suffix
    path = Path(path)
    if path.suffix.lower() != ".csv":
        raise ValueError(f"the shock curve is written as CSV; use a .csv file name, not {path}")

    # write: one header line, then x, y per grid line
    points = np.column_stack([extraction.x, extraction.y])
    np.savetxt(path, points, delimiter=",", header="x,y", comments="", fmt="%.10e")

    return path


# --------------------------------------------------
# detectors: where a jump may start on a line, from outside in
# --------------------------------------------------


def _detect_by_walk(
    values: NDArray[np.float64], q_inf: float, min_rise: float, indicator_threshold: float
) -> _Detection:
    """Points where q rises through (1 + min_rise) q_inf, walking inward."""

    # (the indicator threshold is the other detector's setting)
    del indicator_threshold

    # the first point above the threshold after one at or below it (outer neighbor)
    threshold = (1.0 + min_rise) * q_inf
    above = values > threshold
    starts = []
    for j in range(values.size - 2, -1, -1):
        if above[j] and not above[j + 1]:
            starts.append(j)

    detection = _Detection(starts=starts, least_tail_value=threshold, code=DETECTED_BY_WALK)

    return detection


def _detect_by_indicator(
    values: NDArray[np.float64], q_inf: float, min_rise: float, indicator_threshold: float
) -> _Detection:
    """Outer ends of the runs of points whose smoothness indicator reaches the threshold."""

    # (min_rise is the walk's setting)
    del min_rise

    # flagged points, and the first one of each run from outside
    flagged = smoothness_indicator(values, q_inf) >= indicator_threshold
    starts = []
    for j in range(values.size - 2, -1, -1):
        if flagged[j] and not flagged[j + 1]:
            starts.append(j)

    # any compression counts: the indicator itself decides what is sharp enough
    detection = _Detection(starts=starts, least_tail_value=q_inf, code=DETECTED_BY_INDICATOR)

    return detection


# a detector: (values, q_inf, min_rise, indicator_threshold) -> proposed jump starts
Detector = Callable[[NDArray[np.float64], float, float, float], _Detection]

# detector registry, and the detectors each method runs
DETECTORS: dict[str, Detector] = {
    "walk": _detect_by_walk,
    "indicator": _detect_by_indicator,
}
METHODS: dict[str, tuple[str, ...]] = {
    "walk": ("walk",),
    "indicator": ("indicator",),
    "hybrid": ("walk", "indicator"),
}


# --------------------------------------------------
# helpers
# --------------------------------------------------


def _first_shock(
    values: NDArray[np.float64],
    detection: _Detection,
    q_inf: float,
    level: float,
    max_thickness: float,
) -> ShockCrossing | None:
    """The first proposed start, from outside, whose jump is a shock.

    From a start, the steepest cell within max_thickness cells inward (and two outward,
    for a weak jump whose steepest part lies below the walk's threshold) is the jump's
    core; the tail is where, walking inward past the jump's center, the rise per cell
    stops shrinking noticeably (the knee into the post-shock plateau, or the top of an
    overshoot).
    """

    # one cell per grid point j: from j to j + 1 (rise toward the wall is positive)
    rises = values[:-1] - values[1:]
    window = int(np.ceil(max_thickness))

    for start in detection.starts:
        # core: the steepest cell around the start
        low = max(0, start - window)
        high = min(rises.size - 1, start + 2)
        j_peak = low + int(np.argmax(rises[low : high + 1]))
        steepest = float(rises[j_peak])
        if steepest <= 0.0:
            continue

        # tail: inward from the core, the rise per cell shrinks across the jump's shoulder;
        # past the center, where it stops shrinking noticeably, the profile has leveled
        # off (plateau or overshoot top)
        j_tail = 0
        for j in range(j_peak - 1, -1, -1):
            past_center = rises[j] < CENTER_FRACTION * steepest
            shrink = rises[j + 1] - rises[j]
            if past_center and shrink < FLATTENING * steepest:
                j_tail = j + 1
                break
        q_tail = float(values[j_tail])

        # check: a jump big enough for this detector
        if q_tail < detection.least_tail_value or q_tail <= q_inf:
            continue

        # thickness: from the 10 % crossing (outer) to the 90 % crossing (inner)
        j_outer = _level_crossing(values, j_tail, q_inf, q_tail, THICKNESS_LEVELS[0])
        j_inner = _level_crossing(values, j_tail, q_inf, q_tail, THICKNESS_LEVELS[1])
        thickness_cells = j_outer - j_inner

        # check: a shock, not a smooth compression
        if thickness_cells > max_thickness:
            continue

        # shock point: the level crossing between freestream and post-shock values
        j_shock = _level_crossing(values, j_tail, q_inf, q_tail, level)

        crossing = ShockCrossing(
            j=j_shock,
            j_tail=j_tail,
            j_foot=j_outer,
            thickness_cells=thickness_cells,
            detected_by=detection.code,
        )
        return crossing

    return None


def _level_crossing(
    values: NDArray[np.float64], j_tail: int, q_inf: float, q_tail: float, level: float
) -> float:
    """Fractional j where q falls through q_inf + level (q_tail - q_inf), from the tail out.

    Walking outward from the tail means walking through the jump toward the freestream,
    so the first crossing is the shock's (a bump farther out can't be mistaken for it).
    """

    # target value inside the jump
    target = q_inf + level * (q_tail - q_inf)

    # first cell, from the tail outward, where q drops below the target
    for j in range(j_tail, values.size - 1):
        if values[j] >= target > values[j + 1]:
            fraction = (values[j] - target) / (values[j] - values[j + 1])
            return j + float(fraction)

    # the outer end holds q_inf < target, so the loop always returns; guard anyway
    raise ValueError("no level crossing between the shock's tail and the outer end")


def _fit_stretch(arc: NDArray[np.float64], k: int, stretch: float) -> tuple[int, int]:
    """Index range [start, stop) of the shock points within a stretch around point k.

    The stretch is centered on k and shifted inward where it would run past the curve's
    ends; it holds at least MIN_FIT_POINTS points (the nearest ones, if it is shorter).
    """

    # centered stretch, shifted inward at the ends (keeping its length)
    low = arc[k] - 0.5 * stretch
    high = arc[k] + 0.5 * stretch
    if low < arc[0]:
        low, high = arc[0], arc[0] + stretch
    if high > arc[-1]:
        low, high = arc[-1] - stretch, arc[-1]
    start = int(np.searchsorted(arc, low, side="left"))
    stop = int(np.searchsorted(arc, high, side="right"))

    # at least MIN_FIT_POINTS points: the nearest ones around k
    n_points = arc.size
    if stop - start < MIN_FIT_POINTS:
        start = min(max(k - MIN_FIT_POINTS // 2, 0), n_points - MIN_FIT_POINTS)
        stop = start + MIN_FIT_POINTS

    return start, stop


def _step_onto_quadratic(
    c0: float, c1: float, c2: float, line_along: float, line_across: float
) -> float:
    """Step along a grid line, from a shock point, onto a quadratic fitted around it.

    In the local frame (origin at the point) the line is (step * a, step * b), with a and b
    the line direction's components along and across the chord, and the fit is
    across = c0 + c1 along + c2 along^2. Landing on the fit means

        c2 a^2 step^2 + (c1 a - b) step + c0 = 0,

    and the step is the root nearest zero (the point is already close to the fit).
    """

    # coefficients of the quadratic in the step
    quadratic = c2 * line_along**2
    linear = c1 * line_along - line_across
    constant = c0

    # nearly straight fit: the linear equation
    discriminant = linear**2 - 4.0 * quadratic * constant
    if abs(quadratic * constant) <= 1.0e-12 * linear**2 or discriminant < 0.0:
        return -constant / linear

    # root nearest zero, in the form that doesn't cancel digits
    root = -2.0 * constant / (linear + np.copysign(np.sqrt(discriminant), linear))

    return float(root)


def _interpolate(values: NDArray[np.float64], j_below: int, fraction: float) -> float:
    """Linear interpolation between grid points j_below and j_below + 1."""

    # weight the two neighbors
    value_below = float(values[j_below])
    value_above = float(values[j_below + 1])
    value = value_below + fraction * (value_above - value_below)

    return value
