"""Tests for flow_props.shock: the shock point on a grid line, and the shock curve."""

import numpy as np
import pytest
from cfd_io.dataset import Dataset, StructuredGrid

from flow_props.calculators.shock import _index_ranges
from flow_props.shock import (
    DETECTED_BY_INDICATOR,
    DETECTED_BY_WALK,
    ShockExtraction,
    extract_shock,
    find_shock_on_line,
    smooth_shock,
    smoothness_indicator,
    write_shock_curve,
)
from tests.conftest import shock_solution

# true shock position of the synthetic lines (fractional j)
SHOCK_J = 60.3

# allowed error of the shock point, in cells
TOLERANCE_CELLS = 0.15


def _tolerance(crossing) -> float:
    """Allowed error for a crossing: TOLERANCE_CELLS, or 2 % of a widely smeared jump.

    With a post-shock gradient, the post-shock value of a shock smeared over many cells is
    taken several cells behind it (where the profile levels off), which moves the middle
    of the jump inward by up to about 2 % of the captured thickness.
    """
    return max(TOLERANCE_CELLS, 0.02 * crossing.thickness_cells)


def _captured_shock(width: float, jump: float, slope: float = 0.004) -> np.ndarray:
    """A captured shock along one line: tanh jump at SHOCK_J, freestream 1 at the end."""
    j = np.arange(100, dtype=float)
    post_shock = 1.0 + jump + slope * jump * (SHOCK_J - j)
    shape = 0.5 * (1.0 - np.tanh((j - SHOCK_J) / width))
    values = 1.0 + shape * (post_shock - 1.0)
    values[-1] = 1.0
    return values


# --------------------------------------------------
# one grid line
# --------------------------------------------------
@pytest.mark.parametrize("width", [0.6, 1.0, 2.5, 4.0])
@pytest.mark.parametrize("jump", [0.3, 4.0, 40.0])
def test_shock_point_sits_in_the_middle_of_the_jump(width, jump):
    """Weak or strong, thin or smeared: level 0.5 lands on the jump's center."""
    crossing = find_shock_on_line(_captured_shock(width, jump))
    assert crossing is not None
    assert abs(crossing.j - SHOCK_J) < _tolerance(crossing)
    assert crossing.j_tail < crossing.j < crossing.j_foot


@pytest.mark.parametrize("width", [1.5, 2.5, 4.0])
def test_thickness_is_the_10_to_90_percent_width(width):
    """A tanh jump's 10-90 % width is 2 atanh(0.8) = 2.197 widths."""
    crossing = find_shock_on_line(_captured_shock(width, 4.0, slope=0.0))
    expected = 2.0 * np.arctanh(0.8) * width
    assert crossing.thickness_cells == pytest.approx(expected, rel=0.05)


@pytest.mark.parametrize("overshoot", [0.0, 0.03])
def test_no_staircase_when_the_shock_drifts_across_cells(overshoot):
    """From line to line the shock moves by a fraction of a cell; so must the point.

    A shock drifting through two cells in 80 steps: the point's error may vary smoothly
    (linear interpolation inside a cell), but must not jump when the jump's tail moves
    to the next grid point. With and without a post-shock overshoot.
    """
    j = np.arange(120.0)
    errors = []
    for center in np.linspace(60.0, 62.0, 81):
        shape = 0.5 * (1.0 - np.tanh((j - center) / 1.0))
        bump = overshoot * np.exp(-(((j - (center - 2.0)) / 1.2) ** 2))
        values = 1.0 + 0.8 * shape + bump
        values[-1] = 1.0
        errors.append(find_shock_on_line(values).j - center)

    jumps = np.abs(np.diff(errors))
    assert jumps.max() < 0.015
    assert np.abs(errors).max() < 0.05


def test_level_moves_the_point_through_the_jump():
    """Toward 0 the point moves to the freestream side (larger j), toward 1 inward."""
    values = _captured_shock(2.0, 4.0)
    outer = find_shock_on_line(values, level=0.2).j
    middle = find_shock_on_line(values, level=0.5).j
    inner = find_shock_on_line(values, level=0.8).j
    assert outer > middle > inner


def test_tail_stops_at_the_jump_despite_a_post_shock_gradient():
    """A steady rise behind the shock (toward the stagnation point) isn't part of the jump."""
    crossing = find_shock_on_line(_captured_shock(1.0, 4.0, slope=0.02))
    assert crossing.j_tail > SHOCK_J - 6.0


def test_the_first_jump_from_outside_is_the_shock():
    """A second, inner rise (e.g. a recompression) doesn't count."""
    values = _captured_shock(1.0, 4.0, slope=0.0)
    values[:20] += 3.0
    crossing = find_shock_on_line(values)
    assert abs(crossing.j - SHOCK_J) < TOLERANCE_CELLS


def test_no_shock_on_a_line_without_a_jump():
    """A line inside the shock layer (or in uniform flow) has no shock."""
    assert find_shock_on_line(np.full(50, 3.0)) is None
    assert find_shock_on_line(np.linspace(1.04, 1.0, 50)) is None


def test_a_smooth_compression_is_not_a_shock():
    """A rise spread over many cells (a line ending inside the shock layer) is rejected."""
    values = np.linspace(1.5, 1.0, 60)
    assert find_shock_on_line(values) is None
    assert find_shock_on_line(values, max_thickness=100.0) is not None


def test_min_rise_decides_what_counts_as_a_shock():
    """A 3 % compression is a shock for min_rise 0.01, not for the default 0.05."""
    values = _captured_shock(1.0, 0.03, slope=0.0)
    assert find_shock_on_line(values) is None
    assert find_shock_on_line(values, min_rise=0.01) is not None


def test_bad_line_inputs_raise():
    with pytest.raises(ValueError, match="five points"):
        find_shock_on_line(np.array([3.0, 3.0, 2.0, 1.0]))
    with pytest.raises(ValueError, match="level"):
        find_shock_on_line(_captured_shock(1.0, 4.0), level=1.0)
    with pytest.raises(ValueError, match="freestream"):
        find_shock_on_line(np.array([1.0, 0.8, 0.6, 0.5, 0.0]))
    with pytest.raises(ValueError, match="method"):
        find_shock_on_line(_captured_shock(1.0, 4.0), method="weno")


# --------------------------------------------------
# detection methods: walk, smoothness indicator, hybrid
# --------------------------------------------------
def test_smoothness_indicator_is_the_squared_rise_per_cell_on_a_ramp():
    """Divided by 5040 q_ref^2, the measure of a ramp is (slope / q_ref)^2; constant: 0."""
    ramp = 2.0 + 0.1 * np.arange(20.0)
    indicator = smoothness_indicator(ramp, reference=2.0)
    np.testing.assert_allclose(indicator[2:-2], (0.1 / 2.0) ** 2, rtol=1e-9)
    assert indicator[0] == indicator[-1] == 0.0
    assert np.all(smoothness_indicator(np.full(20, 3.0), reference=3.0) == 0.0)


def test_indicator_finds_a_sharp_shock_at_the_walks_point():
    """Both detectors find a sharp jump; the level crossing places the same point."""
    values = _captured_shock(1.0, 4.0)
    walk = find_shock_on_line(values, method="walk")
    indicator = find_shock_on_line(values, method="indicator")
    hybrid = find_shock_on_line(values, method="hybrid")

    assert indicator.j == pytest.approx(walk.j, abs=1e-12)
    assert walk.detected_by == DETECTED_BY_WALK
    assert indicator.detected_by == DETECTED_BY_INDICATOR
    assert hybrid.detected_by == DETECTED_BY_WALK + DETECTED_BY_INDICATOR


def test_indicator_misses_a_weak_smeared_shock_the_walk_finds():
    """A jump of 0.3 q_inf over about 9 cells stays below the indicator's threshold."""
    values = _captured_shock(4.0, 0.3)
    assert find_shock_on_line(values, method="indicator") is None

    hybrid = find_shock_on_line(values, method="hybrid")
    assert hybrid.detected_by == DETECTED_BY_WALK
    assert abs(hybrid.j - SHOCK_J) < _tolerance(hybrid)


def test_indicator_finds_a_sharp_shock_weaker_than_min_rise():
    """A thin 4 % jump: below the walk's 5 %, but sharp enough for the indicator."""
    values = _captured_shock(0.6, 0.04, slope=0.0)
    assert find_shock_on_line(values, method="walk") is None

    hybrid = find_shock_on_line(values, method="hybrid")
    assert hybrid.detected_by == DETECTED_BY_INDICATOR
    assert abs(hybrid.j - SHOCK_J) < TOLERANCE_CELLS


def test_hybrid_takes_the_outermost_jump():
    """A smeared outer shock (walk only) wins over a sharp inner jump (indicator only).

    The inner jump is ten times freestream, so that it is a shock on its own (measured from
    freestream, its 10 % crossing lies in it, not in the outer jump).
    """
    j = np.arange(100, dtype=float)
    outer = 0.5 * (1.0 - np.tanh((j - 70.3) / 4.0)) * 0.3
    inner = 0.5 * (1.0 - np.tanh((j - 30.3) / 0.6)) * 10.0
    values = 1.0 + outer + inner
    values[-1] = 1.0

    assert abs(find_shock_on_line(values, method="indicator").j - 30.3) < 0.5
    hybrid = find_shock_on_line(values, method="hybrid")
    assert abs(hybrid.j - 70.3) < TOLERANCE_CELLS
    assert hybrid.detected_by == DETECTED_BY_WALK


# --------------------------------------------------
# every grid line of a solution
# --------------------------------------------------
def test_extract_shock_finds_the_known_curve():
    """Every line of the synthetic solution: fractional j and the point on the line."""
    dataset, j_true = shock_solution()
    extraction = extract_shock(dataset)

    # every line has its shock, at the true j
    assert extraction.station.tolist() == list(range(j_true.size))
    assert extraction.stations_without_shock == []
    np.testing.assert_allclose(extraction.j, j_true, atol=TOLERANCE_CELLS)

    # the point lies on the line (straight lines at the wall x), at the stated distance
    np.testing.assert_allclose(extraction.x, dataset.grid.x[:, 0, 0], atol=1e-14)
    np.testing.assert_allclose(extraction.y, extraction.distance, atol=1e-14)


def test_lines_whose_outer_end_is_inside_the_shock_layer_are_skipped():
    """Only lines whose outer end lies in the freestream can show the shock."""
    dataset, _ = shock_solution()

    # lines 3-5 end behind the shock: raise their outer pressure to the post-shock value
    pres = dataset.flow["pres"].data
    pres[3:6, 45:, 0] = pres[3:6, 45, 0][:, None]
    extraction = extract_shock(dataset)

    assert extraction.stations_without_shock == [3, 4, 5]
    assert 3 not in extraction.station.tolist()


def test_extract_shock_by_density():
    """Density works like pressure (only the field name differs)."""
    dataset, j_true = shock_solution()
    dataset.flow["dens"] = dataset.flow.pop("pres")
    extraction = extract_shock(dataset, quantity="density")
    np.testing.assert_allclose(extraction.j, j_true, atol=TOLERANCE_CELLS)


def test_extract_shock_errors():
    dataset, _ = shock_solution()

    # missing field
    with pytest.raises(ValueError, match="no density field"):
        extract_shock(dataset, quantity="density")

    # no shock anywhere
    dataset.flow["pres"].data[:] = 1000.0
    with pytest.raises(ValueError, match="no shock found"):
        extract_shock(dataset)

    # not a structured grid
    with pytest.raises(TypeError):
        extract_shock(Dataset(grid=None, flow={}, attrs={}))


# --------------------------------------------------
# smoothing
# --------------------------------------------------
def _with_distance(dataset, extraction: ShockExtraction, distance: np.ndarray) -> ShockExtraction:
    """The same extraction with the points moved to the given distances (straight lines)."""
    return ShockExtraction(
        station=extraction.station,
        x=extraction.x,
        y=distance,
        j=np.interp(distance, dataset.grid.y[0, :, 0], np.arange(dataset.grid.shape[1])),
        distance=distance,
        thickness_cells=extraction.thickness_cells,
        detected_by=extraction.detected_by,
        stations_without_shock=[],
    )


def test_smoothing_keeps_a_straight_shock_over_lines_that_turn_at_a_junction():
    """Like a nose-body tangency: the line tilt changes slope at i = 20, the shock is straight.

    The distance along the lines then has a kink although the shock is a straight line;
    smoothing in the shock's own frame must leave every point where it is.
    """
    # grid: wall along x, straight lines tilted by an angle that turns until i = 20
    ni, nj, height = 41, 31, 0.01
    wall_x = np.linspace(0.0, 0.04, ni)
    tilt = np.deg2rad(np.minimum(np.arange(ni), 20) * 1.5)
    x = np.zeros((ni, nj, 1))
    y = np.zeros((ni, nj, 1))
    for j in range(nj):
        along = 2.0 * height * j / (nj - 1)
        x[:, j, 0] = wall_x + along * np.sin(tilt)
        y[:, j, 0] = along * np.cos(tilt)
    dataset = Dataset(grid=StructuredGrid(x=x, y=y, z=np.zeros_like(x)), flow={}, attrs={})

    # shock: the straight line y = height, crossed by every grid line
    distance = height / np.cos(tilt)
    extraction = ShockExtraction(
        station=np.arange(ni),
        x=wall_x + distance * np.sin(tilt),
        y=np.full(ni, height),
        j=distance / (2.0 * height) * (nj - 1),
        distance=distance,
        thickness_cells=np.full(ni, 2.0),
        detected_by=np.full(ni, DETECTED_BY_WALK),
        stations_without_shock=[],
    )

    smoothed = smooth_shock(dataset, extraction, length=1.0)

    np.testing.assert_allclose(smoothed.y, height, rtol=0, atol=1e-12 * height)
    np.testing.assert_allclose(smoothed.j, extraction.j, rtol=0, atol=1e-9)


def test_smoothing_averages_out_jitter_and_keeps_points_on_their_lines():
    """Line-to-line jitter shrinks; each smoothed point stays on its own grid line."""
    dataset, _ = shock_solution(ni=61)
    extraction = extract_shock(dataset)
    s = dataset.grid.x[:, 0, 0]
    smooth_curve = 0.02 + 0.01 * s
    jitter = 2.0e-4 * np.random.default_rng(3).standard_normal(s.size)
    noisy = _with_distance(dataset, extraction, smooth_curve + jitter)

    smoothed = smooth_shock(dataset, noisy, length=3.0)

    error_before = np.sqrt(np.mean((noisy.distance - smooth_curve) ** 2))
    error_after = np.sqrt(np.mean((smoothed.distance - smooth_curve) ** 2))
    assert error_after < 0.5 * error_before
    np.testing.assert_allclose(smoothed.x, dataset.grid.x[:, 0, 0], atol=1e-14)
    np.testing.assert_allclose(smoothed.y, smoothed.distance, atol=1e-14)


def test_smoothing_stretch_scales_with_the_shock_distance():
    """Wiggles shorter than the stretch go; a bend on the scale of d stays.

    The same jitter on a shock twice as far from the wall is smoothed over a stretch
    twice as long (in points), so it shrinks more; a shock bending over a few shock
    distances (like a bow shock around a nose) keeps its shape.
    """
    dataset, _ = shock_solution(ni=121)
    extraction = extract_shock(dataset)
    s = dataset.grid.x[:, 0, 0]
    jitter = 1.0e-4 * np.random.default_rng(5).standard_normal(s.size)

    # jitter around a near and a far shock: the far one is smoothed over more points
    near = smooth_shock(dataset, _with_distance(dataset, extraction, 0.01 + jitter), length=1.0)
    far = smooth_shock(dataset, _with_distance(dataset, extraction, 0.02 + jitter), length=1.0)
    assert np.std(far.distance - 0.02) < np.std(near.distance - 0.01)

    # a bend over a few shock distances survives a stretch of half a shock distance
    bend = 0.02 + 0.004 * np.tanh((s - 0.2) / 0.03)
    kept = smooth_shock(dataset, _with_distance(dataset, extraction, bend), length=0.5)
    np.testing.assert_allclose(kept.distance, bend, rtol=0, atol=2e-5)


def test_smoothing_checks():
    dataset, _ = shock_solution(ni=4)
    extraction = extract_shock(dataset)
    with pytest.raises(ValueError, match="positive"):
        smooth_shock(dataset, extraction, length=0.0)
    with pytest.raises(ValueError, match="at least 5 lines"):
        smooth_shock(dataset, extraction, length=1.0)


def test_shock_curve_csv_has_a_header_and_x_y_columns(tmp_path):
    """The layout grid-generator's [shock] curve reads: one header line, then x, y."""
    dataset, _ = shock_solution()
    extraction = extract_shock(dataset)
    path = write_shock_curve(tmp_path / "shock_cfd.csv", extraction)

    assert path.read_text().splitlines()[0] == "x,y"
    values = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    np.testing.assert_allclose(values[:, 0], extraction.x)
    np.testing.assert_allclose(values[:, 1], extraction.y)

    with pytest.raises(ValueError, match="csv"):
        write_shock_curve(tmp_path / "shock.dat", extraction)


def test_index_ranges_text():
    assert _index_ranges([0, 1, 2, 7, 9, 10]) == "0-2, 7, 9-10"
    assert _index_ranges([4]) == "4"
