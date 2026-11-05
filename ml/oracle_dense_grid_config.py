"""Explicit nested exposure refinement for dense-map experiment drivers."""
import math
from dataclasses import replace
import numpy as np
from oracle_account_grid import AccountGrid


def exposure_axis(step=None):
    if step is None:return AccountGrid.exposure_points
    if not math.isfinite(step) or step<=0:raise ValueError('Exposure step must be positive and finite')
    if step<.0005:raise ValueError('At most 10000 exposure intervals are supported')
    intervals=round(5/step)
    if intervals<2 or intervals>10000 or intervals%2 or not math.isclose(intervals*step,5.,rel_tol=0,abs_tol=1e-10):
        raise ValueError('Exposure step must partition [-2.5,2.5], include zero, and use at most 10000 intervals')
    points=tuple(float(x) for x in np.linspace(-2.5,2.5,intervals+1))
    if any(min(abs(x-y) for y in points)>1e-10 for x in AccountGrid.exposure_points):
        raise ValueError('Exposure refinement must retain all original exposure targets')
    return points


def future_candidate_count(grid, requested=None):
    # The sampler always evaluates the entire zero-order exposure axis.
    # Preserve nine additional order-bearing proposals as the axis grows.
    minimum=len(grid.exposure_points)
    count=minimum+(9 if grid.max_orders else 0) if requested is None else requested
    if not isinstance(count,(int,np.integer)) or isinstance(count,(bool,np.bool_)):
        raise ValueError('Candidate budget must be an integer')
    if count<minimum or (grid.max_orders and count==minimum):
        raise ValueError('Candidate budget must cover the exposure axis and include order-bearing proposals')
    return int(count)


def replay_grid(source,parent=None,exposure_step=None,candidates=None):
    """Resolve a new grid or retain the saved grid when resuming a replay."""
    saved=parent if parent is not None else source
    config={**saved['grid'],'exposure_points':tuple(saved['grid']['exposure_points']),
        'wealth_points':tuple(saved['grid']['wealth_points'])}
    grid=AccountGrid(**config)
    if exposure_step is not None:grid=replace(grid,exposure_points=exposure_axis(exposure_step))
    requested=candidates
    if requested is None and exposure_step is None:requested=saved['arguments']['candidates']
    return grid,future_candidate_count(grid,requested)
