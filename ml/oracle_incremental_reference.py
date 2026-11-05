"""Independent full finite-table grading; never updates the tested table."""
import time
import numpy as np
from oracle_dense_value_map import DenseValueMap


def full_reference(table,on_progress=None):
    started = time.perf_counter()
    exact = DenseValueMap(table.grid,table.cloud,table.rules,None)
    run = exact.sweep(table.seed,table.candidates,table.B,backend='cuda',on_progress=on_progress)
    maximum = 0.; square = 0.; count = 0; support = 0; false_finite = 0; missed = 0
    worst = None; layers = []
    # The root is evaluated directly at actual attainable post-states. Stored
    # depth zero is never queried by this planner and need not be refreshed.
    for depth in range(1,table.H):
        actual = table.field(depth).cpu().numpy(); expected = exact.values[depth]
        finite = np.isfinite(actual)&np.isfinite(expected)
        support += int(np.count_nonzero(np.isfinite(actual)!=np.isfinite(expected)))
        false_here = int(np.count_nonzero(np.isfinite(actual)&~np.isfinite(expected)))
        missed_here = int(np.count_nonzero(~np.isfinite(actual)&np.isfinite(expected)))
        false_finite += false_here; missed += missed_here
        delta = np.zeros_like(actual); np.subtract(actual,expected,out=delta,where=finite)
        error = delta[finite]
        largest = float(np.max(abs(error))) if len(error) else 0.
        if largest>maximum:
            maximum = largest
            index = np.unravel_index(np.argmax(np.where(finite,abs(delta),-np.inf)),actual.shape)
            worst = dict(depth=depth,wealthIndex=int(index[0]),accountCell=int(index[1]),marketNode=int(index[2]),
                actual=float(actual[index]),reference=float(expected[index]))
        square += float(np.square(error).sum()); count += len(error)
        layers.append(dict(depth=depth,maximumLogValueError=largest,falseFinite=false_here,missedFeasible=missed_here))
    stats = dict(seconds=time.perf_counter()-started,sweepSeconds=run['seconds'],
        maximumLogValueError=maximum,rmsLogValueError=float(np.sqrt(square/count)) if count else None,
        comparedFiniteCells=count,supportDisagreements=support,
        falseFinite=false_finite,missedFeasible=missed,worstFiniteCell=worst,layers=layers,
        depthRange=[1,table.H-1],scope='Stored continuation fields; direct root choices graded separately')
    return exact.values[min(1,table.H-1)],stats
