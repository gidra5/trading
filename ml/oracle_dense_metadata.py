"""Reuse a map's prepared grid metadata in root search and error recording.

The original grid methods rebuild arrays on every call. A read-only view
supplies the already constructed arrays while executing those same methods,
preserving their random draw order, tie breaking and arithmetic. Sweeps and
archives keep the original grid and operator identity.
"""
from dataclasses import dataclass
from oracle_account_grid import AccountGrid
from oracle_dense_observe_batch import observe_batched
from oracle_dense_value_map import DenseValueMap


@dataclass(frozen=True)
class _PreparedGrid:
    grid: AccountGrid
    metadata: dict

    def __getattr__(self, name):
        return getattr(self.grid, name)

    def arrays(self):
        return self.metadata

    sample_cell = AccountGrid.sample_cell
    canonical_cell = AccountGrid.canonical_cell


class _PreparedMap(DenseValueMap):
    observe = observe_batched


def prepared(map):
    # A changed view shares the value field, account metadata and error builder.
    # The original map's grid is never replaced, even temporarily.
    # A class method avoids a self-bound callback cycle retaining mmap handles.
    view = _PreparedMap.__new__(_PreparedMap)
    view.__dict__.update(map.__dict__)
    view.grid = _PreparedGrid(map.grid, map.a)
    return view
