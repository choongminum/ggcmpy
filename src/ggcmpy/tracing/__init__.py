"""
tracing.py

Particle tracing
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from scipy import constants  # type: ignore[import-untyped]

from ggcmpy import _jrrle  # type: ignore[attr-defined]

from .legacy import BorisIntegrator_cxx as BorisIntegrator_cxx
from .legacy import BorisIntegrator_python as BorisIntegrator_python
from .legacy import BorisIntegratorBase
from .legacy import FieldInterpolator_f2py as FieldInterpolator_f2py
from .legacy import FieldInterpolatorYee_f2py as FieldInterpolatorYee_f2py

# pylint: disable=C0103,I1101


def make_vector_field(
    grid: Sequence[tuple[str, tuple[str, ...]]],
    coords: dict[str, np.ndarray],
    vector_field: Callable[[np.ndarray], np.ndarray],
) -> dict[str, Any]:
    flds = {}
    for d, (fld_name, dims) in enumerate(grid):
        crds = [coords[dim] for dim in dims]
        fld = np.empty(tuple(len(c) for c in crds))
        for i in range(fld.shape[0]):
            for j in range(fld.shape[1]):
                for k in range(fld.shape[2]):
                    val = vector_field(np.array([crds[0][i], crds[1][j], crds[2][k]]))
                    fld[i, j, k] = val[d]
        # fld = xr.apply_ufunc(lambda x, y, z: vector_field(np.array([x, y, z]))[d], *crds, vectorize=True)
        flds[fld_name] = (dims, fld)

    return flds


def discretize_emfields_cc(coords: dict[str, np.ndarray], fields: Any) -> xr.Dataset:
    b_grid = [("bx", ("x", "y", "z")), ("by", ("x", "y", "z")), ("bz", ("x", "y", "z"))]
    e_grid = [("ex", ("x", "y", "z")), ("ey", ("x", "y", "z")), ("ez", ("x", "y", "z"))]

    return xr.Dataset(
        make_vector_field(b_grid, coords, fields.B)
        | make_vector_field(e_grid, coords, fields.E),
        coords=coords,
    )


def discretize_emfields_yee(coords: dict[str, np.ndarray], fields: Any) -> xr.Dataset:
    b1_grid = [
        ("bx1", ("x_nc", "y", "z")),
        ("by1", ("x", "y_nc", "z")),
        ("bz1", ("x", "y", "z_nc")),
    ]
    e1_grid = [
        ("eflx", ("x", "y_nc", "z_nc")),
        ("efly", ("x_nc", "y", "z_nc")),
        ("eflz", ("x_nc", "y_nc", "z")),
    ]

    return xr.Dataset(
        make_vector_field(b1_grid, coords, fields.B)
        | make_vector_field(e1_grid, coords, fields.E),
        coords=coords,
    )


class BorisIntegrator_f2py(BorisIntegratorBase):
    """
    BorisIntegrator_f2py provides an interface for integrating charged particle trajectories
    using the Boris algorithm, with field interpolation via f2py-wrapped Fortran routines.

    Args:
        ds (xr.Dataset or emfields.interpolator_python or emfields.interpolator_yee_python):
            The dataset containing electromagnetic field data, or a pre-initialized field interpolator.
        q (float, optional):
            Particle charge in Coulombs. Defaults to the elementary charge (constants.e).
        m (float, optional):
            Particle mass in kilograms. Defaults to the electron mass (constants.m_e).

    Attributes:
        q (float): Particle charge.
        m (float): Particle mass.

    Methods:
        integrate(x0, v0, t_final, dt) -> pd.DataFrame:
            Integrates the particle trajectory using the Boris algorithm.
    """

    def __init__(self, df, q=constants.e, m=constants.m_e) -> None:
        _jrrle.particle_tracing_f2py.boris_init(q, m)
        if isinstance(df, xr.Dataset):
            fields = FieldInterpolatorYee_f2py(df)
        else:
            assert isinstance(df, FieldInterpolatorYee_f2py)
            fields = df

        super().__init__(fields, q, m)

    def integrate(self, x0, u0, t_final, dt_max=1.0, dt_max_gyro=0.1) -> pd.DataFrame:
        n_steps = int(t_final / dt_max) + 2  # add some extra space for round-off issues
        data = np.zeros((7, n_steps), dtype=np.float32, order="F")
        n_out = _jrrle.particle_tracing_f2py.boris_integrate(
            x0, u0, t_final, dt_max, dt_max_gyro, data
        )
        return pd.DataFrame(
            data.T[:n_out], columns=["time", "x", "y", "z", "ux", "uy", "uz"]
        )


BorisIntegrator = BorisIntegrator_python
