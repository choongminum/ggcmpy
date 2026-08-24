"""
legacy.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from ggcmpy import constants
from ggcmpy.tracing import emfields, integrator


class BorisIntegratorBase:
    """
    Base class for Boris integrators.
    """

    def __init__(
        self,
        fields: emfields.emfields,
        q=constants.e,
        m=constants.m_e,
        integrator_boris_cls=None,
    ):
        self._fields = fields
        self._q = q
        self._m = m
        self._integrator_boris_cls = integrator_boris_cls

    def integrate(self, x0, u0, t_final, dt_max=1.0, dt_max_gyro=0.1) -> pd.DataFrame:
        integrator_boris: integrator.boris_base = self._integrator_boris_cls(
            self._fields, self._q, self._m
        )

        prts_df = pd.DataFrame(
            np.array([[0.0, *x0, *u0]]),
            columns=["time", "x", "y", "z", "ux", "uy", "uz"],
        )

        return integrator_boris.integrate(
            prts_df,
            t_final=t_final,
            snapshot_interval_steps=1,
            dt_max=dt_max,
            dt_max_gyro=dt_max_gyro,
        )


class BorisIntegrator_python(BorisIntegratorBase):
    """
    BorisIntegrator_python implements the Boris algorithm for integrating the motion of charged particles in electromagnetic fields.
    This class supports both Yee and non-Yee field interpolators, automatically selecting the appropriate interpolator based on the input dataset.

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

    def __init__(self, ds, q=constants.e, m=constants.m_e) -> None:
        if isinstance(ds, xr.Dataset):
            if {"bx1", "by1", "bz1", "eflx", "efly", "eflz"} <= ds.data_vars.keys():
                fields: emfields.emfields = emfields.yee_cic_python(ds)
            else:
                fields = emfields.interpolator_python(ds)
        else:
            fields = ds  # assume it's already an emfields.emfields

        super().__init__(fields, q, m, integrator_boris_cls=integrator.boris_python)


class BorisIntegrator_cxx(BorisIntegratorBase):
    """
    BorisIntegrator_cxx provides an interface for integrating charged particle trajectories
    using the Boris algorithm, with field interpolation via C++ routines.

    Args:
        df (xr.Dataset or emfields.interpolator_cxx or emfields.interpolator_yee_cxx):
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

    def __init__(self, df, q=constants.e, m=constants.m_e):
        if isinstance(df, xr.Dataset):
            fields = emfields.yee_cic_cxx(df)
        else:
            assert isinstance(
                df, (emfields.uniform_cxx, emfields.dipole_cxx, emfields.yee_cic_cxx)
            )
            fields = df

        super().__init__(fields, q, m, integrator_boris_cls=integrator.boris_cxx)
