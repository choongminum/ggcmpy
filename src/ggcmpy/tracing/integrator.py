"""
integrator.py

Test Particle Integrators
"""

from __future__ import annotations

import pandas as pd
import xarray as xr

from ggcmpy import (
    constants,
)
from ggcmpy.tracing import emfields
from ggcmpy.tracing.boris_push import boris_push_cxx, boris_push_python

# pylint: disable=C0103


class boris_base:
    """
    Base class for Boris particle pusher.

    Methods:
        integrate(prts, ...):
            Pushes the particles in `prts`.
    """

    def __init__(
        self,
        fields: emfields.emfields,
        q=constants.e,
        m=constants.m_e,
        boris_push_cls=None,
    ):
        self._fields = fields
        self._q = q
        self._m = m
        self._boris_push_cls = boris_push_cls

    def integrate(
        self,
        prts_df: pd.DataFrame,
        t_final: float | None = None,
        dt_max: float | None = None,
        dt_max_gyro: float = 0.1,
        snapshot_interval_steps: int | None = None,
    ) -> pd.DataFrame:
        boris = self._boris_push_cls(self._fields, self._q, self._m)

        snapshots = [prts_df]

        while prts_df.iloc[0].time < t_final:
            prts_df = boris.push(
                prts_df,
                t_final=t_final,
                max_steps=snapshot_interval_steps,
                dt_max=dt_max,
                dt_max_gyro=dt_max_gyro,
            )
            snapshots.append(prts_df)

        return pd.concat(snapshots, ignore_index=True)


class boris_python(boris_base):
    """
    Boris particle pusher implemented in pure Python

    Methods:
        push(prts, t_final, dt_max, dt_max_gyro):
            Pushes the particles in `prts` from their current state to a maximum time of `t_final`,
            using a maximum time step of `dt_max` and a maximum gyro period of `dt_max_gyro`.
    """

    def __init__(
        self,
        fields: xr.Dataset | emfields.emfields,
        q=constants.e,
        m=constants.m_e,
    ):
        if isinstance(fields, xr.Dataset):
            fields = emfields.yee_cic_python(fields)

        super().__init__(fields, q, m, boris_push_cls=boris_push_python)


class boris_cxx(boris_base):
    """
    Boris particle pusher implemented in C++.

    Methods:
        push(prts, t_final, dt_max, dt_max_gyro):
            Pushes the particles in `prts` from their current state to a maximum time of `t_final`,
            using a maximum time step of `dt_max` and a maximum gyro period of `dt_max_gyro`.
    """

    def __init__(
        self,
        fields: xr.Dataset | emfields.emfields,
        q=constants.e,
        m=constants.m_e,
    ):
        if isinstance(fields, xr.Dataset):
            fields = emfields.yee_cic_cxx(fields)

        super().__init__(fields, q, m, boris_push_cls=boris_push_cxx)
