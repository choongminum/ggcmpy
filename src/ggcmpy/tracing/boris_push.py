"""
boris_push.py

Boris particle pusher implementations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ggcmpy import constants
from ggcmpy.tracing import emfields


class boris_push_python:
    """
    A class implementing the Boris particle pusher algorithm in pure Python.
    """

    def __init__(self, fields: emfields.emfields, q=constants.e, m=constants.m_e):
        self._fields = fields
        self._q = q
        self._m = m

    def push(
        self,
        prts_df: pd.DataFrame,
        t_final: float | None = None,
        max_steps: int | None = None,
        dt_max: float | None = None,
        dt_max_gyro: float = 0.1,
    ) -> pd.DataFrame:
        """
        Pushes the particles in `prts_df` from their current state.

        Args:
            prts_df (pd.DataFrame): DataFrame containing particle states with columns ['time', 'x', 'y', 'z', 'ux', 'uy', 'uz'].
            t_final (float | None): Final time to push the particles to.
            max_steps (int | None): Maximum number of steps to take.
            dt_max (float | None): Maximum time step for the integration.
            dt_max_gyro (float): Maximum time step as fraction of the gyroperiod.
        """
        qprime = 0.5 * self._q / self._m
        B = self._fields.B(prts_df.loc[0, ["x", "y", "z"]].to_numpy())
        u = prts_df.loc[0, ["ux", "uy", "uz"]].to_numpy()
        gamma = np.sqrt(1 + np.linalg.norm(u) ** 2)
        om_c = 2.0 * np.abs(qprime) * np.linalg.norm(B) / gamma
        dt = dt_max_gyro * 2.0 * np.pi / om_c
        if dt_max is not None:
            dt = min(dt_max, dt)

        assert t_final is not None or max_steps is not None

        step = 0
        while True:
            if t_final is not None and prts_df.loc[0, "time"] >= t_final:  # type: ignore[operator]
                break

            if max_steps is not None and step >= max_steps:
                break

            prts_df.loc[0, ["x", "y", "z"]] = self.push_x(
                prts_df.loc[0, ["x", "y", "z"]].to_numpy(),
                prts_df.loc[0, ["ux", "uy", "uz"]].to_numpy(),
                0.5 * dt,
            )
            B = self._fields.B(prts_df.loc[0, ["x", "y", "z"]].to_numpy())
            E = self._fields.E(prts_df.loc[0, ["x", "y", "z"]].to_numpy())
            prts_df.loc[0, ["ux", "uy", "uz"]] = self.push_u(
                prts_df.iloc[0][["ux", "uy", "uz"]].to_numpy(), E, B, qprime * dt
            )
            prts_df.loc[0, ["x", "y", "z"]] = self.push_x(
                prts_df.loc[0, ["x", "y", "z"]].to_numpy(),
                prts_df.loc[0, ["ux", "uy", "uz"]].to_numpy(),
                0.5 * dt,
            )
            prts_df.loc[0, "time"] += dt
            step += 1

        return prts_df

    @staticmethod
    def push_x(x: np.ndarray, u: np.ndarray, dt: float) -> np.ndarray:
        inv_gamma = 1.0 / np.sqrt(1 + np.linalg.norm(u) ** 2)
        return x + dt * u * inv_gamma * constants.c  # type: ignore[no-any-return]

    @staticmethod
    def push_u(u: np.ndarray, E: np.ndarray, B: np.ndarray, dq: float):
        um = u + dq * E / constants.c

        # Rotation due to magnetic field
        root = dq / np.sqrt(1.0 + np.linalg.norm(um) ** 2)
        tau = root * B
        tau_norm = 1.0 / (1.0 + np.linalg.norm(tau) ** 2)
        up = np.array(
            [
                (
                    (1.0 + (tau[0]) ** 2 - (tau[1]) ** 2 - (tau[2]) ** 2) * um[0]
                    + (2.0 * tau[0] * tau[1] + 2.0 * tau[2]) * um[1]
                    + (2.0 * tau[0] * tau[2] - 2.0 * tau[1]) * um[2]
                )
                * tau_norm,
                (
                    (2.0 * tau[0] * tau[1] - 2.0 * tau[2]) * um[0]
                    + (1.0 - (tau[0]) ** 2 + (tau[1]) ** 2 - (tau[2]) ** 2) * um[1]
                    + (2.0 * tau[1] * tau[2] + 2.0 * tau[0]) * um[2]
                )
                * tau_norm,
                (
                    (2.0 * tau[0] * tau[2] + 2.0 * tau[1]) * um[0]
                    + (2.0 * tau[1] * tau[2] - 2.0 * tau[0]) * um[1]
                    + (1.0 - (tau[0]) ** 2 - (tau[1]) ** 2 + (tau[2]) ** 2) * um[2]
                )
                * tau_norm,
            ]
        )
        return up + dq * E / constants.c
