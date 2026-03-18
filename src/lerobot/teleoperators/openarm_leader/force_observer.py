#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import math
import os
from typing import Iterable

import numpy as np

from .openarm_kinematic_processor import OpenArmIK

logger = logging.getLogger(__name__)


def _first_order_lpf_alpha(cutoff_hz: float, dt_s: float | None) -> float:
    if dt_s is None or dt_s <= 0.0 or cutoff_hz <= 0.0:
        return 1.0
    return 1.0 - math.exp(-2.0 * math.pi * cutoff_hz * dt_s)


class ForceObserver:
    """
    Simple external torque estimator for OpenArm.

    Estimates external torque by subtracting model gravity torque from
    measured motor torque. A first-order low-pass filter is applied.
    """

    def __init__(
        self,
        urdf_path: str,
        gravity_vector: Iterable[float],
        gravity_gain: float,
        lpf_cutoff_hz: float,
        torque_limits: Iterable[float],
        num_arm_joints: int = 7,
    ) -> None:
        expanded_urdf_path = os.path.expandvars(urdf_path)
        self.arm_ik = OpenArmIK(
            urdf_path=expanded_urdf_path,
            gravity_vector=np.array(list(gravity_vector), dtype=float),
            num_arm_joints=num_arm_joints,
        )
        self.gravity_gain = gravity_gain
        self.lpf_cutoff_hz = lpf_cutoff_hz
        self.torque_limits = np.array(list(torque_limits), dtype=float)
        self.num_arm_joints = num_arm_joints
        self._lpf_state = np.zeros(self.num_arm_joints, dtype=float)
        self._initialized = False

    def estimate(
        self, observation: dict[str, float], dt_s: float | None = None
    ) -> np.ndarray:
        joint_positions_deg = []
        tau_meas = []
        for i in range(1, self.num_arm_joints + 1):
            joint_positions_deg.append(observation.get(f"joint_{i}.pos", 0.0))
            tau_meas.append(observation.get(f"joint_{i}.torque", 0.0))

        q = np.deg2rad(np.array(joint_positions_deg, dtype=float))
        tau_meas = np.array(tau_meas, dtype=float)

        tau_grav = self.arm_ik.solve_tau(q) * self.gravity_gain
        tau_ext = tau_meas - tau_grav

        if self.torque_limits.shape[0] == self.num_arm_joints:
            tau_ext = np.clip(tau_ext, -self.torque_limits, self.torque_limits)

        alpha = _first_order_lpf_alpha(self.lpf_cutoff_hz, dt_s)
        if alpha >= 1.0:
            self._lpf_state = tau_ext
            self._initialized = True
            return tau_ext

        if not self._initialized:
            self._lpf_state = tau_ext
            self._initialized = True
        else:
            self._lpf_state = self._lpf_state + alpha * (tau_ext - self._lpf_state)

        return self._lpf_state

    def estimate_with_diagnostics(
        self, observation: dict[str, float], dt_s: float | None = None
    ) -> tuple[np.ndarray, dict[str, float | bool | int | str]]:
        tau_ext = self.estimate(observation, dt_s=dt_s)
        saturated_count = int(np.sum(np.abs(tau_ext) >= self.torque_limits - 1e-9))
        diagnostics: dict[str, float | bool | int | str] = {
            "observer_type": "simple",
            "diverged": False,
            "confidence": 1.0,
            "saturated_joint_count": saturated_count,
        }
        return tau_ext, diagnostics


class DobRfobForceObserver:
    """
    DOB+RFOB style external torque estimator for OpenArm.

    This observer estimates lumped disturbance using a nominal internal model
    (gravity + simple friction), then exposes the filtered disturbance as
    estimated external torque.
    """

    def __init__(
        self,
        urdf_path: str,
        gravity_vector: Iterable[float],
        gravity_gain: float,
        dob_lpf_cutoff_hz: float,
        rfob_lpf_cutoff_hz: float,
        torque_limits: Iterable[float],
        friction_viscous: Iterable[float],
        friction_coulomb: Iterable[float],
        velocity_lpf_cutoff_hz: float,
        divergence_threshold_nm: float,
        num_arm_joints: int = 7,
    ) -> None:
        expanded_urdf_path = os.path.expandvars(urdf_path)
        self.arm_ik = OpenArmIK(
            urdf_path=expanded_urdf_path,
            gravity_vector=np.array(list(gravity_vector), dtype=float),
            num_arm_joints=num_arm_joints,
        )
        self.gravity_gain = gravity_gain
        self.dob_lpf_cutoff_hz = dob_lpf_cutoff_hz
        self.rfob_lpf_cutoff_hz = rfob_lpf_cutoff_hz
        self.velocity_lpf_cutoff_hz = velocity_lpf_cutoff_hz
        self.divergence_threshold_nm = divergence_threshold_nm
        self.num_arm_joints = num_arm_joints

        self.torque_limits = np.array(list(torque_limits), dtype=float)
        self.friction_viscous = np.array(list(friction_viscous), dtype=float)
        self.friction_coulomb = np.array(list(friction_coulomb), dtype=float)

        if self.friction_viscous.shape[0] != self.num_arm_joints:
            self.friction_viscous = np.zeros(self.num_arm_joints, dtype=float)
        if self.friction_coulomb.shape[0] != self.num_arm_joints:
            self.friction_coulomb = np.zeros(self.num_arm_joints, dtype=float)

        self._prev_q_rad: np.ndarray | None = None
        self._vel_lpf_state = np.zeros(self.num_arm_joints, dtype=float)
        self._dob_state = np.zeros(self.num_arm_joints, dtype=float)
        self._rfob_state = np.zeros(self.num_arm_joints, dtype=float)
        self._last_tau_dis_raw = np.zeros(self.num_arm_joints, dtype=float)
        self._initialized = False

    def estimate(
        self, observation: dict[str, float], dt_s: float | None = None
    ) -> np.ndarray:
        joint_positions_deg = []
        tau_meas = []
        for i in range(1, self.num_arm_joints + 1):
            joint_positions_deg.append(observation.get(f"joint_{i}.pos", 0.0))
            tau_meas.append(observation.get(f"joint_{i}.torque", 0.0))

        q_rad = np.deg2rad(np.array(joint_positions_deg, dtype=float))
        tau_meas = np.array(tau_meas, dtype=float)

        if dt_s is None or dt_s <= 0.0 or self._prev_q_rad is None:
            raw_vel = np.zeros(self.num_arm_joints, dtype=float)
        else:
            raw_vel = (q_rad - self._prev_q_rad) / dt_s
        self._prev_q_rad = q_rad

        vel_alpha = _first_order_lpf_alpha(self.velocity_lpf_cutoff_hz, dt_s)
        self._vel_lpf_state = self._vel_lpf_state + vel_alpha * (raw_vel - self._vel_lpf_state)
        vel_est = self._vel_lpf_state

        tau_grav = self.arm_ik.solve_tau(q_rad) * self.gravity_gain
        tau_fric = self.friction_viscous * vel_est + self.friction_coulomb * np.tanh(20.0 * vel_est)
        tau_internal = tau_grav + tau_fric

        # DOB: estimate lumped disturbance around nominal internal torque model.
        tau_dis_raw = tau_meas - tau_internal
        self._last_tau_dis_raw = tau_dis_raw
        dob_alpha = _first_order_lpf_alpha(self.dob_lpf_cutoff_hz, dt_s)
        if not self._initialized or dob_alpha >= 1.0:
            self._dob_state = tau_dis_raw
        else:
            self._dob_state = self._dob_state + dob_alpha * (tau_dis_raw - self._dob_state)

        # RFOB: filtered external torque estimate from disturbance channel.
        rfob_alpha = _first_order_lpf_alpha(self.rfob_lpf_cutoff_hz, dt_s)
        if not self._initialized or rfob_alpha >= 1.0:
            self._rfob_state = self._dob_state
        else:
            self._rfob_state = self._rfob_state + rfob_alpha * (self._dob_state - self._rfob_state)

        self._initialized = True
        tau_ext = self._rfob_state

        if self.torque_limits.shape[0] == self.num_arm_joints:
            tau_ext = np.clip(tau_ext, -self.torque_limits, self.torque_limits)

        return tau_ext

    def estimate_with_diagnostics(
        self, observation: dict[str, float], dt_s: float | None = None
    ) -> tuple[np.ndarray, dict[str, float | bool | int | str]]:
        tau_ext = self.estimate(observation, dt_s=dt_s)

        residual = self._last_tau_dis_raw - self._dob_state
        residual_rms = float(np.sqrt(np.mean(np.square(residual))))
        confidence = float(1.0 / (1.0 + residual_rms))

        diverged = bool(
            np.any(np.abs(self._last_tau_dis_raw) > self.divergence_threshold_nm)
            or np.any(~np.isfinite(tau_ext))
        )
        saturated_count = int(np.sum(np.abs(tau_ext) >= self.torque_limits - 1e-9))

        diagnostics: dict[str, float | bool | int | str] = {
            "observer_type": "dob_rfob",
            "diverged": diverged,
            "confidence": confidence,
            "residual_rms": residual_rms,
            "saturated_joint_count": saturated_count,
        }
        return tau_ext, diagnostics


def create_force_observer(
    observer_type: str,
    urdf_path: str,
    gravity_vector: Iterable[float],
    gravity_gain: float,
    lpf_cutoff_hz: float,
    torque_limits: Iterable[float],
    dob_lpf_cutoff_hz: float,
    friction_viscous: Iterable[float],
    friction_coulomb: Iterable[float],
    velocity_lpf_cutoff_hz: float,
    divergence_threshold_nm: float,
    num_arm_joints: int = 7,
) -> ForceObserver | DobRfobForceObserver:
    observer_type_normalized = observer_type.strip().lower()
    if observer_type_normalized in {"dob_rfob", "dob-rfob", "dob"}:
        logger.info("Using DOB+RFOB force observer")
        return DobRfobForceObserver(
            urdf_path=urdf_path,
            gravity_vector=gravity_vector,
            gravity_gain=gravity_gain,
            dob_lpf_cutoff_hz=dob_lpf_cutoff_hz,
            rfob_lpf_cutoff_hz=lpf_cutoff_hz,
            torque_limits=torque_limits,
            friction_viscous=friction_viscous,
            friction_coulomb=friction_coulomb,
            velocity_lpf_cutoff_hz=velocity_lpf_cutoff_hz,
            divergence_threshold_nm=divergence_threshold_nm,
            num_arm_joints=num_arm_joints,
        )

    logger.info("Using legacy force observer")
    return ForceObserver(
        urdf_path=urdf_path,
        gravity_vector=gravity_vector,
        gravity_gain=gravity_gain,
        lpf_cutoff_hz=lpf_cutoff_hz,
        torque_limits=torque_limits,
        num_arm_joints=num_arm_joints,
    )
