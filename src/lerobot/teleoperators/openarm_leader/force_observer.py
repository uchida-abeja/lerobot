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
from typing import Iterable

import numpy as np

from .openarm_kinematic_processor import OpenArmIK

logger = logging.getLogger(__name__)


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
        self.arm_ik = OpenArmIK(
            urdf_path=urdf_path,
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

        if dt_s is None or dt_s <= 0.0 or self.lpf_cutoff_hz <= 0.0:
            self._lpf_state = tau_ext
            self._initialized = True
            return tau_ext

        alpha = 1.0 - math.exp(-2.0 * math.pi * self.lpf_cutoff_hz * dt_s)
        if not self._initialized:
            self._lpf_state = tau_ext
            self._initialized = True
        else:
            self._lpf_state = self._lpf_state + alpha * (tau_ext - self._lpf_state)

        return self._lpf_state
