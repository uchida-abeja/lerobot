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

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .openarm_kinematic_processor import OpenArmIK


ARM_JOINT_NAMES = tuple(f"joint_{i}" for i in range(1, 8))
ALL_JOINT_NAMES = ARM_JOINT_NAMES + ("gripper",)


def compute_friction(
    velocity_rad_s: np.ndarray,
    fc: np.ndarray,
    fv: np.ndarray,
    fo: np.ndarray,
    k: np.ndarray,
) -> np.ndarray:
    """Joint-wise tanh friction model used by the position synchronization controller."""
    return fc * np.tanh(k * velocity_rad_s) + fv * velocity_rad_s + fo


def is_position_sync_controller(config: object) -> bool:
    controller_type = str(getattr(config, "force_feedback_controller_type", "observer")).strip().lower()
    return bool(getattr(config, "force_feedback_enabled", False)) and controller_type in {
        "position_sync",
        "position-sync",
        "pos_sync",
    }


def _state_vector(data: dict[str, float], suffix: str) -> np.ndarray:
    return np.array([float(data.get(f"{joint}.{suffix}", 0.0)) for joint in ARM_JOINT_NAMES], dtype=float)


def _scalar_state(data: dict[str, float], name: str, suffix: str) -> float:
    return float(data.get(f"{name}.{suffix}", 0.0))


@dataclass
class PositionSyncDiagnostics:
    leader_gravity_torque_nm: np.ndarray
    leader_friction_torque_nm: np.ndarray
    follower_gravity_torque_nm: np.ndarray
    follower_friction_torque_nm: np.ndarray


class OpenArmPositionSyncController:
    """
    ROS2-style bilateral position synchronization controller for OpenArm.

    The controller uses the motor-side MIT PD loop and supplies only the
    feedforward term (gravity + friction) from Python.
    """

    def __init__(
        self,
        urdf_path: str,
        gravity_vector: Iterable[float],
        gravity_gain: float,
        software_torque_limits: Iterable[float],
        leader_position_kp: Iterable[float],
        leader_position_kd: Iterable[float],
        follower_position_kp: Iterable[float],
        follower_position_kd: Iterable[float],
        friction_fc: Iterable[float],
        friction_fv: Iterable[float],
        friction_fo: Iterable[float],
        friction_k: Iterable[float],
    ) -> None:
        self.arm_ik = OpenArmIK(urdf_path=urdf_path, gravity_vector=np.array(list(gravity_vector), dtype=float))
        self.gravity_gain = float(gravity_gain)
        self.software_torque_limits = np.array(list(software_torque_limits), dtype=float)
        self.leader_position_kp = np.array(list(leader_position_kp), dtype=float)
        self.leader_position_kd = np.array(list(leader_position_kd), dtype=float)
        self.follower_position_kp = np.array(list(follower_position_kp), dtype=float)
        self.follower_position_kd = np.array(list(follower_position_kd), dtype=float)
        self.friction_fc = np.array(list(friction_fc), dtype=float)
        self.friction_fv = np.array(list(friction_fv), dtype=float)
        self.friction_fo = np.array(list(friction_fo), dtype=float)
        self.friction_k = np.array(list(friction_k), dtype=float)

        self._ensure_length(self.software_torque_limits, 7, "software_torque_limits")
        self._ensure_length(self.leader_position_kp, 8, "leader_position_kp")
        self._ensure_length(self.leader_position_kd, 8, "leader_position_kd")
        self._ensure_length(self.follower_position_kp, 8, "follower_position_kp")
        self._ensure_length(self.follower_position_kd, 8, "follower_position_kd")
        self._ensure_length(self.friction_fc, 7, "friction_fc")
        self._ensure_length(self.friction_fv, 7, "friction_fv")
        self._ensure_length(self.friction_fo, 7, "friction_fo")
        self._ensure_length(self.friction_k, 7, "friction_k")

    @staticmethod
    def _ensure_length(values: np.ndarray, expected: int, name: str) -> None:
        if values.shape[0] != expected:
            raise ValueError(f"{name} must have length {expected}, got {values.shape[0]}")

    def _gravity_and_friction(self, state: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
        q_deg = _state_vector(state, "pos")
        dq_deg_s = _state_vector(state, "vel")
        q_rad = np.deg2rad(q_deg)
        dq_rad_s = np.deg2rad(dq_deg_s)

        tau_gravity = self.arm_ik.solve_tau(q_rad) * self.gravity_gain
        tau_friction = compute_friction(
            velocity_rad_s=dq_rad_s,
            fc=self.friction_fc,
            fv=self.friction_fv,
            fo=self.friction_fo,
            k=self.friction_k,
        )
        tau_ff = np.clip(
            tau_gravity + tau_friction,
            -self.software_torque_limits,
            self.software_torque_limits,
        )
        return tau_ff, tau_gravity

    def compute_commands(
        self,
        leader_state: dict[str, float],
        follower_state: dict[str, float],
        follower_goal_action: dict[str, float] | None = None,
    ) -> tuple[dict[str, tuple[float, float, float, float, float]], dict[str, tuple[float, float, float, float, float]], PositionSyncDiagnostics]:
        follower_goal_action = follower_goal_action or {}

        leader_tau_ff, leader_tau_gravity = self._gravity_and_friction(leader_state)
        follower_tau_ff, follower_tau_gravity = self._gravity_and_friction(follower_state)

        leader_friction = leader_tau_ff - leader_tau_gravity
        follower_friction = follower_tau_ff - follower_tau_gravity

        leader_commands: dict[str, tuple[float, float, float, float, float]] = {}
        follower_commands: dict[str, tuple[float, float, float, float, float]] = {}

        for index, joint_name in enumerate(ARM_JOINT_NAMES):
            follower_target_pos = float(follower_goal_action.get(f"{joint_name}.pos", leader_state.get(f"{joint_name}.pos", 0.0)))
            follower_target_vel = float(leader_state.get(f"{joint_name}.vel", 0.0))
            leader_target_pos = float(follower_state.get(f"{joint_name}.pos", 0.0))
            leader_target_vel = float(follower_state.get(f"{joint_name}.vel", 0.0))

            follower_commands[joint_name] = (
                float(self.follower_position_kp[index]),
                float(self.follower_position_kd[index]),
                follower_target_pos,
                follower_target_vel,
                float(follower_tau_ff[index]),
            )
            leader_commands[joint_name] = (
                float(self.leader_position_kp[index]),
                float(self.leader_position_kd[index]),
                leader_target_pos,
                leader_target_vel,
                float(leader_tau_ff[index]),
            )

        gripper_index = 7
        follower_commands["gripper"] = (
            float(self.follower_position_kp[gripper_index]),
            float(self.follower_position_kd[gripper_index]),
            float(follower_goal_action.get("gripper.pos", _scalar_state(leader_state, "gripper", "pos"))),
            float(leader_state.get("gripper.vel", 0.0)),
            0.0,
        )
        leader_commands["gripper"] = (
            float(self.leader_position_kp[gripper_index]),
            float(self.leader_position_kd[gripper_index]),
            _scalar_state(follower_state, "gripper", "pos"),
            _scalar_state(follower_state, "gripper", "vel"),
            0.0,
        )

        diagnostics = PositionSyncDiagnostics(
            leader_gravity_torque_nm=leader_tau_gravity,
            leader_friction_torque_nm=leader_friction,
            follower_gravity_torque_nm=follower_tau_gravity,
            follower_friction_torque_nm=follower_friction,
        )
        return leader_commands, follower_commands, diagnostics

