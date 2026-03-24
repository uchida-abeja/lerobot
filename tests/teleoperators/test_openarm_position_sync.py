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

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

LEROBOT_ROOT = Path(__file__).parent.parent.parent
if str(LEROBOT_ROOT / "src") not in sys.path:
    sys.path.append(str(LEROBOT_ROOT / "src"))


def _make_state(prefix: str = "") -> dict[str, float]:
    state: dict[str, float] = {}
    for i in range(1, 8):
        state[f"{prefix}joint_{i}.pos"] = float(i * 10)
        state[f"{prefix}joint_{i}.vel"] = float(i)
        state[f"{prefix}joint_{i}.torque"] = 0.0
    state[f"{prefix}gripper.pos"] = -5.0
    state[f"{prefix}gripper.vel"] = -1.0
    state[f"{prefix}gripper.torque"] = 0.0
    return state


def test_compute_friction_matches_tanh_model():
    from lerobot.teleoperators.openarm_leader.position_sync import compute_friction

    velocity = np.array([0.1, -0.2])
    fc = np.array([0.5, 0.3])
    fv = np.array([0.2, 0.1])
    fo = np.array([0.05, -0.02])
    k = np.array([10.0, 8.0])

    friction = compute_friction(velocity, fc, fv, fo, k)
    expected = fc * np.tanh(k * velocity) + fv * velocity + fo

    assert np.allclose(friction, expected)


def test_position_sync_controller_uses_partner_positions_and_feedforward():
    from lerobot.teleoperators.openarm_leader.position_sync import OpenArmPositionSyncController

    leader_state = _make_state()
    follower_state = _make_state()
    follower_goal_action = {f"joint_{i}.pos": float(100 + i) for i in range(1, 8)}
    follower_goal_action["gripper.pos"] = -12.0

    with patch("lerobot.teleoperators.openarm_leader.position_sync.OpenArmIK") as mock_ik:
        ik_instance = MagicMock()
        ik_instance.solve_tau.side_effect = [
            np.array([0.4, 0.4, 0.3, 0.3, 0.2, 0.2, 0.1]),
            np.array([0.5, 0.5, 0.4, 0.4, 0.3, 0.3, 0.2]),
        ]
        mock_ik.return_value = ik_instance

        controller = OpenArmPositionSyncController(
            urdf_path="/fake/openarm.urdf",
            gravity_vector=[0.0, 0.0, -9.81],
            gravity_gain=1.0,
            software_torque_limits=[10.0] * 7,
            leader_position_kp=[1.0] * 8,
            leader_position_kd=[0.1] * 8,
            follower_position_kp=[2.0] * 8,
            follower_position_kd=[0.2] * 8,
            friction_fc=[0.1] * 7,
            friction_fv=[0.0] * 7,
            friction_fo=[0.0] * 7,
            friction_k=[20.0] * 7,
        )

        leader_commands, follower_commands, diagnostics = controller.compute_commands(
            leader_state=leader_state,
            follower_state=follower_state,
            follower_goal_action=follower_goal_action,
        )

    assert follower_commands["joint_1"][2] == follower_goal_action["joint_1.pos"]
    assert follower_commands["joint_1"][3] == leader_state["joint_1.vel"]
    assert leader_commands["joint_1"][2] == follower_state["joint_1.pos"]
    assert leader_commands["joint_1"][3] == follower_state["joint_1.vel"]
    assert follower_commands["gripper"][2] == follower_goal_action["gripper.pos"]
    assert leader_commands["gripper"][2] == follower_state["gripper.pos"]
    assert diagnostics.leader_gravity_torque_nm.shape == (7,)
    assert diagnostics.follower_gravity_torque_nm.shape == (7,)


def test_position_sync_controller_clips_feedforward_torque():
    from lerobot.teleoperators.openarm_leader.position_sync import OpenArmPositionSyncController

    leader_state = _make_state()
    follower_state = _make_state()

    with patch("lerobot.teleoperators.openarm_leader.position_sync.OpenArmIK") as mock_ik:
        ik_instance = MagicMock()
        ik_instance.solve_tau.return_value = np.array([5.0] * 7)
        mock_ik.return_value = ik_instance

        controller = OpenArmPositionSyncController(
            urdf_path="/fake/openarm.urdf",
            gravity_vector=[0.0, 0.0, -9.81],
            gravity_gain=1.0,
            software_torque_limits=[1.0] * 7,
            leader_position_kp=[1.0] * 8,
            leader_position_kd=[0.1] * 8,
            follower_position_kp=[2.0] * 8,
            follower_position_kd=[0.2] * 8,
            friction_fc=[0.0] * 7,
            friction_fv=[0.0] * 7,
            friction_fo=[0.0] * 7,
            friction_k=[20.0] * 7,
        )

        leader_commands, follower_commands, _ = controller.compute_commands(
            leader_state=leader_state,
            follower_state=follower_state,
        )

    assert all(command[4] == 1.0 for name, command in leader_commands.items() if name != "gripper")
    assert all(command[4] == 1.0 for name, command in follower_commands.items() if name != "gripper")
    assert leader_commands["gripper"][4] == 0.0
    assert follower_commands["gripper"][4] == 0.0
