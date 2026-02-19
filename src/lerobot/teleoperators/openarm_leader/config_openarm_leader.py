#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

from dataclasses import dataclass, field

from ..config import TeleoperatorConfig


@dataclass
class OpenArmLeaderConfigBase:
    """Base configuration for the OpenArms leader/teleoperator with Damiao motors."""

    # CAN interfaces - one per arm
    # Arm CAN interface (e.g., "can3")
    # Linux: "can0", "can1", etc.
    port: str

    # CAN interface type: "socketcan" (Linux), "slcan" (serial), or "auto" (auto-detect)
    can_interface: str = "socketcan"

    # CAN FD settings (OpenArms uses CAN FD by default)
    use_can_fd: bool = True
    can_bitrate: int = 1000000  # Nominal bitrate (1 Mbps)
    can_data_bitrate: int = 5000000  # Data bitrate for CAN FD (5 Mbps)

    # Motor configuration for OpenArms (7 DOF per arm)
    # Maps motor names to (send_can_id, recv_can_id, motor_type)
    # Based on: https://docs.openarm.dev/software/setup/configure-test
    # OpenArms uses 4 types of motors:
    # - DM8009 (DM-J8009P-2EC) for shoulders (high torque)
    # - DM4340P and DM4340 for shoulder rotation and elbow
    # - DM4310 (DM-J4310-2EC V1.1) for wrist and gripper
    motor_config: dict[str, tuple[int, int, str]] = field(
        default_factory=lambda: {
            "joint_1": (0x01, 0x11, "dm8009"),  # J1 - Shoulder pan (DM8009)
            "joint_2": (0x02, 0x12, "dm8009"),  # J2 - Shoulder lift (DM8009)
            "joint_3": (0x03, 0x13, "dm4340"),  # J3 - Shoulder rotation (DM4340)
            "joint_4": (0x04, 0x14, "dm4340"),  # J4 - Elbow flex (DM4340)
            "joint_5": (0x05, 0x15, "dm4310"),  # J5 - Wrist roll (DM4310)
            "joint_6": (0x06, 0x16, "dm4310"),  # J6 - Wrist pitch (DM4310)
            "joint_7": (0x07, 0x17, "dm4310"),  # J7 - Wrist rotation (DM4310)
            "gripper": (0x08, 0x18, "dm4310"),  # J8 - Gripper (DM4310)
        }
    )

    # Torque mode settings for manual control
    # When enabled, motors have torque disabled for manual movement
    manual_control: bool = True

    # TODO(Steven, Pepijn): Not used ... ?
    # MIT control parameters (used when manual_control=False for torque control)
    # List of 8 values: [joint_1, joint_2, joint_3, joint_4, joint_5, joint_6, joint_7, gripper]
    position_kp: list[float] = field(
        default_factory=lambda: [240.0, 240.0, 240.0, 240.0, 24.0, 31.0, 25.0, 16.0]
    )
    position_kd: list[float] = field(default_factory=lambda: [3.0, 3.0, 3.0, 3.0, 0.2, 0.2, 0.2, 0.2])

    # === Gravity Compensation Settings ===

    # Enable gravity compensation (requires manual_control=False and valid URDF)
    # When enabled, the leader arm will apply torques to counteract gravity,
    # reducing the perceived weight during teleoperation
    gravity_compensation: bool = False

    # Path to OpenArm URDF file (supports environment variable expansion)
    # The URDF must be obtained from: https://github.com/enactic/openarm
    # See URDF_README.md for setup instructions
    urdf_path: str = "${LEROBOT_ROOT}/src/lerobot/teleoperators/openarm_leader/openarm.urdf"

    # Gravity compensation gain (scaling factor for computed torques)
    # Adjust this to compensate for URDF parameter inaccuracies
    # - If arm drifts down: increase gain (e.g., 1.05, 1.10)
    # - If arm floats up: decrease gain (e.g., 0.95, 0.90)
    # Typical range: 0.7 to 1.3
    gravity_compensation_gain: float = 0.7

    # Gravity vector in base frame [x, y, z] in m/s²
    # Default: [0, 0, -9.81] for standard mounting (gravity pointing down)
    # Modify for wall-mounted or tilted installations
    gravity_vector: list[float] = field(default_factory=lambda: [0.0, 0.0, -9.81])

    # Software torque limits for gravity compensation [Nm]
    # Applied per joint to prevent excessive torques from URDF errors or sensor noise
    # Values are set below hardware limits for safety margin (~80% of motor rating)
    # [joint_1, joint_2, joint_3, joint_4, joint_5, joint_6, joint_7]
    software_torque_limits: list[float] = field(
        default_factory=lambda: [
            8.0,  # joint_1 (DM8009, rated 10Nm)
            8.0,  # joint_2 (DM8009, rated 10Nm)
            5.0,  # joint_3 (DM4340, rated 6Nm)
            5.0,  # joint_4 (DM4340, rated 6Nm)
            2.0,  # joint_5 (DM4310, rated 2.5Nm)
            2.0,  # joint_6 (DM4310, rated 2.5Nm)
            2.0,  # joint_7 (DM4310, rated 2.5Nm)
        ]
    )

    # MIT control gains for gravity compensation mode (softer than normal control)
    # Lower Kp/Kd values reduce oscillations and improve compliance during teleoperation
    # List of 8 values: [joint_1, joint_2, joint_3, joint_4, joint_5, joint_6, joint_7, gripper]
    gravity_comp_position_kp: list[float] = field(
        default_factory=lambda: [50.0, 50.0, 50.0, 50.0, 10.0, 10.0, 10.0, 10.0]
    )
    gravity_comp_position_kd: list[float] = field(
        default_factory=lambda: [2.0, 2.0, 2.0, 2.0, 0.2, 0.2, 0.2, 0.2]
    )


@TeleoperatorConfig.register_subclass("openarm_leader")
@dataclass
class OpenArmLeaderConfig(TeleoperatorConfig, OpenArmLeaderConfigBase):
    pass
