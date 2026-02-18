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

from dataclasses import dataclass, field

from lerobot.teleoperators.openarm_leader import OpenArmLeaderConfigBase

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("bi_openarm_leader")
@dataclass
class BiOpenArmLeaderConfig(TeleoperatorConfig):
    """Configuration class for Bi OpenArm Leader robots."""

    left_arm_config: OpenArmLeaderConfigBase
    right_arm_config: OpenArmLeaderConfigBase

    # Bimanual gravity compensation settings
    # When enabled, both arms will apply gravity compensation independently
    gravity_compensation: bool = False

    # Gravity compensation gain (applied to both arms unless overridden in arm configs)
    # Adjust to compensate for URDF parameter inaccuracies
    gravity_compensation_gain: float = 0.7

    # URDF path and gravity vector for dynamics computations
    urdf_path: str = "${LEROBOT_ROOT}/src/lerobot/teleoperators/openarm_leader/openarm.urdf"
    gravity_vector: list[float] = field(default_factory=lambda: [0.0, 0.0, -9.81])

    # Software torque limits for gravity compensation [Nm]
    software_torque_limits: list[float] = field(
        default_factory=lambda: [8.0, 8.0, 5.0, 5.0, 2.0, 2.0, 2.0]
    )

    # Force feedback settings (applied to both arms)
    force_feedback_enabled: bool = False
    force_feedback_gain: float = 0.1
    force_feedback_lpf_cutoff_hz: float = 10.0
    force_feedback_torque_limits: list[float] = field(
        default_factory=lambda: [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5]
    )
    force_feedback_position_kp: list[float] = field(
        default_factory=lambda: [50.0, 50.0, 50.0, 50.0, 10.0, 10.0, 10.0, 10.0]
    )
    force_feedback_position_kd: list[float] = field(
        default_factory=lambda: [2.0, 2.0, 2.0, 2.0, 0.2, 0.2, 0.2, 0.2]
    )
