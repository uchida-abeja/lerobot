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

from dataclasses import dataclass

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
