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

import logging
import os
from pathlib import Path
import time
from typing import Any

import numpy as np

from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.damiao import DamiaoMotorsBus
from lerobot.processor import RobotAction
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..teleoperator import Teleoperator
from .config_openarm_leader import OpenArmLeaderConfig

logger = logging.getLogger(__name__)


class OpenArmLeader(Teleoperator):
    """
    OpenArm Leader/Teleoperator Arm with Damiao motors.

    This teleoperator uses CAN bus communication to read positions from
    Damiao motors that are manually moved (torque disabled).
    """

    config_class = OpenArmLeaderConfig
    name = "openarm_leader"

    def __init__(self, config: OpenArmLeaderConfig):
        super().__init__(config)
        self.config = config

        # Arm motors
        motors: dict[str, Motor] = {}
        for motor_name, (send_id, recv_id, motor_type_str) in config.motor_config.items():
            motor = Motor(
                send_id, motor_type_str, MotorNormMode.DEGREES
            )  # Always use degrees for Damiao motors
            motor.recv_id = recv_id
            motor.motor_type_str = motor_type_str
            motors[motor_name] = motor

        self.bus = DamiaoMotorsBus(
            port=self.config.port,
            motors=motors,
            calibration=self.calibration,
            can_interface=self.config.can_interface,
            use_can_fd=self.config.use_can_fd,
            bitrate=self.config.can_bitrate,
            data_bitrate=self.config.can_data_bitrate if self.config.use_can_fd else None,
        )

        # Initialize gravity compensation
        self.arm_ik = None
        if self.config.gravity_compensation:
            from .openarm_kinematic_processor import OpenArmIK

            # Expand environment variables in URDF path and convert to Path
            urdf_path = Path(os.path.expandvars(self.config.urdf_path))

            try:
                self.arm_ik = OpenArmIK(
                    urdf_path=urdf_path,
                    gravity_vector=np.array(self.config.gravity_vector),
                    num_arm_joints=7,
                )
                logger.info(
                    f"Gravity compensation enabled with gain={self.config.gravity_compensation_gain}"
                )
                if not self.config.manual_control:
                    logger.warning(
                        "Gravity compensation requires manual_control=False. "
                        "Please set manual_control=False in config."
                    )
            except Exception as e:
                logger.error(f"Failed to initialize gravity compensation: {e}")
                logger.error("Continuing without gravity compensation.")
                self.arm_ik = None

    @property
    def action_features(self) -> dict[str, type]:
        """Features produced by this teleoperator."""
        features: dict[str, type] = {}
        for motor in self.bus.motors:
            features[f"{motor}.pos"] = float
            features[f"{motor}.vel"] = float
            features[f"{motor}.torque"] = float
        return features

    @property
    def feedback_features(self) -> dict[str, type]:
        """Feedback features (not implemented for OpenArms)."""
        return {}

    @property
    def is_connected(self) -> bool:
        """Check if teleoperator is connected."""
        return self.bus.is_connected

    def connect(self, calibrate: bool = True) -> None:
        """
        Connect to the teleoperator.

        For manual control, we disable torque after connecting so the
        arm can be moved by hand.
        """
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        # Connect to CAN bus
        logger.info(f"Connecting arm on {self.config.port}...")
        self.bus.connect()

        # Run calibration if needed
        if not self.is_calibrated and calibrate:
            logger.info(
                "Mismatch between calibration values in the motor and the calibration file or no calibration file found"
            )
            self.calibrate()

        self.configure()

        if self.is_calibrated:
            self.bus.set_zero_position()

        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        """Check if teleoperator is calibrated."""
        return self.bus.is_calibrated

    def calibrate(self) -> None:
        """
        Run calibration procedure for OpenArms leader.

        The calibration procedure:
        1. Disable torque (if not already disabled)
        2. Ask user to position arm in zero position (hanging with gripper closed)
        3. Set this as zero position
        4. Record range of motion for each joint
        5. Save calibration
        """
        if self.calibration:
            # Calibration file exists, ask user whether to use it or run new calibration
            user_input = input(
                f"Press ENTER to use provided calibration file associated with the id {self.id}, or type 'c' and press ENTER to run calibration: "
            )
            if user_input.strip().lower() != "c":
                logger.info(f"Writing calibration file associated with the id {self.id} to the motors")
                self.bus.write_calibration(self.calibration)
                return

        logger.info(f"\nRunning calibration for {self}")
        self.bus.disable_torque()

        # Step 1: Set zero position
        input(
            "\nCalibration: Set Zero Position)\n"
            "Position the arm in the following configuration:\n"
            "  - Arm hanging straight down\n"
            "  - Gripper closed\n"
            "Press ENTER when ready..."
        )

        # Set current position as zero for all motors
        self.bus.set_zero_position()
        logger.info("Arm zero position set.")

        logger.info("Setting range: -90° to +90° by default for all joints")
        # TODO(Steven, Pepijn): Check if MotorCalibration is actually needed here given that we only use Degrees
        for motor_name, motor in self.bus.motors.items():
            self.calibration[motor_name] = MotorCalibration(
                id=motor.id,
                drive_mode=0,
                homing_offset=0,
                range_min=-90,
                range_max=90,
            )

        self.bus.write_calibration(self.calibration)
        self._save_calibration()
        print(f"Calibration saved to {self.calibration_fpath}")

    def configure(self) -> None:
        """
        Configure motors for teleoperation.

        Behavior:
        - gravity_compensation=True: Enable MIT control with soft gains for gravity compensation
        - manual_control=True: Disable torque for manual movement
        - Otherwise: Enable MIT control with configured gains
        """
        if self.config.gravity_compensation and self.arm_ik is not None:
            logger.info("Configuring motors for gravity compensation mode (soft control)")
            # When gravity compensation is enabled, we'll use soft MIT control
            # The actual gains will be applied in _apply_gravity_compensation()
            return self.bus.configure_motors()
        elif self.config.manual_control:
            logger.info("Configuring motors for manual control (torque disabled)")
            return self.bus.disable_torque()
        else:
            logger.info("Configuring motors for MIT control")
            return self.bus.configure_motors()

    def setup_motors(self) -> None:
        raise NotImplementedError(
            "Motor ID configuration is typically done via manufacturer tools for CAN motors."
        )

    def get_action(self) -> RobotAction:
        """
        Get current action from the leader arm.

        This is the main method for teleoperators - it reads the current state
        of the leader arm and returns it as an action that can be sent to a follower.

        When gravity compensation is enabled, this method also computes and applies
        gravity compensation torques to reduce the perceived weight of the arm.

        Reads all motor states (pos/vel/torque) in one CAN refresh cycle.
        """
        start = time.perf_counter()
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        action_dict: dict[str, Any] = {}

        # Use sync_read_all_states to get pos/vel/torque in one go
        states = self.bus.sync_read_all_states()

        # Extract arm joint positions for gravity compensation (exclude gripper)
        arm_positions = []
        for i in range(1, 8):  # joint_1 through joint_7
            motor_name = f"joint_{i}"
            state = states.get(motor_name, {})
            position_deg = state.get("position")
            action_dict[f"{motor_name}.pos"] = position_deg
            action_dict[f"{motor_name}.vel"] = state.get("velocity")
            action_dict[f"{motor_name}.torque"] = state.get("torque")

            # Convert to radians for gravity compensation computation
            if position_deg is not None:
                arm_positions.append(np.deg2rad(position_deg))

        # Add gripper state
        gripper_state = states.get("gripper", {})
        action_dict["gripper.pos"] = gripper_state.get("position")
        action_dict["gripper.vel"] = gripper_state.get("velocity")
        action_dict["gripper.torque"] = gripper_state.get("torque")

        # Apply gravity compensation if enabled
        if self.config.gravity_compensation and self.arm_ik is not None:
            if len(arm_positions) == 7:
                try:
                    q = np.array(arm_positions)

                    # Compute gravity compensation torques using RNEA
                    gravity_torques_raw = self.arm_ik.solve_tau(q)
                    logger.debug(f"Raw gravity torques [Nm]: {gravity_torques_raw}")

                    # Apply gain adjustment for individual robot calibration
                    gravity_torques = gravity_torques_raw * self.config.gravity_compensation_gain
                    logger.debug(
                        f"After gain ({self.config.gravity_compensation_gain}): {gravity_torques}"
                    )

                    # Apply software torque limits (safety)
                    torques_limited = []
                    for i in range(7):
                        limit = self.config.software_torque_limits[i]
                        original = gravity_torques[i]
                        gravity_torques[i] = np.clip(gravity_torques[i], -limit, limit)
                        if abs(original) > limit:
                            torques_limited.append((i + 1, original, gravity_torques[i]))
                    
                    if torques_limited:
                        logger.warning(
                            f"Torque limits applied: "
                            + ", ".join([f"joint_{j}: {o:.3f} -> {c:.3f} Nm" 
                                        for j, o, c in torques_limited])
                        )
                    
                    logger.debug(f"Final torques [Nm]: {gravity_torques}")

                    # Send gravity compensation torques via MIT control
                    self._apply_gravity_compensation(gravity_torques, states)

                except Exception as e:
                    logger.error(f"Gravity compensation error: {e}", exc_info=True)
            else:
                logger.warning(
                    f"Expected 7 arm joint positions, got {len(arm_positions)}. "
                    "Skipping gravity compensation."
                )

        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state + gravity comp: {dt_ms:.1f}ms")

        return action_dict

    def _apply_gravity_compensation(
        self, gravity_torques: np.ndarray, current_states: dict
    ) -> None:
        """
        Apply gravity compensation torques to the arm motors.

        Uses MIT control with soft gains to apply torques while maintaining the current
        position. This reduces the perceived weight of the arm during teleoperation.

        Args:
            gravity_torques: Computed gravity torques [Nm] for 7 arm joints
            current_states: Current motor states (position, velocity, torque)
        """
        commands = {}

        for i in range(7):
            motor_name = f"joint_{i+1}"
            state = current_states.get(motor_name, {})

            # Target position is current position (hold in place with gravity compensation)
            target_pos_deg = state.get("position", 0.0)

            # Use soft control gains to avoid oscillations
            kp = self.config.gravity_comp_position_kp[i]
            kd = self.config.gravity_comp_position_kd[i]

            commands[motor_name] = (
                kp,
                kd,
                target_pos_deg,
                0.0,  # target velocity = 0
                gravity_torques[i],  # feedforward torque
            )

        # Send commands via batch MIT control (efficient CAN bus usage)
        try:
            self.bus._mit_control_batch(commands)
        except Exception as e:
            logger.error(f"Failed to send gravity compensation commands: {e}")

    def send_feedback(self, feedback: dict[str, float]) -> None:
        raise NotImplementedError("Feedback is not yet implemented for OpenArm leader.")

    def disconnect(self) -> None:
        """Disconnect from teleoperator."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Disconnect CAN bus
        # For manual control or gravity compensation, ensure torque is disabled before disconnecting
        disable_torque = self.config.manual_control or self.config.gravity_compensation
        self.bus.disconnect(disable_torque=disable_torque)
        logger.info(f"{self} disconnected.")
