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

"""
OpenArm Inverse Dynamics Solver for Gravity Compensation

This module provides gravity compensation torque computation for OpenArm using
Pinocchio's Recursive Newton-Euler Algorithm (RNEA).

Key Features:
- Computes gravity compensation torques from joint positions
- Configurable gravity vector (for different mounting orientations)
- Software torque limits for safety
- Designed for future friction compensation extension

Note:
    URDF physical parameters (mass, inertia) may not match the actual robot due to
    manufacturing variations. Use gravity_compensation_gain to calibrate.
"""

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class OpenArmIK:
    """
    OpenArm Inverse Dynamics Solver for Gravity Compensation.

    Uses Pinocchio's RNEA (Recursive Newton-Euler Algorithm) to compute
    gravity compensation torques based on the current joint configuration.

    The solver computes: tau = G(q)
    where G(q) is the gravity torque vector, obtained from:
        tau = RNEA(q, v=0, a=0)

    Attributes:
        model: Pinocchio robot model loaded from URDF
        data: Pinocchio data structure (cached for performance)
        num_arm_joints: Number of arm joints (excluding gripper)

    Example:
        >>> ik = OpenArmIK("openarm.urdf", num_arm_joints=7)
        >>> q = np.array([0.0, 0.5, 0.0, -1.0, 0.0, 0.5, 0.0])  # radians
        >>> gravity_torques = ik.solve_tau(q)
        >>> print(gravity_torques)  # [tau_1, tau_2, ..., tau_7] in Nm
    """

    def __init__(
        self,
        urdf_path: str | Path,
        gravity_vector: np.ndarray = np.array([0.0, 0.0, -9.81]),
        num_arm_joints: int = 7,
    ):
        """
        Initialize the OpenArm IK solver.

        Args:
            urdf_path: Path to the OpenArm URDF file
            gravity_vector: Gravity vector in base frame [x, y, z] in m/s²
                           Default: [0, 0, -9.81] (standard gravity pointing down)
                           Can be modified for wall-mounted or tilted installations
            num_arm_joints: Number of arm joints (default: 7, excluding gripper)

        Raises:
            FileNotFoundError: If URDF file does not exist
            ImportError: If Pinocchio is not installed
            RuntimeError: If URDF model cannot be loaded
        """
        self.urdf_path = Path(urdf_path)
        if not self.urdf_path.exists():
            raise FileNotFoundError(
                f"URDF file not found: {self.urdf_path}\n"
                f"Please obtain the OpenArm URDF from: https://github.com/enactic/openarm\n"
                f"See {self.urdf_path.parent / 'URDF_README.md'} for instructions."
            )

        try:
            import pinocchio as pin
        except ImportError as e:
            raise ImportError(
                "Pinocchio is required for gravity compensation. "
                "Install with: pip install pin"
            ) from e

        self._pin = pin

        # Load Pinocchio model from URDF
        logger.info(f"Loading OpenArm URDF from: {self.urdf_path}")
        try:
            self.model = pin.buildModelFromUrdf(str(self.urdf_path))
            # Create data structure (cached as instance variable to minimize latency)
            self.data = self.model.createData()
        except Exception as e:
            raise RuntimeError(f"Failed to load URDF model: {e}") from e

        # Configure gravity vector
        self.model.gravity.linear = gravity_vector

        self.num_arm_joints = num_arm_joints

        logger.info(f"OpenArmIK initialized:")
        logger.info(f"  - DOF: {self.model.nv}")
        logger.info(f"  - Arm joints: {self.num_arm_joints}")
        logger.info(f"  - Gravity vector: {gravity_vector}")

        # Validate model
        if self.model.nv < self.num_arm_joints:
            logger.warning(
                f"URDF has only {self.model.nv} DOF, expected at least {self.num_arm_joints}"
            )

    def solve_tau(self, q: np.ndarray, v: np.ndarray | None = None) -> np.ndarray:
        """
        Compute gravity compensation torques for the current joint configuration.

        This method uses RNEA with zero velocity and acceleration to isolate
        the gravity term: tau = M(q)*0 + C(q,0)*0 + G(q) = G(q)

        Args:
            q: Joint positions in radians, shape (num_arm_joints,)
               Order: [joint_1, joint_2, joint_3, joint_4, joint_5, joint_6, joint_7]
            v: Joint velocities in rad/s (optional, for future friction compensation)
               Currently unused, reserved for future extension

        Returns:
            gravity_torques: Gravity compensation torques in Nm, shape (num_arm_joints,)
                            These torques should be applied to counteract gravity

        Raises:
            ValueError: If q has incorrect shape
            RuntimeError: If RNEA computation fails

        Note:
            - Returned torques are in the same order as input positions
            - Torques should be clipped to motor limits before sending to hardware
            - If computation fails, returns zero torques for safety
        """
        # Validate input shape
        if q.shape[0] != self.num_arm_joints:
            raise ValueError(
                f"Expected {self.num_arm_joints} joint positions, got {q.shape[0]}"
            )

        try:
            # Prepare velocity and acceleration vectors (zeros for pure gravity)
            v_zeros = np.zeros(self.model.nv)
            a_zeros = np.zeros(self.model.nv)

            # Extend q to full model size if necessary
            if self.model.nv > self.num_arm_joints:
                q_full = np.zeros(self.model.nv)
                q_full[: self.num_arm_joints] = q
                q_input = q_full
            else:
                q_input = q

            # RNEA: Recursive Newton-Euler Algorithm
            # Computes inverse dynamics: tau = M(q)a + C(q,v)v + G(q)
            # With v=0, a=0: tau = G(q) (gravity torques only)
            tau = self._pin.rnea(self.model, self.data, q_input, v_zeros, a_zeros)

            # Return only arm joint torques (exclude gripper or additional joints)
            return tau[: self.num_arm_joints]

        except Exception as e:
            logger.error(f"Gravity torque computation failed: {e}")
            logger.error(f"Joint positions: {q}")
            # Return zero torques on failure (safe fallback)
            return np.zeros(self.num_arm_joints)

    def validate_urdf(self) -> dict[str, any]:
        """
        Validate the loaded URDF model and return diagnostic information.

        Returns:
            dict with validation results:
                - 'valid': bool, whether model passed basic validation
                - 'nq': number of configuration variables
                - 'nv': number of velocity variables
                - 'joint_names': list of joint names
                - 'has_inertia': bool, whether links have inertia data
                - 'warnings': list of warning messages

        Example:
            >>> ik = OpenArmIK("openarm.urdf")
            >>> info = ik.validate_urdf()
            >>> if not info['valid']:
            >>>     print("Validation warnings:", info['warnings'])
        """
        warnings = []
        joint_names = [self.model.names[i] for i in range(1, self.model.njoints)]

        # Check if all links have inertia data
        has_inertia = True
        for i in range(1, self.model.njoints):
            inertia = self.model.inertias[i]
            if np.allclose(inertia.mass, 0.0):
                warnings.append(f"Link {self.model.names[i]} has zero mass")
                has_inertia = False

        # Check DOF count
        if self.model.nv != self.num_arm_joints:
            warnings.append(
                f"Model has {self.model.nv} DOF, expected {self.num_arm_joints}"
            )

        return {
            "valid": len(warnings) == 0,
            "nq": self.model.nq,
            "nv": self.model.nv,
            "joint_names": joint_names,
            "has_inertia": has_inertia,
            "warnings": warnings,
        }
