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
Unit tests for OpenArm ForceObserver.

Tests external torque estimation, filtering, and clamping without requiring hardware or URDF.
"""

import math
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Add src to path
LEROBOT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(LEROBOT_ROOT / "src"))


class TestForceObserverInit:
    """Test ForceObserver initialization."""

    def test_init_requires_urdf_path(self):
        """ForceObserver should require a valid URDF path."""
        from lerobot.teleoperators.openarm_leader.force_observer import ForceObserver

        # Mock OpenArmIK to avoid URDF dependency
        with patch(
            "lerobot.teleoperators.openarm_leader.force_observer.OpenArmIK"
        ) as mock_ik:
            mock_ik.return_value = MagicMock()

            observer = ForceObserver(
                urdf_path="/fake/path.urdf",
                gravity_vector=[0.0, 0.0, -9.81],
                gravity_gain=1.0,
                lpf_cutoff_hz=10.0,
                torque_limits=[1.0] * 7,
                num_arm_joints=7,
            )

            assert observer.num_arm_joints == 7
            assert observer.gravity_gain == 1.0
            assert observer.lpf_cutoff_hz == 10.0
            assert len(observer.torque_limits) == 7

    def test_init_lpf_state(self):
        """LPF state should be initialized to zero."""
        from lerobot.teleoperators.openarm_leader.force_observer import ForceObserver

        with patch(
            "lerobot.teleoperators.openarm_leader.force_observer.OpenArmIK"
        ) as mock_ik:
            mock_ik.return_value = MagicMock()

            observer = ForceObserver(
                urdf_path="/fake/path.urdf",
                gravity_vector=[0.0, 0.0, -9.81],
                gravity_gain=1.0,
                lpf_cutoff_hz=10.0,
                torque_limits=[1.0] * 7,
                num_arm_joints=7,
            )

            assert observer._lpf_state.shape == (7,)
            assert np.allclose(observer._lpf_state, 0.0)


class TestForceObserverEstimate:
    """Test external torque estimation."""

    def test_estimate_free_space_zero_torque(self):
        """In free space (no external force), estimated torque should be near zero."""
        from lerobot.teleoperators.openarm_leader.force_observer import ForceObserver

        with patch(
            "lerobot.teleoperators.openarm_leader.force_observer.OpenArmIK"
        ) as mock_ik:
            # Mock IK to return gravity torques matching measured torques
            # tau_ext = tau_meas - tau_grav = tau_grav - tau_grav = 0
            mock_ik_instance = MagicMock()
            mock_ik_instance.solve_tau.return_value = np.array(
                [0.5, 0.5, 0.3, 0.3, 0.1, 0.1, 0.1]
            )
            mock_ik.return_value = mock_ik_instance

            observer = ForceObserver(
                urdf_path="/fake/path.urdf",
                gravity_vector=[0.0, 0.0, -9.81],
                gravity_gain=1.0,
                lpf_cutoff_hz=0.0,  # No filtering for this test
                torque_limits=[10.0] * 7,
                num_arm_joints=7,
            )

            # Observation with measured torques = gravity torques (no external force)
            observation = {
                f"joint_{i}.pos": 0.0 for i in range(1, 8)
            }
            observation.update(
                {
                    "joint_1.torque": 0.5,
                    "joint_2.torque": 0.5,
                    "joint_3.torque": 0.3,
                    "joint_4.torque": 0.3,
                    "joint_5.torque": 0.1,
                    "joint_6.torque": 0.1,
                    "joint_7.torque": 0.1,
                }
            )

            tau_ext = observer.estimate(observation, dt_s=None)

            assert tau_ext.shape == (7,)
            assert np.allclose(tau_ext, 0.0, atol=1e-6)

    def test_estimate_contact_positive_torque(self):
        """When follower touches an object, estimated torque should be positive."""
        from lerobot.teleoperators.openarm_leader.force_observer import ForceObserver

        with patch(
            "lerobot.teleoperators.openarm_leader.force_observer.OpenArmIK"
        ) as mock_ik:
            mock_ik_instance = MagicMock()
            mock_ik_instance.solve_tau.return_value = np.array(
                [0.5, 0.5, 0.3, 0.3, 0.1, 0.1, 0.1]
            )
            mock_ik.return_value = mock_ik_instance

            observer = ForceObserver(
                urdf_path="/fake/path.urdf",
                gravity_vector=[0.0, 0.0, -9.81],
                gravity_gain=1.0,
                lpf_cutoff_hz=0.0,
                torque_limits=[10.0] * 7,
                num_arm_joints=7,
            )

            # Measured torques include external force (0.2 Nm more than gravity)
            observation = {
                f"joint_{i}.pos": 0.0 for i in range(1, 8)
            }
            observation.update(
                {
                    "joint_1.torque": 0.7,  # 0.5 + 0.2 external
                    "joint_2.torque": 0.5,
                    "joint_3.torque": 0.3,
                    "joint_4.torque": 0.3,
                    "joint_5.torque": 0.1,
                    "joint_6.torque": 0.1,
                    "joint_7.torque": 0.1,
                }
            )

            tau_ext = observer.estimate(observation, dt_s=None)

            assert tau_ext.shape == (7,)
            assert tau_ext[0] > 0.19  # First joint has external torque
            assert np.allclose(tau_ext[1:], 0.0, atol=1e-6)

    def test_estimate_torque_clamping(self):
        """Estimated torques should be clamped to limits."""
        from lerobot.teleoperators.openarm_leader.force_observer import ForceObserver

        with patch(
            "lerobot.teleoperators.openarm_leader.force_observer.OpenArmIK"
        ) as mock_ik:
            mock_ik_instance = MagicMock()
            mock_ik_instance.solve_tau.return_value = np.zeros(7)
            mock_ik.return_value = mock_ik_instance

            observer = ForceObserver(
                urdf_path="/fake/path.urdf",
                gravity_vector=[0.0, 0.0, -9.81],
                gravity_gain=1.0,
                lpf_cutoff_hz=0.0,
                torque_limits=[0.5] * 7,  # Limit to 0.5 Nm
                num_arm_joints=7,
            )

            # Measured torques exceed limits
            observation = {
                f"joint_{i}.pos": 0.0 for i in range(1, 8)
            }
            observation.update(
                {
                    f"joint_{i}.torque": 2.0 for i in range(1, 8)  # Way above limit
                }
            )

            tau_ext = observer.estimate(observation, dt_s=None)

            # All should be clamped to +0.5
            assert np.allclose(tau_ext, 0.5)

    def test_estimate_negative_torque_clamping(self):
        """Negative torques should be clamped symmetrically."""
        from lerobot.teleoperators.openarm_leader.force_observer import ForceObserver

        with patch(
            "lerobot.teleoperators.openarm_leader.force_observer.OpenArmIK"
        ) as mock_ik:
            mock_ik_instance = MagicMock()
            mock_ik_instance.solve_tau.return_value = np.zeros(7)
            mock_ik.return_value = mock_ik_instance

            observer = ForceObserver(
                urdf_path="/fake/path.urdf",
                gravity_vector=[0.0, 0.0, -9.81],
                gravity_gain=1.0,
                lpf_cutoff_hz=0.0,
                torque_limits=[0.5] * 7,
                num_arm_joints=7,
            )

            observation = {
                f"joint_{i}.pos": 0.0 for i in range(1, 8)
            }
            observation.update(
                {
                    f"joint_{i}.torque": -2.0 for i in range(1, 8)
                }
            )

            tau_ext = observer.estimate(observation, dt_s=None)

            # All should be clamped to -0.5
            assert np.allclose(tau_ext, -0.5)


class TestForceObserverFiltering:
    """Test low-pass filtering of external torque."""

    def test_lpf_disabled_no_filtering(self):
        """When LPF cutoff is <= 0, no filtering should be applied."""
        from lerobot.teleoperators.openarm_leader.force_observer import ForceObserver

        with patch(
            "lerobot.teleoperators.openarm_leader.force_observer.OpenArmIK"
        ) as mock_ik:
            mock_ik_instance = MagicMock()
            mock_ik_instance.solve_tau.return_value = np.zeros(7)
            mock_ik.return_value = mock_ik_instance

            observer = ForceObserver(
                urdf_path="/fake/path.urdf",
                gravity_vector=[0.0, 0.0, -9.81],
                gravity_gain=1.0,
                lpf_cutoff_hz=0.0,  # Disabled
                torque_limits=[10.0] * 7,
                num_arm_joints=7,
            )

            observation = {
                f"joint_{i}.pos": 0.0 for i in range(1, 8)
            }
            observation.update(
                {f"joint_{i}.torque": 1.0 for i in range(1, 8)}
            )

            tau_ext = observer.estimate(observation, dt_s=0.01)

            # No filtering, should be 1.0
            assert np.allclose(tau_ext, 1.0)

    def test_lpf_smooths_step_response(self):
        """LPF should smooth sudden changes in torque."""
        from lerobot.teleoperators.openarm_leader.force_observer import ForceObserver

        with patch(
            "lerobot.teleoperators.openarm_leader.force_observer.OpenArmIK"
        ) as mock_ik:
            mock_ik_instance = MagicMock()
            mock_ik_instance.solve_tau.return_value = np.zeros(7)
            mock_ik.return_value = mock_ik_instance

            observer = ForceObserver(
                urdf_path="/fake/path.urdf",
                gravity_vector=[0.0, 0.0, -9.81],
                gravity_gain=1.0,
                lpf_cutoff_hz=10.0,  # 10 Hz cutoff
                torque_limits=[10.0] * 7,
                num_arm_joints=7,
            )

            observation_zero = {
                f"joint_{i}.pos": 0.0 for i in range(1, 8)
            }
            observation_zero.update(
                {f"joint_{i}.torque": 0.0 for i in range(1, 8)}
            )

            observation_step = {
                f"joint_{i}.pos": 0.0 for i in range(1, 8)
            }
            observation_step.update(
                {f"joint_{i}.torque": 1.0 for i in range(1, 8)}
            )

            # First call with zero torque
            tau_ext_zero = observer.estimate(observation_zero, dt_s=None)
            assert np.allclose(tau_ext_zero, 0.0)

            # Step input: torque jumps to 1.0
            tau_ext_step = observer.estimate(observation_step, dt_s=0.01)

            # Output should be between 0 and 1 due to filter
            assert np.all(tau_ext_step > 0.0)
            assert np.all(tau_ext_step < 1.0)

            # Second step: output should increase further (exponential approach)
            tau_ext_step_2 = observer.estimate(observation_step, dt_s=0.01)
            assert np.all(tau_ext_step_2 > tau_ext_step)
            assert np.all(tau_ext_step_2 < 1.0)

    def test_lpf_alpha_calculation(self):
        """LPF alpha should be correctly calculated from cutoff and dt."""
        from lerobot.teleoperators.openarm_leader.force_observer import ForceObserver

        with patch(
            "lerobot.teleoperators.openarm_leader.force_observer.OpenArmIK"
        ) as mock_ik:
            mock_ik_instance = MagicMock()
            mock_ik_instance.solve_tau.return_value = np.zeros(7)
            mock_ik.return_value = mock_ik_instance

            observer = ForceObserver(
                urdf_path="/fake/path.urdf",
                gravity_vector=[0.0, 0.0, -9.81],
                gravity_gain=1.0,
                lpf_cutoff_hz=1.0,  # 1 Hz cutoff
                torque_limits=[10.0] * 7,
                num_arm_joints=7,
            )

            # For a given cutoff (1 Hz) and dt (0.1 s):
            # alpha = 1 - exp(-2*pi*f_c*dt) = 1 - exp(-2*pi*1*0.1) ≈ 0.468
            dt = 0.1
            fc = 1.0
            expected_alpha = 1.0 - math.exp(-2.0 * math.pi * fc * dt)

            observation_init = {
                f"joint_{i}.pos": 0.0 for i in range(1, 8)
            }
            observation_init.update(
                {f"joint_{i}.torque": 0.0 for i in range(1, 8)}
            )

            # Prime the filter
            observer.estimate(observation_init, dt_s=None)

            observation_step = {
                f"joint_{i}.pos": 0.0 for i in range(1, 8)
            }
            observation_step.update(
                {f"joint_{i}.torque": 1.0 for i in range(1, 8)}
            )

            tau_ext = observer.estimate(observation_step, dt_s=dt)

            # Expected: new_state = old_state + alpha*(input - old_state)
            #         = 0 + alpha*(1 - 0) = alpha
            expected_output = expected_alpha
            assert np.allclose(tau_ext, expected_output, atol=1e-3)


class TestForceObserverGravityGain:
    """Test gravity gain scaling."""

    def test_gravity_gain_scales_gravity_torque(self):
        """Gravity gain should scale the subtracted gravity torque."""
        from lerobot.teleoperators.openarm_leader.force_observer import ForceObserver

        with patch(
            "lerobot.teleoperators.openarm_leader.force_observer.OpenArmIK"
        ) as mock_ik:
            mock_ik_instance = MagicMock()
            mock_ik_instance.solve_tau.return_value = np.array(
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
            )
            mock_ik.return_value = mock_ik_instance

            observer_gain_1 = ForceObserver(
                urdf_path="/fake/path.urdf",
                gravity_vector=[0.0, 0.0, -9.81],
                gravity_gain=1.0,
                lpf_cutoff_hz=0.0,
                torque_limits=[10.0] * 7,
                num_arm_joints=7,
            )

            observer_gain_0_5 = ForceObserver(
                urdf_path="/fake/path.urdf",
                gravity_vector=[0.0, 0.0, -9.81],
                gravity_gain=0.5,
                lpf_cutoff_hz=0.0,
                torque_limits=[10.0] * 7,
                num_arm_joints=7,
            )

            observation = {
                f"joint_{i}.pos": 0.0 for i in range(1, 8)
            }
            observation.update(
                {f"joint_{i}.torque": 2.0 for i in range(1, 8)}
            )

            # With gain=1.0: tau_ext = 2.0 - 1.0 = 1.0
            tau_ext_1 = observer_gain_1.estimate(observation, dt_s=None)
            assert np.allclose(tau_ext_1, 1.0)

            # With gain=0.5: tau_ext = 2.0 - 0.5 = 1.5
            tau_ext_0_5 = observer_gain_0_5.estimate(observation, dt_s=None)
            assert np.allclose(tau_ext_0_5, 1.5)


class TestForceObserverObservationHandling:
    """Test handling of observation dictionaries."""

    def test_missing_joint_position_defaults_to_zero(self):
        """Missing joint positions should default to 0.0."""
        from lerobot.teleoperators.openarm_leader.force_observer import ForceObserver

        with patch(
            "lerobot.teleoperators.openarm_leader.force_observer.OpenArmIK"
        ) as mock_ik:
            mock_ik_instance = MagicMock()
            mock_ik_instance.solve_tau.return_value = np.zeros(7)
            mock_ik.return_value = mock_ik_instance

            observer = ForceObserver(
                urdf_path="/fake/path.urdf",
                gravity_vector=[0.0, 0.0, -9.81],
                gravity_gain=1.0,
                lpf_cutoff_hz=0.0,
                torque_limits=[10.0] * 7,
                num_arm_joints=7,
            )

            # Sparse observation (no position, only torque)
            observation = {
                "joint_1.torque": 0.5,
                "joint_2.torque": 0.5,
            }

            # Should not raise, defaults to zero positions
            tau_ext = observer.estimate(observation, dt_s=None)
            assert tau_ext.shape == (7,)

    def test_missing_torque_defaults_to_zero(self):
        """Missing torques should default to 0.0."""
        from lerobot.teleoperators.openarm_leader.force_observer import ForceObserver

        with patch(
            "lerobot.teleoperators.openarm_leader.force_observer.OpenArmIK"
        ) as mock_ik:
            mock_ik_instance = MagicMock()
            mock_ik_instance.solve_tau.return_value = np.ones(7)
            mock_ik.return_value = mock_ik_instance

            observer = ForceObserver(
                urdf_path="/fake/path.urdf",
                gravity_vector=[0.0, 0.0, -9.81],
                gravity_gain=1.0,
                lpf_cutoff_hz=0.0,
                torque_limits=[10.0] * 7,
                num_arm_joints=7,
            )

            # Only positions, no torques
            observation = {f"joint_{i}.pos": 0.0 for i in range(1, 8)}

            tau_ext = observer.estimate(observation, dt_s=None)

            # tau_ext = 0 (measured) - 1.0 (gravity) = -1.0
            assert np.allclose(tau_ext, -1.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
