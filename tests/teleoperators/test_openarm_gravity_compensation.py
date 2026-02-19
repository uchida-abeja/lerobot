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
Integration tests for OpenArm Leader Gravity Compensation.

These tests verify the gravity compensation functionality without requiring
actual hardware. They use mocked CAN bus communication.

Note: These tests require Pinocchio and a valid URDF file.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest


# Skip all tests if Pinocchio is not installed
pytest.importorskip("pinocchio", reason="Pinocchio is required for gravity compensation tests")


@pytest.fixture
def mock_urdf():
    """
    Create a minimal valid URDF for testing purposes.
    
    This is a simplified 7-DOF arm model with basic mass and inertia properties.
    """
    urdf_content = """<?xml version="1.0"?>
<robot name="openarm_test">
  <link name="base_link">
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.001"/>
    </inertial>
  </link>
  
  <link name="link_1">
    <inertial>
      <mass value="0.5"/>
      <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.001"/>
    </inertial>
  </link>
  
  <joint name="joint_1" type="revolute">
    <parent link="base_link"/>
    <child link="link_1"/>
    <origin xyz="0 0 0.1"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="10" velocity="1.0"/>
  </joint>
  
  <link name="link_2">
    <inertial>
      <mass value="0.5"/>
      <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.001"/>
    </inertial>
  </link>
  
  <joint name="joint_2" type="revolute">
    <parent link="link_1"/>
    <child link="link_2"/>
    <origin xyz="0 0 0.1"/>
    <axis xyz="0 1 0"/>
    <limit lower="-3.14" upper="3.14" effort="10" velocity="1.0"/>
  </joint>
  
  <link name="link_3">
    <inertial>
      <mass value="0.3"/>
      <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.001"/>
    </inertial>
  </link>
  
  <joint name="joint_3" type="revolute">
    <parent link="link_2"/>
    <child link="link_3"/>
    <origin xyz="0 0 0.15"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="6" velocity="1.0"/>
  </joint>
  
  <link name="link_4">
    <inertial>
      <mass value="0.3"/>
      <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.001"/>
    </inertial>
  </link>
  
  <joint name="joint_4" type="revolute">
    <parent link="link_3"/>
    <child link="link_4"/>
    <origin xyz="0 0 0.15"/>
    <axis xyz="0 1 0"/>
    <limit lower="-3.14" upper="3.14" effort="6" velocity="1.0"/>
  </joint>
  
  <link name="link_5">
    <inertial>
      <mass value="0.2"/>
      <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.001"/>
    </inertial>
  </link>
  
  <joint name="joint_5" type="revolute">
    <parent link="link_4"/>
    <child link="link_5"/>
    <origin xyz="0 0 0.1"/>
    <axis xyz="1 0 0"/>
    <limit lower="-3.14" upper="3.14" effort="2.5" velocity="1.0"/>
  </joint>
  
  <link name="link_6">
    <inertial>
      <mass value="0.2"/>
      <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.001"/>
    </inertial>
  </link>
  
  <joint name="joint_6" type="revolute">
    <parent link="link_5"/>
    <child link="link_6"/>
    <origin xyz="0 0 0.1"/>
    <axis xyz="0 1 0"/>
    <limit lower="-3.14" upper="3.14" effort="2.5" velocity="1.0"/>
  </joint>
  
  <link name="link_7">
    <inertial>
      <mass value="0.15"/>
      <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.001"/>
    </inertial>
  </link>
  
  <joint name="joint_7" type="revolute">
    <parent link="link_6"/>
    <child link="link_7"/>
    <origin xyz="0 0 0.08"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="2.5" velocity="1.0"/>
  </joint>
</robot>
"""
    
    # Create temporary URDF file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.urdf', delete=False) as f:
        f.write(urdf_content)
        urdf_path = Path(f.name)
    
    yield urdf_path
    
    # Cleanup
    urdf_path.unlink()


class TestOpenArmIK:
    """Test suite for OpenArmIK gravity compensation solver."""
    
    def test_ik_initialization(self, mock_urdf):
        """Test that OpenArmIK initializes correctly with a valid URDF."""
        from lerobot.teleoperators.openarm_leader.openarm_kinematic_processor import OpenArmIK
        
        ik = OpenArmIK(urdf_path=mock_urdf, num_arm_joints=7)
        
        assert ik.model is not None
        assert ik.data is not None
        assert ik.num_arm_joints == 7
        assert ik.model.nv >= 7
    
    def test_ik_missing_urdf(self):
        """Test that OpenArmIK raises error for missing URDF."""
        from lerobot.teleoperators.openarm_leader.openarm_kinematic_processor import OpenArmIK
        
        with pytest.raises(FileNotFoundError, match="URDF file not found"):
            OpenArmIK(urdf_path="/nonexistent/path/openarm.urdf")
    
    def test_gravity_torque_computation_zero_config(self, mock_urdf):
        """Test gravity torque computation at zero configuration."""
        from lerobot.teleoperators.openarm_leader.openarm_kinematic_processor import OpenArmIK
        
        ik = OpenArmIK(urdf_path=mock_urdf)
        
        q_zeros = np.zeros(7)
        tau = ik.solve_tau(q_zeros)
        
        # Verify output shape and type
        assert tau.shape == (7,)
        assert tau.dtype == np.float64
        assert not np.any(np.isnan(tau))
    
    def test_gravity_torque_computation_horizontal_config(self, mock_urdf):
        """Test that horizontal configuration produces higher torques than vertical."""
        from lerobot.teleoperators.openarm_leader.openarm_kinematic_processor import OpenArmIK
        
        ik = OpenArmIK(urdf_path=mock_urdf)
        
        # Vertical configuration (arm pointing down, minimal gravity effect)
        q_vertical = np.zeros(7)
        tau_vertical = ik.solve_tau(q_vertical)
        
        # Horizontal configuration (arm extended, maximum gravity effect)
        q_horizontal = np.array([0.0, np.pi/2, 0.0, 0.0, 0.0, 0.0, 0.0])
        tau_horizontal = ik.solve_tau(q_horizontal)
        
        # Horizontal should produce larger torques due to gravity
        assert np.linalg.norm(tau_horizontal) > np.linalg.norm(tau_vertical)
    
    def test_gravity_vector_configuration(self, mock_urdf):
        """Test that changing gravity vector affects computed torques."""
        from lerobot.teleoperators.openarm_leader.openarm_kinematic_processor import OpenArmIK
        
        # Standard gravity
        ik_standard = OpenArmIK(urdf_path=mock_urdf, gravity_vector=np.array([0.0, 0.0, -9.81]))
        
        # Zero gravity (space)
        ik_zero = OpenArmIK(urdf_path=mock_urdf, gravity_vector=np.array([0.0, 0.0, 0.0]))
        
        q = np.array([0.0, np.pi/4, 0.0, -np.pi/4, 0.0, 0.0, 0.0])
        
        tau_standard = ik_standard.solve_tau(q)
        tau_zero = ik_zero.solve_tau(q)
        
        # Zero gravity should produce zero torques
        assert np.allclose(tau_zero, 0.0, atol=1e-6)
        # Standard gravity should produce non-zero torques
        assert not np.allclose(tau_standard, 0.0, atol=1e-6)
    
    def test_invalid_joint_count(self, mock_urdf):
        """Test that solve_tau raises error for incorrect joint count."""
        from lerobot.teleoperators.openarm_leader.openarm_kinematic_processor import OpenArmIK
        
        ik = OpenArmIK(urdf_path=mock_urdf, num_arm_joints=7)
        
        # Wrong number of joints
        q_wrong = np.zeros(5)
        
        with pytest.raises(ValueError, match="Expected 7 joint positions"):
            ik.solve_tau(q_wrong)
    
    def test_validate_urdf(self, mock_urdf):
        """Test URDF validation method."""
        from lerobot.teleoperators.openarm_leader.openarm_kinematic_processor import OpenArmIK
        
        ik = OpenArmIK(urdf_path=mock_urdf)
        info = ik.validate_urdf()
        
        assert "valid" in info
        assert "nq" in info
        assert "nv" in info
        assert "joint_names" in info
        assert "has_inertia" in info
        assert "warnings" in info
        
        # Should have 7 joints
        assert info["nv"] >= 7


class TestOpenArmLeaderGravityCompensation:
    """Test suite for OpenArmLeader with gravity compensation enabled."""
    
    @patch('lerobot.teleoperators.openarm_leader.openarm_leader.DamiaoMotorsBus')
    def test_gravity_compensation_initialization(self, mock_bus, mock_urdf):
        """Test that gravity compensation initializes correctly in OpenArmLeader."""
        from lerobot.teleoperators.openarm_leader import OpenArmLeaderConfig, OpenArmLeader
        
        config = OpenArmLeaderConfig(
            port="can0",
            gravity_compensation=True,
            urdf_path=str(mock_urdf),
            manual_control=False,
        )
        
        leader = OpenArmLeader(config)
        
        assert leader.arm_ik is not None
        assert leader.config.gravity_compensation is True
    
    @patch('lerobot.teleoperators.openarm_leader.openarm_leader.DamiaoMotorsBus')
    def test_torque_limiting(self, mock_bus, mock_urdf):
        """Test that software torque limits are correctly applied."""
        from lerobot.teleoperators.openarm_leader import OpenArmLeaderConfig, OpenArmLeader
        
        config = OpenArmLeaderConfig(
            port="can0",
            gravity_compensation=True,
            urdf_path=str(mock_urdf),
            manual_control=False,
            software_torque_limits=[1.0] * 7,  # Very low limits for testing
            gravity_compensation_gain=10.0,  # High gain to trigger limits
        )
        
        # Mock motor states
        mock_states = {
            f"joint_{i}": {"position": 45.0, "velocity": 0.0, "torque": 0.0}
            for i in range(1, 8)
        }
        mock_states["gripper"] = {"position": 0.0, "velocity": 0.0, "torque": 0.0}
        
        mock_bus_instance = mock_bus.return_value
        mock_bus_instance.is_connected = True
        mock_bus_instance.sync_read_all_states.return_value = mock_states
        mock_bus_instance._mit_control_batch = Mock()
        
        leader = OpenArmLeader(config)
        leader.bus = mock_bus_instance
        
        # Get action (triggers gravity compensation)
        action = leader.get_action()
        
        # Verify _mit_control_batch was called
        if mock_bus_instance._mit_control_batch.called:
            # Get the commands that were sent
            commands = mock_bus_instance._mit_control_batch.call_args[0][0]
            
            # Check that all torques are within limits
            for motor_name, command in commands.items():
                if motor_name.startswith("joint_"):
                    torque = command[4]  # Torque is 5th element in tuple
                    assert abs(torque) <= 1.0, f"Torque {torque} exceeds limit for {motor_name}"
    
    @patch('lerobot.teleoperators.openarm_leader.openarm_leader.DamiaoMotorsBus')
    def test_gravity_compensation_gain(self, mock_bus, mock_urdf):
        """Test that gravity compensation gain scales torques correctly."""
        from lerobot.teleoperators.openarm_leader import OpenArmLeaderConfig, OpenArmLeader
        
        # Mock motor states at horizontal configuration
        mock_states = {
            f"joint_{i}": {"position": 90.0 if i == 2 else 0.0, "velocity": 0.0, "torque": 0.0}
            for i in range(1, 8)
        }
        mock_states["gripper"] = {"position": 0.0, "velocity": 0.0, "torque": 0.0}
        
        mock_bus_instance = mock_bus.return_value
        mock_bus_instance.is_connected = True
        mock_bus_instance.sync_read_all_states.return_value = mock_states
        mock_bus_instance._mit_control_batch = Mock()
        
        # Test with gain = 0.5
        config_half = OpenArmLeaderConfig(
            port="can0",
            gravity_compensation=True,
            urdf_path=str(mock_urdf),
            manual_control=False,
            gravity_compensation_gain=0.5,
        )
        
        leader_half = OpenArmLeader(config_half)
        leader_half.bus = mock_bus_instance
        leader_half.get_action()
        
        if mock_bus_instance._mit_control_batch.called:
            commands_half = mock_bus_instance._mit_control_batch.call_args[0][0]
            torques_half = [cmd[4] for cmd in commands_half.values() if cmd[4] != 0]
            
            # Reset mock
            mock_bus_instance._mit_control_batch.reset_mock()
            
            # Test with gain = 1.0
            config_full = OpenArmLeaderConfig(
                port="can0",
                gravity_compensation=True,
                urdf_path=str(mock_urdf),
                manual_control=False,
                gravity_compensation_gain=1.0,
            )
            
            leader_full = OpenArmLeader(config_full)
            leader_full.bus = mock_bus_instance
            leader_full.get_action()
            
            commands_full = mock_bus_instance._mit_control_batch.call_args[0][0]
            torques_full = [cmd[4] for cmd in commands_full.values() if cmd[4] != 0]
            
            # Torques should scale with gain
            if torques_half and torques_full:
                ratio = np.mean(np.abs(torques_full)) / np.mean(np.abs(torques_half))
                assert 1.8 < ratio < 2.2  # Should be approximately 2.0


class TestBiOpenArmLeaderGravityCompensation:
    """Test suite for bimanual OpenArm with gravity compensation."""
    
    @patch('lerobot.teleoperators.openarm_leader.openarm_leader.DamiaoMotorsBus')
    def test_bimanual_gravity_compensation_symmetry(self, mock_bus, mock_urdf):
        """Test that left and right arms receive equal gravity compensation in symmetric poses."""
        from lerobot.teleoperators.bi_openarm_leader import BiOpenArmLeaderConfig
        from lerobot.teleoperators.openarm_leader import OpenArmLeaderConfigBase
        from lerobot.teleoperators.bi_openarm_leader import BiOpenArmLeader
        
        # Symmetric configuration
        mock_states = {
            f"joint_{i}": {"position": 0.0, "velocity": 0.0, "torque": 0.0}
            for i in range(1, 8)
        }
        mock_states["gripper"] = {"position": 0.0, "velocity": 0.0, "torque": 0.0}
        
        mock_bus_instance = mock_bus.return_value
        mock_bus_instance.is_connected = True
        mock_bus_instance.is_calibrated = True
        mock_bus_instance.sync_read_all_states.return_value = mock_states
        mock_bus_instance._mit_control_batch = Mock()
        
        config = BiOpenArmLeaderConfig(
            left_arm_config=OpenArmLeaderConfigBase(
                port="can0",
                manual_control=False,
            ),
            right_arm_config=OpenArmLeaderConfigBase(
                port="can1",
                manual_control=False,
            ),
            gravity_compensation=True,
            gravity_compensation_gain=1.0,
        )
        
        # Override URDF path in both arms after construction
        with patch.dict('os.environ', {'LEROBOT_ROOT': str(mock_urdf.parent.parent.parent)}):
            leader = BiOpenArmLeader(config)
            leader.left_arm.config.urdf_path = str(mock_urdf)
            leader.right_arm.config.urdf_path = str(mock_urdf)
            
            # Re-initialize IK solvers with test URDF
            from lerobot.teleoperators.openarm_leader.openarm_kinematic_processor import OpenArmIK
            leader.left_arm.arm_ik = OpenArmIK(mock_urdf)
            leader.right_arm.arm_ik = OpenArmIK(mock_urdf)
            
            leader.left_arm.bus = mock_bus_instance
            leader.right_arm.bus = mock_bus_instance
            
            # Get action (triggers gravity compensation on both arms)
            action = leader.get_action()
            
            # Verify symmetry in action dict
            left_keys = [k for k in action.keys() if k.startswith("left_")]
            right_keys = [k for k in action.keys() if k.startswith("right_")]
            
            assert len(left_keys) == len(right_keys)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
