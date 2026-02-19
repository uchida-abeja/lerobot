# OpenArm URDF Model

This directory should contain the OpenArm 7-DOF robot URDF model file for gravity compensation calculations.

## Required File

- `openarm.urdf` - OpenArm 7-DOF arm URDF model

## How to Obtain the URDF

The OpenArm URDF model is not included in this repository and must be obtained from the official OpenArm repository:

**Source:** [github.com/enactic/openarm](https://github.com/enactic/openarm)

### Steps:

1. Clone or download the official OpenArm repository
2. Navigate to the URDF/descriptions directory
3. Locate the 7-DOF arm URDF file (typically named `openarm_7dof.urdf` or similar)
4. Copy it to this directory as `openarm.urdf`

## URDF Requirements

The URDF must include:

- 7 arm joint definitions (joint_1 through joint_7)
- Link mass properties (mass, inertia tensor)
- Proper coordinate frame definitions
- Joint limits and dynamics parameters

## Version Information

Please document the URDF version you are using:

- **Source URL:** https://github.com/enactic/openarm
- **Version/Commit:** 73ca62f3e4d04f61d2d1efa112927fe763d4b1f1 (tag: 1.0.3)
- **Date Obtained:** 2026-01-15
- **Commit Message:** urdf/ros2_control: add control gain config (#38)
- **Notes:** This version includes configurable control gains for ROS2 control 

## Important Notes

### Individual Robot Variations

⚠️ **The physical parameters (mass, inertia) in the URDF may not exactly match your physical robot due to:**

- 3D printing density variations
- Different infill percentages
- Cable routing and additional wiring
- Mounted cameras or sensors
- Manufacturing tolerances

**Solution:** Use the `gravity_compensation_gain` parameter in the config to scale the computed torques to match your specific robot. See the [calibration guide](../../../docs/source/openarm_gravity_calibration.mdx) for details.

### Coordinate System

Ensure the URDF coordinate system matches the OpenArm convention:
- Gravity direction: typically [0, 0, -9.81] in the base frame
- Joint zero positions should align with the physical robot's zero position

## Troubleshooting

If gravity compensation is not working as expected:

1. **Verify URDF exists:** Check that `openarm.urdf` is in this directory
2. **Check file permissions:** Ensure the file is readable
3. **Validate URDF:** Use `check_urdf` tool or Pinocchio to validate the model
4. **Calibrate gain:** Follow the calibration procedure to adjust `gravity_compensation_gain`

For more information, see [OpenArm documentation](../../../docs/source/openarm.mdx).
