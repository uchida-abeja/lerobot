# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
Simple script to control a robot from teleoperation.

Example:

```shell
lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/tty.usbmodem58760431541 \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 1920, height: 1080, fps: 30}}" \
    --robot.id=black \
    --teleop.type=so101_leader \
    --teleop.port=/dev/tty.usbmodem58760431551 \
    --teleop.id=blue \
    --display_data=true
```

Example teleoperation with bimanual so100:

```shell
lerobot-teleoperate \
  --robot.type=bi_so_follower \
  --robot.left_arm_config.port=/dev/tty.usbmodem5A460822851 \
  --robot.right_arm_config.port=/dev/tty.usbmodem5A460814411 \
  --robot.id=bimanual_follower \
  --robot.left_arm_config.cameras='{
    wrist: {"type": "opencv", "index_or_path": 1, "width": 640, "height": 480, "fps": 30},
  }' --robot.right_arm_config.cameras='{
    wrist: {"type": "opencv", "index_or_path": 2, "width": 640, "height": 480, "fps": 30},
  }' \
  --teleop.type=bi_so_leader \
  --teleop.left_arm_config.port=/dev/tty.usbmodem5A460852721 \
  --teleop.right_arm_config.port=/dev/tty.usbmodem5A460819811 \
  --teleop.id=bimanual_leader \
  --display_data=true
```

"""

import logging
import time
from dataclasses import asdict, dataclass
from pprint import pformat

import rerun as rr

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.processor import (
    RobotAction,
    RobotObservation,
    RobotProcessorPipeline,
    make_default_processors,
)
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    bi_openarm_follower,
    bi_so_follower,
    earthrover_mini_plus,
    hope_jr,
    koch_follower,
    make_robot_from_config,
    omx_follower,
    openarm_follower,
    reachy2,
    so_follower,
    unitree_g1 as unitree_g1_robot,
)
from lerobot.robots.bi_openarm_follower import BiOpenArmFollower
from lerobot.robots.openarm_follower import OpenArmFollower
from lerobot.teleoperators import (  # noqa: F401
    Teleoperator,
    TeleoperatorConfig,
    bi_openarm_leader,
    bi_so_leader,
    gamepad,
    homunculus,
    keyboard,
    koch_leader,
    make_teleoperator_from_config,
    omx_leader,
    openarm_leader,
    reachy2_teleoperator,
    so_leader,
    unitree_g1,
)
from lerobot.teleoperators.bi_openarm_leader import BiOpenArmLeader
from lerobot.teleoperators.openarm_leader import OpenArmLeader
from lerobot.teleoperators.openarm_leader.force_observer import ForceObserver
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging, move_cursor_up
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data


def display_motor_status(
    obs: dict[str, float],
    label: str = "Leader Status",
    max_joints: int = 8,
) -> int:
    """
    Display motor status (position, velocity, torque) in a table format.
    
    Args:
        obs: Observation dictionary with keys like "joint_1.pos", "joint_1.vel", "joint_1.torque", etc.
        label: Title for the display
        max_joints: Maximum number of joints to display
    
    Returns:
        Number of lines printed (for cursor management)
    """
    print(f"\n{label}:")
    print(f"{'Joint':<12} | {'Pos [°]':>10} | {'Vel [°/s]':>10} | {'Torque [Nm]':>12}")
    print("-" * 60)
    
    lines_printed = 3  # Header lines
    
    for i in range(1, max_joints + 1):
        joint_name = f"joint_{i}"
        pos_key = f"{joint_name}.pos"
        vel_key = f"{joint_name}.vel"
        torque_key = f"{joint_name}.torque"
        
        # Try both with and without prefix for bimanual robots
        if pos_key not in obs:
            # This might be a prefixed key, skip for now
            continue
            
        pos = obs.get(pos_key, 0.0)
        vel = obs.get(vel_key, 0.0)
        torque = obs.get(torque_key, 0.0)
        
        if isinstance(pos, (int, float)) and isinstance(vel, (int, float)) and isinstance(torque, (int, float)):
            print(f"{joint_name:<12} | {pos:>10.2f} | {vel:>10.2f} | {torque:>12.3f}")
            lines_printed += 1
    
    gripper_pos = obs.get("gripper.pos")
    if gripper_pos is not None:
        gripper_vel = obs.get("gripper.vel", 0.0)
        gripper_torque = obs.get("gripper.torque", 0.0)
        print(f"{'gripper':<12} | {gripper_pos:>10.2f} | {gripper_vel:>10.2f} | {gripper_torque:>12.3f}")
        lines_printed += 1
    
    return lines_printed


def display_bimanual_motor_status(
    obs: dict[str, float],
) -> int:
    """
    Display motor status for bimanual robots (left and right arms).
    
    Args:
        obs: Observation dictionary with prefixed keys like "left_joint_1.pos", "right_joint_1.pos", etc.
    
    Returns:
        Total number of lines printed
    """
    total_lines = 0
    
    # Display left arm
    left_obs = {key.removeprefix("left_"): value for key, value in obs.items() if key.startswith("left_")}
    if left_obs:
        left_lines = display_motor_status(left_obs, label="LEFT ARM Status", max_joints=7)
        total_lines += left_lines
    
    # Display right arm
    right_obs = {key.removeprefix("right_"): value for key, value in obs.items() if key.startswith("right_")}
    if right_obs:
        right_lines = display_motor_status(right_obs, label="RIGHT ARM Status", max_joints=7)
        total_lines += right_lines
    
    return total_lines


@dataclass
class TeleoperateConfig:
    # TODO: pepijn, steven: if more robots require multiple teleoperators (like lekiwi) its good to make this possibele in teleop.py and record.py with List[Teleoperator]
    teleop: TeleoperatorConfig
    robot: RobotConfig
    # Limit the maximum frames per second.
    fps: int = 60
    teleop_time_s: float | None = None
    # Display all cameras on screen
    display_data: bool = False
    # Display data on a remote Rerun server
    display_ip: str | None = None
    # Port of the remote Rerun server
    display_port: int | None = None
    # Whether to  display compressed images in Rerun
    display_compressed_images: bool = False


def teleop_loop(
    teleop: Teleoperator,
    robot: Robot,
    fps: int,
    teleop_action_processor: RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction],
    robot_action_processor: RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction],
    robot_observation_processor: RobotProcessorPipeline[RobotObservation, RobotObservation],
    display_data: bool = False,
    duration: float | None = None,
    display_compressed_images: bool = False,
):
    """
    This function continuously reads actions from a teleoperation device, processes them through optional
    pipelines, sends them to a robot, and optionally displays the robot's state. The loop runs at a
    specified frequency until a set duration is reached or it is manually interrupted.

    Args:
        teleop: The teleoperator device instance providing control actions.
        robot: The robot instance being controlled.
        fps: The target frequency for the control loop in frames per second.
        display_data: If True, fetches robot observations and displays them in the console and Rerun.
        display_compressed_images: If True, compresses images before sending them to Rerun for display.
        duration: The maximum duration of the teleoperation loop in seconds. If None, the loop runs indefinitely.
        teleop_action_processor: An optional pipeline to process raw actions from the teleoperator.
        robot_action_processor: An optional pipeline to process actions before they are sent to the robot.
        robot_observation_processor: An optional pipeline to process raw observations from the robot.
    """

    display_len = max(len(key) for key in robot.action_features)
    force_observer = None
    last_feedback_time = None
    force_observers: dict[str, ForceObserver] = {}
    last_feedback_times: dict[str, float | None] = {}

    if (
        isinstance(teleop, OpenArmLeader)
        and isinstance(robot, OpenArmFollower)
        and getattr(teleop.config, "force_feedback_enabled", False)
    ):
        try:
            force_observer = ForceObserver(
                urdf_path=teleop.config.urdf_path,
                gravity_vector=teleop.config.gravity_vector,
                gravity_gain=teleop.config.gravity_compensation_gain,
                lpf_cutoff_hz=teleop.config.force_feedback_lpf_cutoff_hz,
                torque_limits=teleop.config.force_feedback_torque_limits,
            )
        except Exception as exc:
            logging.warning(f"Force feedback disabled: failed to init observer ({exc})")
            force_observer = None
    elif (
        isinstance(teleop, BiOpenArmLeader)
        and isinstance(robot, BiOpenArmFollower)
        and getattr(teleop.config, "force_feedback_enabled", False)
    ):
        for side in ("left", "right"):
            try:
                force_observers[side] = ForceObserver(
                    urdf_path=teleop.config.urdf_path,
                    gravity_vector=teleop.config.gravity_vector,
                    gravity_gain=teleop.config.gravity_compensation_gain,
                    lpf_cutoff_hz=teleop.config.force_feedback_lpf_cutoff_hz,
                    torque_limits=teleop.config.force_feedback_torque_limits,
                )
                last_feedback_times[side] = None
            except Exception as exc:
                logging.warning(
                    "Force feedback disabled: failed to init observer for "
                    f"{side} arm ({exc})"
                )
    start = time.perf_counter()

    while True:
        loop_start = time.perf_counter()

        # Get robot observation
        # Not really needed for now other than for visualization
        # teleop_action_processor can take None as an observation
        # given that it is the identity processor as default
        obs = robot.get_observation()

        # Get teleop action
        raw_action = teleop.get_action()

        # Process teleop action through pipeline
        teleop_action = teleop_action_processor((raw_action, obs))

        # Process action for robot through pipeline
        robot_action_to_send = robot_action_processor((teleop_action, obs))

        # Send processed action to robot (robot_action_processor.to_output should return RobotAction)
        _ = robot.send_action(robot_action_to_send)

        if force_observer is not None:
            now = time.perf_counter()
            dt_s = None if last_feedback_time is None else now - last_feedback_time
            last_feedback_time = now

            tau_ext = force_observer.estimate(obs, dt_s=dt_s)
            feedback = {f"joint_{i}.tau_ext": float(tau_ext[i - 1]) for i in range(1, 8)}
            teleop.send_feedback(feedback)
        elif force_observers:
            now = time.perf_counter()
            feedback: dict[str, float] = {}
            for side, observer in force_observers.items():
                side_obs = {
                    key.removeprefix(f"{side}_"): value
                    for key, value in obs.items()
                    if key.startswith(f"{side}_")
                }
                dt_s = None
                if side in last_feedback_times and last_feedback_times[side] is not None:
                    dt_s = now - last_feedback_times[side]
                last_feedback_times[side] = now

                tau_ext = observer.estimate(side_obs, dt_s=dt_s)
                feedback.update(
                    {
                        f"{side}_joint_{i}.tau_ext": float(tau_ext[i - 1])
                        for i in range(1, 8)
                    }
                )
            if feedback:
                teleop.send_feedback(feedback)

        if display_data:
            # Process robot observation through pipeline
            obs_transition = robot_observation_processor(obs)

            log_rerun_data(
                observation=obs_transition,
                action=teleop_action,
                compress_images=display_compressed_images,
            )

            # Display motor status from observation
            # Check if this is a bimanual robot
            is_bimanual = any(key.startswith("left_") and key.endswith(".pos") for key in obs.keys())
            
            if is_bimanual:
                lines_printed = display_bimanual_motor_status(obs)
            else:
                lines_printed = display_motor_status(obs, label="Motor Status")
            
            # Display action commands
            print(f"\n{'Action Commands Sent':<40}")
            print(f"{'Name':<{display_len}} | {'Value':>10}")
            print("-" * (display_len + 15))
            action_lines = 0
            for motor, value in robot_action_to_send.items():
                print(f"{motor:<{display_len}} | {value:>10.2f}")
                action_lines += 1
            
            # Calculate total lines for cursor movement
            total_lines = lines_printed + action_lines + 4  # +4 for header lines
        else:
            total_lines = 0

        dt_s = time.perf_counter() - loop_start
        precise_sleep(max(1 / fps - dt_s, 0.0))
        loop_s = time.perf_counter() - loop_start
        
        # Display loop time and performance metrics
        if display_data:
            print(f"\n{'Teleop loop time':<40} | {loop_s * 1e3:>10.2f} ms ({1 / loop_s:>6.0f} Hz)")
            move_cursor_up(total_lines + 2)

        if duration is not None and time.perf_counter() - start >= duration:
            return


@parser.wrap()
def teleoperate(cfg: TeleoperateConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))
    if cfg.display_data:
        init_rerun(session_name="teleoperation", ip=cfg.display_ip, port=cfg.display_port)
    display_compressed_images = (
        True
        if (cfg.display_data and cfg.display_ip is not None and cfg.display_port is not None)
        else cfg.display_compressed_images
    )

    teleop = make_teleoperator_from_config(cfg.teleop)
    robot = make_robot_from_config(cfg.robot)
    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    teleop.connect()
    robot.connect()

    try:
        teleop_loop(
            teleop=teleop,
            robot=robot,
            fps=cfg.fps,
            display_data=cfg.display_data,
            duration=cfg.teleop_time_s,
            teleop_action_processor=teleop_action_processor,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
            display_compressed_images=display_compressed_images,
        )
    except KeyboardInterrupt:
        pass
    finally:
        if cfg.display_data:
            rr.rerun_shutdown()
        teleop.disconnect()
        robot.disconnect()


def main():
    register_third_party_plugins()
    teleoperate()


if __name__ == "__main__":
    main()
