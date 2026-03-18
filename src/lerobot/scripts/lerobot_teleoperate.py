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
import csv
from collections import deque
import time
from pathlib import Path
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
from lerobot.teleoperators.openarm_leader.force_observer import create_force_observer
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
    force_observers: dict[str, object] = {}
    last_feedback_times: dict[str, float | None] = {}
    ff_cycle_count = 0
    ff_metrics_window = int(getattr(teleop.config, "force_feedback_metrics_window", 200))
    ff_metrics_log_interval = int(getattr(teleop.config, "force_feedback_metrics_log_interval", 0))
    ff_metrics_csv_enabled = bool(getattr(teleop.config, "force_feedback_metrics_csv_enabled", False))
    ff_metrics_csv_path = str(getattr(teleop.config, "force_feedback_metrics_csv_path", "")).strip()
    ff_metrics_csv_flush_interval = int(
        getattr(teleop.config, "force_feedback_metrics_csv_flush_interval", 100)
    )
    ff_metrics_csv_rows_pending = 0
    ff_metrics_csv_file = None
    ff_metrics_csv_writer = None
    ff_pending_csv_rows: list[dict[str, str | int]] = []

    if ff_metrics_csv_enabled:
        if not ff_metrics_csv_path:
            ff_metrics_csv_path = "logs/force_feedback_metrics.csv"
        csv_path = Path(ff_metrics_csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        ff_metrics_csv_file = csv_path.open("a", newline="", encoding="utf-8")
        ff_metrics_csv_writer = csv.DictWriter(
            ff_metrics_csv_file,
            fieldnames=[
                "wall_time_s",
                "cycle",
                "arm",
                "observer_type",
                "confidence",
                "applied_scale",
                "diverged",
                "saturated_joint_count",
                "residual_rms",
                "feedback_dt_s",
                "loop_s",
            ],
        )
        if csv_path.stat().st_size == 0:
            ff_metrics_csv_writer.writeheader()
        logging.info("Force feedback CSV logging enabled: %s", csv_path)
    ff_metrics: dict[str, deque[float] | deque[int]] = {
        "confidence": deque(maxlen=max(1, ff_metrics_window)),
        "scale": deque(maxlen=max(1, ff_metrics_window)),
        "diverged": deque(maxlen=max(1, ff_metrics_window)),
        "saturated_joint_count": deque(maxlen=max(1, ff_metrics_window)),
    }

    def _record_ff_metrics(diagnostics: dict[str, float | bool | int | str], scale: float) -> None:
        ff_metrics["confidence"].append(float(diagnostics.get("confidence", 1.0)))
        ff_metrics["scale"].append(float(scale))
        ff_metrics["diverged"].append(1 if bool(diagnostics.get("diverged", False)) else 0)
        ff_metrics["saturated_joint_count"].append(int(diagnostics.get("saturated_joint_count", 0)))

    def _queue_ff_csv_row(
        cycle_id: int,
        arm: str,
        diagnostics: dict[str, float | bool | int | str],
        scale: float,
        feedback_dt_s: float | None,
    ) -> None:
        if ff_metrics_csv_writer is None:
            return

        ff_pending_csv_rows.append(
            {
                "wall_time_s": f"{time.time():.6f}",
                "cycle": cycle_id,
                "arm": arm,
                "observer_type": str(diagnostics.get("observer_type", "unknown")),
                "confidence": f"{float(diagnostics.get('confidence', 1.0)):.6f}",
                "applied_scale": f"{float(scale):.6f}",
                "diverged": int(bool(diagnostics.get("diverged", False))),
                "saturated_joint_count": int(diagnostics.get("saturated_joint_count", 0)),
                "residual_rms": f"{float(diagnostics.get('residual_rms', 0.0)):.6f}",
                "feedback_dt_s": "" if feedback_dt_s is None else f"{feedback_dt_s:.6f}",
                "loop_s": "",
            }
        )

    def _flush_ff_csv_rows(loop_s: float) -> None:
        nonlocal ff_metrics_csv_rows_pending
        if ff_metrics_csv_writer is None:
            return
        if not ff_pending_csv_rows:
            return

        loop_s_str = f"{loop_s:.6f}"
        for row in ff_pending_csv_rows:
            row["loop_s"] = loop_s_str
            ff_metrics_csv_writer.writerow(row)

        ff_metrics_csv_rows_pending += len(ff_pending_csv_rows)
        ff_pending_csv_rows.clear()

        if ff_metrics_csv_file is not None and ff_metrics_csv_rows_pending >= max(1, ff_metrics_csv_flush_interval):
            ff_metrics_csv_file.flush()
            ff_metrics_csv_rows_pending = 0

    def _log_ff_metrics() -> None:
        if ff_metrics_log_interval <= 0:
            return
        if ff_cycle_count == 0 or ff_cycle_count % ff_metrics_log_interval != 0:
            return
        if not ff_metrics["confidence"]:
            return

        confidence_avg = sum(ff_metrics["confidence"]) / len(ff_metrics["confidence"])
        scale_avg = sum(ff_metrics["scale"]) / len(ff_metrics["scale"])
        diverged_rate = sum(ff_metrics["diverged"]) / len(ff_metrics["diverged"])
        saturated_avg = sum(ff_metrics["saturated_joint_count"]) / len(ff_metrics["saturated_joint_count"])

        logging.info(
            "Force feedback metrics | window=%d cycles=%d conf_avg=%.3f scale_avg=%.3f "
            "diverged_rate=%.3f sat_joints_avg=%.2f",
            len(ff_metrics["confidence"]),
            ff_cycle_count,
            confidence_avg,
            scale_avg,
            diverged_rate,
            saturated_avg,
        )

    if (
        isinstance(teleop, OpenArmLeader)
        and isinstance(robot, OpenArmFollower)
        and getattr(teleop.config, "force_feedback_enabled", False)
    ):
        try:
            force_observer = create_force_observer(
                observer_type=getattr(teleop.config, "force_feedback_observer_type", "simple"),
                urdf_path=teleop.config.urdf_path,
                gravity_vector=teleop.config.gravity_vector,
                gravity_gain=teleop.config.gravity_compensation_gain,
                lpf_cutoff_hz=teleop.config.force_feedback_lpf_cutoff_hz,
                torque_limits=teleop.config.force_feedback_torque_limits,
                dob_lpf_cutoff_hz=getattr(teleop.config, "force_feedback_dob_lpf_cutoff_hz", 20.0),
                friction_viscous=getattr(teleop.config, "force_feedback_friction_viscous", [0.0] * 7),
                friction_coulomb=getattr(teleop.config, "force_feedback_friction_coulomb", [0.0] * 7),
                velocity_lpf_cutoff_hz=getattr(
                    teleop.config,
                    "force_feedback_velocity_lpf_cutoff_hz",
                    30.0,
                ),
                divergence_threshold_nm=getattr(
                    teleop.config,
                    "force_feedback_divergence_threshold_nm",
                    3.0,
                ),
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
                force_observers[side] = create_force_observer(
                    observer_type=getattr(teleop.config, "force_feedback_observer_type", "simple"),
                    urdf_path=teleop.config.urdf_path,
                    gravity_vector=teleop.config.gravity_vector,
                    gravity_gain=teleop.config.gravity_compensation_gain,
                    lpf_cutoff_hz=teleop.config.force_feedback_lpf_cutoff_hz,
                    torque_limits=teleop.config.force_feedback_torque_limits,
                    dob_lpf_cutoff_hz=getattr(
                        teleop.config,
                        "force_feedback_dob_lpf_cutoff_hz",
                        20.0,
                    ),
                    friction_viscous=getattr(
                        teleop.config,
                        "force_feedback_friction_viscous",
                        [0.0] * 7,
                    ),
                    friction_coulomb=getattr(
                        teleop.config,
                        "force_feedback_friction_coulomb",
                        [0.0] * 7,
                    ),
                    velocity_lpf_cutoff_hz=getattr(
                        teleop.config,
                        "force_feedback_velocity_lpf_cutoff_hz",
                        30.0,
                    ),
                    divergence_threshold_nm=getattr(
                        teleop.config,
                        "force_feedback_divergence_threshold_nm",
                        3.0,
                    ),
                )
                last_feedback_times[side] = None
            except Exception as exc:
                logging.warning(
                    "Force feedback disabled: failed to init observer for "
                    f"{side} arm ({exc})"
                )
    start = time.perf_counter()

    try:
        while True:
            loop_start = time.perf_counter()
            ff_pending_csv_rows.clear()

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
                ff_cycle_count += 1
                cycle_id = ff_cycle_count

                tau_ext, diagnostics = force_observer.estimate_with_diagnostics(obs, dt_s=dt_s)
                applied_scale = 1.0
                if getattr(teleop.config, "force_feedback_health_monitoring_enabled", True):
                    confidence_floor = getattr(teleop.config, "force_feedback_confidence_floor", 0.5)
                    confidence = float(diagnostics.get("confidence", 1.0))
                    confidence_scale = max(0.0, min(1.0, confidence / max(confidence_floor, 1e-6)))
                    applied_scale = confidence_scale
                    tau_ext = tau_ext * confidence_scale
                    if bool(diagnostics.get("diverged", False)):
                        logging.warning("Force feedback observer diverged. Zeroing feedback torque for safety.")
                        tau_ext = tau_ext * 0.0
                        applied_scale = 0.0

                _record_ff_metrics(diagnostics, applied_scale)
                _queue_ff_csv_row(cycle_id, "single", diagnostics, applied_scale, dt_s)
                _log_ff_metrics()

                feedback = {f"joint_{i}.tau_ext": float(tau_ext[i - 1]) for i in range(1, 8)}
                # Keep gripper haptics independent from arm observer health scaling.
                feedback["gripper.tau_ext"] = float(obs.get("gripper.torque", 0.0))
                feedback["gripper.vel"] = float(obs.get("gripper.vel", 0.0))
                teleop.send_feedback(feedback)
            elif force_observers:
                now = time.perf_counter()
                feedback: dict[str, float] = {}
                ff_cycle_count += 1
                cycle_id = ff_cycle_count
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

                    tau_ext, diagnostics = observer.estimate_with_diagnostics(side_obs, dt_s=dt_s)
                    applied_scale = 1.0
                    if getattr(teleop.config, "force_feedback_health_monitoring_enabled", True):
                        confidence_floor = getattr(teleop.config, "force_feedback_confidence_floor", 0.5)
                        confidence = float(diagnostics.get("confidence", 1.0))
                        confidence_scale = max(0.0, min(1.0, confidence / max(confidence_floor, 1e-6)))
                        applied_scale = confidence_scale
                        tau_ext = tau_ext * confidence_scale
                        if bool(diagnostics.get("diverged", False)):
                            logging.warning(
                                f"Force feedback observer diverged on {side} arm. Zeroing feedback torque for safety."
                            )
                            tau_ext = tau_ext * 0.0
                            applied_scale = 0.0

                    _record_ff_metrics(diagnostics, applied_scale)
                    _queue_ff_csv_row(cycle_id, side, diagnostics, applied_scale, dt_s)

                    feedback.update(
                        {
                            f"{side}_joint_{i}.tau_ext": float(tau_ext[i - 1])
                            for i in range(1, 8)
                        }
                    )
                    # Keep gripper haptics independent from arm observer health scaling.
                    feedback[f"{side}_gripper.tau_ext"] = float(side_obs.get("gripper.torque", 0.0))
                    feedback[f"{side}_gripper.vel"] = float(side_obs.get("gripper.vel", 0.0))
                if feedback:
                    _log_ff_metrics()
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
            _flush_ff_csv_rows(loop_s)

            # Display loop time and performance metrics
            if display_data:
                print(f"\n{'Teleop loop time':<40} | {loop_s * 1e3:>10.2f} ms ({1 / loop_s:>6.0f} Hz)")
                move_cursor_up(total_lines + 2)

            if duration is not None and time.perf_counter() - start >= duration:
                return
    finally:
        if ff_metrics_csv_file is not None:
            ff_metrics_csv_file.flush()
            ff_metrics_csv_file.close()


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
