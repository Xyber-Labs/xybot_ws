# Robot Workspace (`robot_ws`)

This repository is a ROS 2 Humble workspace for a mobile robot that blends perception, navigation, manipulation, and offline voice control. It is meant to be a one-stop launchpad for demos, quick experiments, and production bring-up on the JetAuto/Jetson class platforms.

## Highlights
- Modular ROS 2 packages for core robot behavior (`src/app`), hardware abstraction (`src/driver`), navigation (`src/navigation`), and SLAM (`src/slam`).
- Peripheral integration covering depth cameras, LiDAR, IMUs, servos, and teleoperation utilities (`src/peripherals`, `src/arm_teleop`).
- Offline speech wake-word and ASR pipeline via `xf_mic_asr_offline`, with secure configuration through environment variables.
- Rich example gallery showcasing perception, body tracking, gesture control, and autonomous behaviors (`src/example`).
- Firmware helpers and a command cheat sheet (`firmware/`, `command`) to streamline bring-up on target hardware.

## Workspace Layout
- `src/bringup`: System launch files and health checks that stitch the robot stack together.
- `src/app`: High-level behaviors such as lidar obstacle avoidance, patrol, line following, AR demos, and more.
- `src/driver`: ROS 2 drivers for robot base, kinematics, controller boards, and sensors.
- `src/navigation` / `src/slam`: Navigation2 pipelines, localization, and mapping launch configurations.
- `src/peripherals`: Camera, LiDAR, and IMU visualizers plus calibration utilities.
- `src/example`: Stand-alone perception & ML demos (MediaPipe, YOLOv5, QR codes, etc.).
- `src/micro_ros_setup`: Tooling for micro-ROS firmware workflows.
- `xf_mic_asr_offline` & `_msgs`: Offline voice assistant nodes, resources, and interfaces.
- `command`: Quick-reference notebook of ROS 2 launch and service invocations used in day-to-day operations.

## Quick Start
```bash
# 1. Install ROS 2 Humble and dependencies (Ubuntu 22.04 recommended).

# 2. Fetch workspace dependencies.
vcs import src < src/ros2.repos  # no-op by default, ready for future repos

# 3. Build with colcon.
colcon build --symlink-install \
  --cmake-args -DCMAKE_BUILD_TYPE=Release

# 4. Source the workspace and launch the stack.
source install/setup.bash
ros2 launch bringup bringup.launch.py
```

Refer to the `command` cheat sheet for additional launch patterns (calibration, AR apps, tracking demos, etc.).

## Configuration & Secrets
- Speech services expect `ASR_APPID` to be provided (environment variable or launch argument). The workspace avoids hardcoding credentials; populate secrets through your deployment tooling.
- `.env` files, certificates, and other sensitive assets are ignored via `.gitignore`. Keep credentials out of version control and rotate any historical keys that were ever committed.

## Simulation & Testing
- Launch files under `src/simulations` and `src/example` offer Gazebo/visualization-ready demos for rapid experimentation.
- Use RViz and the scripts in `src/peripherals` to validate sensor streams before engaging autonomy stacks.

## Development Tips
- Run `colcon test` or targeted package builds (`--packages-select`) while developing.
- Follow ROS 2 best practices: format with `ament_uncrustify/black`, keep launch files parameterized, and prefer composable nodes where possible.
- Before pushing, run a secret scan (e.g., Gitleaks) and ensure new assets respect the existing `.gitignore` rules.

Happy hacking — don't forget to share your favorite demo clip once the robot is rolling!
