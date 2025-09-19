from __future__ import annotations

import json
import math
import os
import signal
import subprocess
import threading
import time
import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import psutil
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Point, Pose, PoseStamped, PoseWithCovarianceStamped, Quaternion
from nav2_msgs.action import FollowWaypoints, NavigateThroughPoses, NavigateToPose, Spin
from sensor_msgs.msg import Image
from rclpy.action import ActionClient
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.node import Node
from ros_robot_controller_msgs.msg import BusServoState, GetBusServoCmd
from ros_robot_controller_msgs.srv import GetBusServoState
from servo_controller.action_group_controller import ActionGroupController
from servo_controller.bus_servo_control import set_servo_position
from servo_controller_msgs.msg import ServosPosition
from std_msgs.msg import Empty, Float32MultiArray, String

from poc_pkg.forward_kinematics import ForwardKinematics
from poc_pkg.inverse_kinematics import get_ik
import poc_pkg.transform as transform


PACKAGE_DIR = Path(__file__).resolve().parent
TEMPLATE_PATH = PACKAGE_DIR / "template.png"
CALIBRATION_PATH = PACKAGE_DIR / "camera_calibration_data.json"
ACTION_GROUP_DIR = Path(os.environ.get("POC_ACTION_GROUP_DIR", "/home/ubuntu/software/arm_pc/ActionGroups"))

X_POSE_OFFSET = 0.0
Y_POSE_OFFSET = 0.1
SERVO_IDS = (1, 2, 3, 4, 5, 10)
SERVO_LIMIT = (0, 1000)
LAUNCH_CMD = ["ros2", "launch", "controller", "controller.launch.py"]
HOME_POSE = (X_POSE_OFFSET, Y_POSE_OFFSET, 0.0)
CHEST_TOLERANCE_XY = 0.05
CHEST_TOLERANCE_YAW = 0.2
ARM_INIT_POSE = np.array([0.2, 0.0, 0.3], dtype=float)
DEFAULT_GRIPPER = 100
DEFAULT_ROLL = 500

def stamped_pose(
    node: Node, x: float, y: float, yaw: float, frame_id: str = "map"
) -> PoseStamped:
    ps = PoseStamped()
    ps.header.stamp = node.get_clock().now().to_msg()
    ps.header.frame_id = frame_id
    ps.pose = Pose(position=Point(x=x, y=y, z=0.0), orientation=yaw_to_quat(yaw))
    return ps


def shifted_pose(dx: float = 0.0, dy: float = 0.0, yaw: float = 0.0) -> Tuple[float, float, float]:
    return X_POSE_OFFSET + dx, Y_POSE_OFFSET + dy, yaw


def yaw_to_quat(yaw: float) -> Quaternion:
    """z-yaw -> quaternion (x=y=0, roll=pitch=0)."""
    q = Quaternion()
    q.z = math.sin(yaw / 2.0)
    q.w = math.cos(yaw / 2.0)
    return q

def rpy2qua(roll, pitch, yaw):
    cy = math.cos(yaw*0.5)
    sy = math.sin(yaw*0.5)
    cp = math.cos(pitch*0.5)
    sp = math.sin(pitch*0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    
    q = Pose()
    q.orientation.w = cy * cp * cr + sy * sp * sr
    q.orientation.x = cy * cp * sr - sy * sp * cr
    q.orientation.y = sy * cp * sr + cy * sp * cr
    q.orientation.z = sy * cp * cr - cy * sp * sr
    return q.orientation

def qua2rpy_solid(qua):
    """Преобразование кватерниона в углы Эйлера (roll, pitch, yaw)"""
    x, y, z, w = qua.x, qua.y, qua.z, qua.w
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(2 * (w * y - x * z))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (z * z + y * y))
  
    return roll, pitch, yaw

def qua2rpy(x, y, z, w):
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(2 * (w * y - x * z))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (z * z + y * y))
  
    return roll, pitch, yaw

@dataclass
class MatchResult:
    center: Tuple[int, int]
    quad: np.ndarray


@dataclass
class RobotPose:
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0

    def is_close(self, x: float, y: float, yaw: float, tol_xy: float, tol_yaw: float) -> bool:
        return (
            abs(self.x - x) <= tol_xy
            and abs(self.y - y) <= tol_xy
            and abs(self.yaw - yaw) <= tol_yaw
        )


@dataclass(frozen=True)
class ActionStep:
    command: str
    pose: Optional[Sequence[float]] = None
    delta_yaw: Optional[float] = None
    wait: float = 0.0


class ImageProcessor:
    CLAHE_CLIP_LIMIT = 3.0
    CLAHE_TILE_GRID = (8, 8)
    MATCH_RATIO = 0.6
    MIN_MATCHES = 9
    CAMERA_Z = 0.15
    TARGET_OFFSET = (-0.01, 0.01)
    WINDOW_NAME = "Undistorted Image"

    def __init__(self, node: Node, camera: str = "usb_cam"):
        self.node = node
        self.camera = camera
        self.bridge = CvBridge()
        self.callback_group = MutuallyExclusiveCallbackGroup()
        self.subscription = None

        self._descriptor_norm = cv2.NORM_L2
        self.clahe = cv2.createCLAHE(
            clipLimit=self.CLAHE_CLIP_LIMIT, tileGridSize=self.CLAHE_TILE_GRID
        )

        try:
            self.camera_matrix, self.dist_coeff = self._load_calibration(CALIBRATION_PATH)
            self.detector = self._create_detector()
            self.matcher = self._create_matcher()
            self.template_image = self._load_template(TEMPLATE_PATH)
            self.template_keypoints, self.template_descriptors = self._compute_features(
                self.template_image
            )
        except FileNotFoundError as exc:
            self.node.get_logger().error(str(exc))
            raise

        if self.template_descriptors is None or not self.template_keypoints:
            raise RuntimeError("Failed to compute features for the template image.")

        self.undistorted_image: Optional[np.ndarray] = None
        self.x_err: Optional[float] = None
        self.y_err: Optional[float] = None

    def start_image_processing(self) -> None:
        if self.subscription is not None:
            return

        topic = f"/{self.camera}/image_raw"
        self.subscription = self.node.create_subscription(
            Image,
            topic,
            self.process_image,
            1,
            callback_group=self.callback_group,
        )

    def process_image(self, msg: Image) -> None:
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as exc:  # pragma: no cover - defensive logging
            self.node.get_logger().error(f"Failed to convert image message: {exc}")
            return

        self.undistorted_image = cv2.undistort(cv_image, self.camera_matrix, self.dist_coeff)

        match = self._match_template(self.undistorted_image)
        if match is None:
            self.x_err = None
            self.y_err = None
            return

        self.x_err, self.y_err = self._compute_offsets(match.center)
        self._draw_debug_overlay(match)

    def pixel_to_cam_XY(self, u: float, v: float, K_used: np.ndarray, Zc: float) -> Tuple[float, float]:
        fx, fy = K_used[0, 0], K_used[1, 1]
        cx, cy = K_used[0, 2], K_used[1, 2]
        x_n = (u - cx) / fx
        y_n = (v - cy) / fy
        X = x_n * Zc
        Y = y_n * Zc
        return float(X), float(Y)

    def _load_template(self, path: Path) -> np.ndarray:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Template image not found at '{path}'")
        return image

    def _load_calibration(self, path: Path) -> Tuple[np.ndarray, np.ndarray]:
        if not path.exists():
            raise FileNotFoundError(f"Calibration file not found at '{path}'")
        with open(path, "r", encoding="utf-8") as f:
            calibration_data = json.load(f)
        camera_matrix = np.array(calibration_data["camera_matrix"], dtype=np.float32)
        dist_coeff = np.array(calibration_data["dist_coeff"], dtype=np.float32)
        return camera_matrix, dist_coeff

    def _create_detector(self):
        if hasattr(cv2, "SIFT_create"):
            return cv2.SIFT_create(
                nfeatures=3000,
                contrastThreshold=0.05,
                edgeThreshold=10,
                sigma=1.6,
            )

        self.node.get_logger().warn("SIFT is not available, falling back to AKAZE")
        self._descriptor_norm = cv2.NORM_HAMMING
        return cv2.AKAZE_create()

    def _create_matcher(self):
        if self._descriptor_norm == cv2.NORM_L2:
            index_params = dict(algorithm=1, trees=5)
            search_params = dict(checks=64)
            return cv2.FlannBasedMatcher(index_params, search_params)
        return cv2.BFMatcher(self._descriptor_norm)

    def _compute_features(self, gray_image: np.ndarray):
        equalized = self.clahe.apply(gray_image)
        keypoints, descriptors = self.detector.detectAndCompute(equalized, None)
        if descriptors is None:
            return keypoints, None
        if self._descriptor_norm == cv2.NORM_L2 and descriptors.dtype != np.float32:
            descriptors = descriptors.astype(np.float32)
        return keypoints, descriptors

    def _match_template(self, frame_bgr: np.ndarray) -> Optional[MatchResult]:
        gray_scene = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        kp_scene, des_scene = self._compute_features(gray_scene)
        if des_scene is None or not kp_scene:
            self.node.get_logger().debug("Недостаточно дескрипторов – пропускаю кадр")
            return None

        try:
            matches_12 = self.matcher.knnMatch(self.template_descriptors, des_scene, k=2)
            matches_21 = self.matcher.knnMatch(des_scene, self.template_descriptors, k=2)
        except cv2.error as exc:
            self.node.get_logger().debug(f"Matcher failed: {exc}")
            return None

        good_matches = self._filter_matches(matches_12, matches_21)
        self.node.get_logger().debug(
            "KPs: tpl=%d scene=%d; good=%d",
            len(self.template_keypoints),
            len(kp_scene),
            len(good_matches),
        )

        if len(good_matches) < self.MIN_MATCHES:
            self.node.get_logger().debug("Мало хороших совпадений")
            return None

        src_pts = np.float32(
            [self.template_keypoints[m.queryIdx].pt for m in good_matches]
        ).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_scene[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        matrix, _ = cv2.findHomography(
            src_pts,
            dst_pts,
            cv2.RANSAC,
            ransacReprojThreshold=3.0,
            maxIters=5000,
            confidence=0.999,
        )
        if matrix is None:
            self.node.get_logger().debug("Гомография не найдена")
            return None

        h, w = self.template_image.shape
        template_corners = np.float32(
            [[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]
        ).reshape(-1, 1, 2)
        projected = cv2.perspectiveTransform(template_corners, matrix)
        quad = projected.reshape(4, 2)
        center = tuple(np.mean(quad, axis=0).astype(int))
        return MatchResult(center=center, quad=quad)

    def _filter_matches(self, matches_12, matches_21):
        ratio_matches = []
        for pair in matches_12:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < self.MATCH_RATIO * n.distance:
                ratio_matches.append(m)

        mutual = {}
        for pair in matches_21:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < self.MATCH_RATIO * n.distance:
                mutual[m.queryIdx] = m.trainIdx

        return [m for m in ratio_matches if mutual.get(m.trainIdx) == m.queryIdx]

    def _compute_offsets(self, center: Tuple[int, int]) -> Tuple[float, float]:
        target_x, target_y = self.TARGET_OFFSET
        cam_x, cam_y = self.pixel_to_cam_XY(center[0], center[1], self.camera_matrix, self.CAMERA_Z)
        return target_x - cam_x, target_y - cam_y

    def _draw_debug_overlay(self, match: MatchResult) -> None:
        if self.undistorted_image is None:
            return

        debug_image = self.undistorted_image.copy()
        cv2.polylines(debug_image, [np.int32(match.quad.reshape(-1, 1, 2))], True, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.circle(debug_image, match.center, 5, (0, 0, 255), -1)
        cv2.imshow(self.WINDOW_NAME, debug_image)
        cv2.waitKey(1)


class Mover:
    def __init__(self, node: Node):
        self.node = node
        self.pose = RobotPose()

        self.pose_subscription = self.node.create_subscription(
            PoseWithCovarianceStamped,
            "/amcl_pose",
            self.update_pose,
            1,
            callback_group=ReentrantCallbackGroup(),
        )

        self.pose_pub = self.node.create_publisher(PoseStamped, "/goal_pose", 1)
        self.ac_nav_to_pose = ActionClient(
            self.node,
            NavigateToPose,
            "/navigate_to_pose",
            callback_group=MutuallyExclusiveCallbackGroup(),
        )
        self.ac_nav_through = ActionClient(
            self.node,
            NavigateThroughPoses,
            "/navigate_through_poses",
            callback_group=MutuallyExclusiveCallbackGroup(),
        )
        self.ac_spin = ActionClient(
            self.node,
            Spin,
            "/spin",
            callback_group=MutuallyExclusiveCallbackGroup(),
        )
        self.ac_follow_waypoints = ActionClient(
            self.node,
            FollowWaypoints,
            "/follow_waypoints",
            callback_group=MutuallyExclusiveCallbackGroup(),
        )

        self.action_sequences = self._build_default_actions()

    def spin(self, delta_yaw: float, wait_seconds: float) -> None:
        goal = Spin.Goal()
        goal.target_yaw = float(delta_yaw)
        self.ac_spin.send_goal_async(goal)
        if wait_seconds > 0:
            time.sleep(wait_seconds)

    def navigate_to_pose(
        self,
        x: float,
        y: float,
        yaw: float,
        frame_id: str = "map",
        behavior_tree: str = "",
        timeout_sec: Optional[float] = None,
    ) -> None:
        goal = NavigateToPose.Goal()
        goal.pose = stamped_pose(self.node, x, y, yaw, frame_id)
        goal.behavior_tree = behavior_tree
        self.ac_nav_to_pose.send_goal_async(goal)
        if timeout_sec:
            time.sleep(timeout_sec)

    def navigate_through_poses(
        self,
        poses_xyyaw: Iterable[Tuple[float, float, float]],
        frame_id: str = "map",
        behavior_tree: str = "",
        timeout_sec: Optional[float] = None,
    ) -> None:
        goal = NavigateThroughPoses.Goal()
        goal.poses = [stamped_pose(self.node, x, y, yaw, frame_id) for x, y, yaw in poses_xyyaw]
        goal.behavior_tree = behavior_tree
        self.ac_nav_through.send_goal_async(goal)
        if timeout_sec:
            time.sleep(timeout_sec)

    def follow_waypoints(
        self,
        poses_xyyaw: Iterable[Tuple[float, float, float]],
        frame_id: str = "map",
        timeout_sec: Optional[float] = None,
    ) -> None:
        goal = FollowWaypoints.Goal()
        goal.poses = [stamped_pose(self.node, x, y, yaw, frame_id) for x, y, yaw in poses_xyyaw]
        self.ac_follow_waypoints.send_goal_async(goal)
        if timeout_sec:
            time.sleep(timeout_sec)

    def update_pose(self, msg: PoseWithCovarianceStamped) -> None:
        pose = msg.pose.pose
        roll, pitch, yaw = qua2rpy_solid(pose.orientation)
        self.pose = RobotPose(pose.position.x, pose.position.y, yaw)
        self.node.get_logger().debug(
            "Updated pose: x=%.3f, y=%.3f, yaw=%.3f",
            self.pose.x,
            self.pose.y,
            self.pose.yaw,
        )

    def run_action(self, action: str) -> None:
        sequence = self.action_sequences.get(action)
        if not sequence:
            self.node.get_logger().error(f"Unknown action: {action}")
            return

        self.node.get_logger().info(f"Running action: {action}")
        for step in sequence:
            if step.command == "move" and step.pose is not None:
                self.move_to_pose(step.pose, step.wait)
            elif step.command == "spin" and step.delta_yaw is not None:
                self.spin(step.delta_yaw, step.wait)
            else:
                self.node.get_logger().warn(f"Unsupported action step: {step}")

    def move_to_pose(self, target_pose: Sequence[float], wait_seconds: float) -> None:
        if len(target_pose) < 3:
            raise ValueError("target_pose must contain x, y, yaw")

        x, y, yaw = target_pose[:3]
        target_pose_msg = PoseStamped()
        target_pose_msg.header.frame_id = "map"
        target_pose_msg.pose.position.x = float(x)
        target_pose_msg.pose.position.y = float(y)
        target_pose_msg.pose.position.z = 0.0
        target_pose_msg.pose.orientation = rpy2qua(0.0, 0.0, float(yaw))
        self.pose_pub.publish(target_pose_msg)
        if wait_seconds > 0:
            time.sleep(wait_seconds)

    def _build_default_actions(self) -> Dict[str, List[ActionStep]]:
        actions: Dict[str, List[ActionStep]] = {
            "PoC1": [
                ActionStep("move", pose=HOME_POSE, wait=2.7),
                ActionStep("spin", delta_yaw=-0.3, wait=2.7),
                ActionStep("spin", delta_yaw=0.6, wait=2.7),
                ActionStep("spin", delta_yaw=-0.6, wait=2.7),
                ActionStep("spin", delta_yaw=0.6, wait=2.7),
                ActionStep("spin", delta_yaw=-0.3, wait=2.7),
                ActionStep("move", pose=HOME_POSE, wait=0.0),
            ]
        }

        swing_poses = [shifted_pose(0.07, 0.0, 0.2), shifted_pose(-0.07, 0.0, -0.2)]
        actions["PoC_idle_1"] = self._build_alternating_sequence(swing_poses, repeats=3)

        sway_poses = [shifted_pose(0.0, 0.1, 0.0), shifted_pose(0.0, -0.1, 0.0)]
        actions["PoC_idle_2"] = self._build_alternating_sequence(sway_poses, repeats=4)
        return actions

    def _build_alternating_sequence(
        self,
        poses: Iterable[Tuple[float, float, float]],
        repeats: int,
    ) -> List[ActionStep]:
        sequence: List[ActionStep] = [ActionStep("move", pose=HOME_POSE, wait=2.7)]
        for _ in range(repeats):
            for pose in poses:
                sequence.append(ActionStep("move", pose=pose, wait=2.7))
        sequence.append(ActionStep("move", pose=HOME_POSE, wait=0.0))
        return sequence

class ArmNode(Node):
    def __init__(self):
        super().__init__("arm_node")
        self.get_logger().info("ArmNode has been started.")

        self.proc: Optional[subprocess.Popen] = None
        self.pool = ThreadPoolExecutor(max_workers=3)
        self.controller_future: Optional[Future] = None
        self.mover_future: Optional[Future] = None

        self.controller_works = False
        self.current_servo_positions = np.array([], dtype=float)
        self.current_pose = np.zeros(6, dtype=float)
        self.init_pose = ARM_INIT_POSE.copy()
        self.gripper = DEFAULT_GRIPPER
        self.roll = DEFAULT_ROLL
        self.chest_opened = False
        self.last_feedback_time = time.time()

        self._init_callback_groups()

        self.client = self.create_client(
            GetBusServoState, "ros_robot_controller/bus_servo/get_state"
        )
        self.client.wait_for_service()

        self._create_publishers()
        self._create_subscriptions()
        self._create_timers()

        self.image_processor = ImageProcessor(self)
        self.mover = Mover(self)
        self.fk = ForwardKinematics(debug=False)

        time.sleep(1.0)
        self.mover.move_to_pose(HOME_POSE, 0.5)

    def _init_callback_groups(self) -> None:
        self.action_cb_group = MutuallyExclusiveCallbackGroup()
        self.reset_cb_group = MutuallyExclusiveCallbackGroup()
        self.chest_cb_group = MutuallyExclusiveCallbackGroup()
        self.get_servo_cb_group = MutuallyExclusiveCallbackGroup()
        self.idle_cb_group = MutuallyExclusiveCallbackGroup()
        self.motion_cb_group = MutuallyExclusiveCallbackGroup()
        self.init_cb_group = MutuallyExclusiveCallbackGroup()

    def _create_publishers(self) -> None:
        self.joints_pub = self.create_publisher(ServosPosition, "servo_controller", 1)
        if not ACTION_GROUP_DIR.exists():
            self.get_logger().warn(
                f"Action group directory '{ACTION_GROUP_DIR}' does not exist"
            )
        self.controller = ActionGroupController(self.joints_pub, str(ACTION_GROUP_DIR))
        self.error_pub = self.create_publisher(String, "/poc/error", 1)
        self.status_pub = self.create_publisher(String, "/poc/status", 1)

    def _create_subscriptions(self) -> None:
        self.create_subscription(
            Float32MultiArray,
            "/poc/move_to_pose",
            self._handle_move_to_pose_command,
            1,
            callback_group=self.motion_cb_group,
        )
        self.create_subscription(
            String,
            "/poc/action",
            self.execute_action,
            1,
            callback_group=self.action_cb_group,
        )
        self.create_subscription(
            Empty,
            "/poc/chest",
            self.open_chest,
            1,
            callback_group=self.chest_cb_group,
        )
        self.create_subscription(
            Empty,
            "/poc/reset",
            self.reset_robot,
            1,
            callback_group=self.reset_cb_group,
        )
        self.create_subscription(
            Empty,
            "/poc/stop_idle",
            self.stop_idle,
            1,
            callback_group=self.idle_cb_group,
        )

    def _create_timers(self) -> None:
        self.servos_position_timer = self.create_timer(
            0.5,
            self.update_servos_position,
            callback_group=self.get_servo_cb_group,
        )
        self.initialize_arm_timer = self.create_timer(
            1.3,
            self.initialize_arm,
            callback_group=self.init_cb_group,
        )

    def _handle_move_to_pose_command(self, msg: Float32MultiArray) -> None:
        if len(msg.data) < 3:
            self.publish_error("Move-to-pose command requires x, y, yaw values")
            return
        target = tuple(float(v) for v in msg.data[:3])
        self.get_logger().info(f"Moving to pose: {target}")
        self.mover.move_to_pose(target, 0.0)

    def start_launch(self):
        if self.proc and self.proc.poll() is None:
            self.get_logger().warn("controller.launch.py is already running")
            return

        self.get_logger().info("Запуск controller.launch.py")
        self.proc = subprocess.Popen(
            LAUNCH_CMD,
            preexec_fn=os.setsid,          # Linux / macOS ─ новая группа
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        threading.Thread(target=self._pipe_stdout, daemon=True).start()
    

    def _on_future_done(self, who: str, action: str, fut):
        if fut.cancelled():
            self.get_logger().debug(f"{who} action '{action}' was cancelled")
        else:
            try:
                fut.result()
                self.get_logger().info(f"{who}.run_action('{action}') finished")
            except Exception as e:
                tb = traceback.format_exc()
                self.publish_error(f"{who} failed on '{action}': {e}\n{tb}")

        if who == 'controller':
            self.controller_future = None
        elif who == 'mover':
            self.mover_future = None

    def _kill_tree(self, proc: psutil.Process, sig):
        """Послать signal `sig` всем дочерним + самому `proc`."""
        for p in proc.children(recursive=True):
            try:
                p.send_signal(sig)
            except ProcessLookupError:
                pass
        try:
            proc.send_signal(sig)
        except ProcessLookupError:
            pass

    def stop_launch(self):
        if not self.proc or self.proc.poll() is not None:
            return
        self.get_logger().warn("Останавливаю controller.launch.py")

        parent = psutil.Process(self.proc.pid)

        # 1. Мягко — SIGINT
        self._kill_tree(parent, signal.SIGINT)
        gone, alive = psutil.wait_procs([parent]+parent.children(recursive=True),
                                        timeout=3)

        if alive:                       # 2. Эскалация SIGTERM
            self.get_logger().warn("‣ SIGINT не помог, посылаю SIGTERM…")
            self._kill_tree(parent, signal.SIGTERM)
            gone, alive = psutil.wait_procs(alive, timeout=3)

        if alive:                       # 3. Жёстко SIGKILL
            self.get_logger().error("‣ Всё ещё висят – SIGKILL!")
            self._kill_tree(parent, signal.SIGKILL)
            psutil.wait_procs(alive, timeout=3)

        self.proc = None

    def _pipe_stdout(self):
        if not self.proc or self.proc.stdout is None:
            return
        for line in self.proc.stdout:
            self.get_logger().debug(f"[controller] {line.strip()}")

    def initialize_arm(self):
        try:
            self.set_target_pose(self.init_pose, 0.0, self.gripper, 1.5)
            time.sleep(1.5)
            self.get_logger().info("Arm initialized")
        except Exception as exc:
            self.publish_error(f"Error in setting initial target pose: {exc}")
        finally:
            if self.initialize_arm_timer is not None:
                self.initialize_arm_timer.destroy()
                self.initialize_arm_timer = None

    def update_servos_position(self):
        if not self.client.service_is_ready():
            self.get_logger().debug("Servo state service is not ready yet")
            return

        request = GetBusServoState.Request()
        request.cmd = [GetBusServoCmd(id=i, get_position=1) for i in SERVO_IDS]
        future = self.client.call_async(request)
        future.add_done_callback(self._update_servos_position_callback)

    def _update_servos_position_callback(self, future):
        try:
            response = future.result()
        except Exception as exc:
            self.publish_error(f"Servo state request failed: {exc}")
            return

        if response is None or not getattr(response, "success", False):
            self.publish_error("Servo state request returned no data")
            return

        positions = np.zeros(len(SERVO_IDS), dtype=float)
        for index, state in enumerate(response.state):
            try:
                positions[index] = state.position[0]
            except Exception as exc:
                self.get_logger().error(f"Error parsing servo state {index}: {exc}")

        self.current_servo_positions = positions

        if not self.controller_works:
            self.status_pub.publish(String(data="ok"))
            self.controller_works = True

        self.last_feedback_time = time.time()


    def open_chest(self, msg):
        self.stop_idle(None, chest=True)
        if not self.mover.pose.is_close(0.0, 0.0, 0.0, CHEST_TOLERANCE_XY, CHEST_TOLERANCE_YAW):
            self.get_logger().error(
                "Robot is not in the correct position to open the chest"
            )
            time.sleep(1.0)
        # set_servo_position(self.joints_pub, 2.0, ((10, 0), (5, 500), (4, 92), (3, 344), (2, 319), (1, 147)))
        self.set_target_pose([0.02036, -0.2204, 0.14039], 88.8, 0, 3.0)
        self.image_processor.start_image_processing()
        time.sleep(7)
        if self.image_processor.x_err is None or self.image_processor.y_err is None:
            self.set_target_pose([0.02036, -0.2204, 0.14039], 88.8, 0, 3.0)
            time.sleep(7)

        
        c_x, c_y = 0.02036, -0.2104
        x_err, y_err = self.image_processor.x_err, self.image_processor.y_err
        object_x, object_y = c_x + x_err, c_y - y_err
        self.get_logger().info('XY error: %s, %s' % (x_err, y_err))
        self.set_target_pose([object_x, object_y, 0.14039], 88.8, 0, 0.5)
        time.sleep(2)
        self.set_target_pose([object_x, object_y, 0.065], 88.8, self.gripper, 0.5)
        time.sleep(2)
        self.set_gripper(900, 0.5)
        time.sleep(0.5)
        self.set_gripper(900, 0.5)
        time.sleep(1.5)
        self.set_target_pose([object_x, object_y, 0.18039], 88.8, self.gripper, 1.0)
        time.sleep(1.5)
        self.set_target_pose([object_x-0.07, object_y-0.08, 0.29039], 20.8, self.gripper, 1.0)


    def publish_error(self, error_message):
        self.get_logger().error(error_message)
        self.error_pub.publish(String(data=error_message))

    def stop_idle(self, _msg=None, chest: bool = False) -> None:
        self.get_logger().info("Stopping idle...")
        self.controller.stop_action_group()
        self._cancel_action_futures()
        target_pose = (0.0, 0.0, 0.0) if chest else HOME_POSE
        self.mover.move_to_pose(target_pose, 0.5)

    def _cancel_action_futures(self) -> None:
        for future in (self.controller_future, self.mover_future):
            if future and not future.done():
                future.cancel()
        self.controller_future = None
        self.mover_future = None
        
    def reset_robot(self, msg):
        self.get_logger().info('Resetting robot...')
        self.set_gripper(DEFAULT_GRIPPER, 2.0)
        self.set_roll(DEFAULT_ROLL, 1.0)
        time.sleep(1.2)
        self.set_target_pose(self.init_pose, 0, self.gripper, 2.0)
        time.sleep(2.0)
        self.update_current_pose()

    def set_gripper(self, gripper, duration=1.0):
        self.gripper = int(np.clip(gripper, SERVO_LIMIT[0], SERVO_LIMIT[1]))
        self.get_logger().info(f'Setting gripper position: {self.gripper}')
        try:
            set_servo_position(self.joints_pub, duration, ((10, self.gripper),))
        except Exception as e:
            self.publish_error(f"Error in setting gripper position: {str(e)}")

    def set_roll(self, roll, duration=1.0):
        self.roll = int(np.clip(roll, SERVO_LIMIT[0], SERVO_LIMIT[1]))
        self.get_logger().info(f'Setting roll position: {self.roll}')

        try:
            set_servo_position(self.joints_pub, duration, ((5, self.roll),))
        except Exception as e:
            self.publish_error(f"Error in setting roll position: {str(e)}")

    def execute_action(self, msg):
        action = msg.data
        self.get_logger().info(f'Executing action: {action}')
        self.stop_idle(None)

        self.controller_future = self.pool.submit(self.controller.run_action, action)
        self.mover_future = self.pool.submit(self.mover.run_action, action)

        self.controller_future.add_done_callback(
            lambda fut: self._on_future_done('controller', action, fut)
        )
        self.mover_future.add_done_callback(
            lambda fut: self._on_future_done('mover', action, fut)
        )

    def set_target_pose(self, pose, pitch, gripper, duration):
        self.get_logger().info(f'Setting target pose: {pose}')
        self.gripper = gripper
        try:
            servo_data = self._search_solution(pose, pitch, (-180, 180), 1)
        except Exception as exc:
            self.publish_error(f"Error in setting target pose: {exc}")
            return

        if not servo_data or len(servo_data) < 4:
            self.publish_error("No valid solution found for the target pose.")
            return

        servo_data = [int(np.clip(value, SERVO_LIMIT[0], SERVO_LIMIT[1])) for value in servo_data]
        command = (
            (10, self.gripper),
            (5, self.roll),
            (4, servo_data[3]),
            (3, servo_data[2]),
            (2, servo_data[1]),
            (1, servo_data[0]),
        )
        try:
            set_servo_position(self.joints_pub, duration, command)
        except Exception as exc:
            self.publish_error(f"Error in sending servo command: {exc}")

    def update_current_pose(self):
        try:
            angle = transform.pulse2angle(self.current_servo_positions)
            res = self.fk.get_fk(angle)
            rpy = qua2rpy(res[1].x, res[1].y, res[1].z, res[1].w)
            self.current_pose = np.array(
                [res[0][0], res[0][1], res[0][2], rpy[0], rpy[1], rpy[2]],
                dtype=float,
            )
            self.get_logger().debug(f'Updated current pose: {self.current_pose}')
        except Exception as e:
            self.publish_error(f"Error in updating current pose: {str(e)}")

    def _search_solution(self, position, pitch, pitch_range, resolution) -> Optional[List[float]]:
        try:
            all_solutions = get_ik(list(position), pitch, list(pitch_range), resolution)
        except Exception as exc:
            self.publish_error(f"Error in searching solution: {exc}")
            return None

        if not all_solutions or self.current_servo_positions.size == 0:
            return None

        reference = (
            self.current_servo_positions[:-1]
            if self.current_servo_positions.size > 1
            else self.current_servo_positions
        )

        best_pulses: Optional[List[float]] = None
        min_distance = float("inf")

        for joint_solution, _ in all_solutions:
            try:
                pulse_solutions = transform.angle2pulse(joint_solution, True)
            except Exception as exc:
                self.get_logger().debug(f"choose solution error: {exc}")
                continue

            for pulses in pulse_solutions:
                pulses_array = np.asarray(pulses, dtype=float)
                if reference.size and pulses_array.size >= reference.size:
                    distance = float(np.sum(np.abs(pulses_array[: reference.size] - reference)))
                else:
                    distance = float(np.sum(np.abs(pulses_array)))

                if distance < min_distance:
                    min_distance = distance
                    best_pulses = np.clip(pulses_array, SERVO_LIMIT[0], SERVO_LIMIT[1]).tolist()

        return best_pulses


def main():
    rclpy.init()
    node = ArmNode()

    # Используем многопоточный executor, но благодаря MutuallyExclusiveCallbackGroup обработка callback-ов будет последовательной.
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info("Останавливаем выполнение...")
    finally:
        # node.stop_launch()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
