#!/usr/bin/env python3
"""
aruco_scout_node.py
Passive ArUco marker detection during frontier exploration.

Subscribes to the robot camera, detects ArUco markers, converts each
detection to world_enu frame via TF, and tracks the best (most recent and
most confident) sighting.

Publishes:
  /aruco/best_pose  (geometry_msgs/PoseStamped, TRANSIENT_LOCAL)
      World-frame pose of the best marker detection so far.
      Published on every new detection and again at 1 Hz as a heartbeat.

On shutdown: saves best pose to a JSON file for offline inspection.

Configuration (ROS parameters):
  marker_id      (int,   default 1)     — ArUco marker ID to track
  marker_size    (float, default 0.050) — physical side length (m)
  camera_topic   (str)  — colour image topic
  world_frame    (str)  — global TF frame (default 'world_enu')
  camera_frame   (str)  — camera optical TF frame
  save_path      (str)  — JSON file to write on shutdown
"""

import json
import math
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped, PointStamped
from cv_bridge import CvBridge
from tf2_ros import (Buffer, TransformListener,
                     LookupException, ConnectivityException, ExtrapolationException)

_LATCHED_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)

# Camera intrinsics (turtlebot_featherstone.scn) — overridden by CameraInfo if available
_IMG_W, _IMG_H = 1920, 1080
_HFOV_DEG      = 69.0
_FX = _FY      = (_IMG_W / 2.0) / math.tan(math.radians(_HFOV_DEG / 2.0))
_CX, _CY       = _IMG_W / 2.0, _IMG_H / 2.0
_DEFAULT_CAM_MTX = np.array([[_FX, 0.0, _CX],
                               [0.0, _FY, _CY],
                               [0.0, 0.0, 1.0]], dtype=np.float64)
_DEFAULT_DIST    = np.zeros(5, dtype=np.float64)


class ArucoScoutNode(Node):

    def __init__(self):
        super().__init__('aruco_scout_node')

        # ── Parameters ────────────────────────────────────────────────────────
        self.declare_parameter('marker_id',    1)
        self.declare_parameter('marker_size',  0.050)
        self.declare_parameter('camera_topic', '/turtlebot/camera/color/image_color')
        self.declare_parameter('world_frame',  'world_enu')
        self.declare_parameter('camera_frame', 'camera_color_optical_frame')
        self.declare_parameter('save_path',
                               '/home/haadi/ROS2_Crash_Course/ros2_ws/src/hoi_control/logs/aruco_best_pose.json')

        self._marker_id    = self.get_parameter('marker_id').value
        self._marker_size  = self.get_parameter('marker_size').value
        self._camera_topic = self.get_parameter('camera_topic').value
        self._world_frame  = self.get_parameter('world_frame').value
        self._camera_frame = self.get_parameter('camera_frame').value
        self._save_path    = self.get_parameter('save_path').value

        # ── ArUco detector ────────────────────────────────────────────────────
        aruco_dict   = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
        aruco_params = cv2.aruco.DetectorParameters()
        self._detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

        h = self._marker_size / 2.0
        self._obj_pts = np.array([
            [-h,  h, 0.0], [ h,  h, 0.0],
            [ h, -h, 0.0], [-h, -h, 0.0],
        ], dtype=np.float64)

        # ── Camera intrinsics ─────────────────────────────────────────────────
        self._cam_mtx  = _DEFAULT_CAM_MTX.copy()
        self._dist     = _DEFAULT_DIST.copy()

        # ── State ─────────────────────────────────────────────────────────────
        self._best_pose: PoseStamped | None = None
        self._best_dist      = float('inf')   # closer = better
        self._detection_count = 0

        # ── TF ────────────────────────────────────────────────────────────────
        self._tf_buffer   = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._bridge      = CvBridge()

        # ── I/O ───────────────────────────────────────────────────────────────
        self.create_subscription(Image, self._camera_topic, self._camera_cb, 5)

        self._pub_best = self.create_publisher(PoseStamped, '/aruco/best_pose', _LATCHED_QOS)
        self.create_timer(1.0, self._heartbeat)

        self.get_logger().info(
            f'ArucoScoutNode started — tracking ID {self._marker_id}, '
            f'size {self._marker_size:.3f} m')

    # ── Camera callback ───────────────────────────────────────────────────────

    def _camera_cb(self, msg: Image):
        try:
            frame = self._bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception:
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self._detector.detectMarkers(gray)

        if ids is None:
            return

        detected_ids = ids.flatten().tolist()
        if self._marker_id not in detected_ids:
            return

        idx = detected_ids.index(self._marker_id)
        mc  = corners[idx]

        ok, rvec, tvec = cv2.solvePnP(
            self._obj_pts, mc.reshape(4, 2), self._cam_mtx, self._dist)
        if not ok:
            return

        dist_m = float(np.linalg.norm(tvec))
        world_pos = self._tvec_to_world(tvec)
        if world_pos is None:
            return

        self._detection_count += 1
        # Accept if this is the first detection or closer than the previous best
        if dist_m < self._best_dist:
            self._best_dist = dist_m
            pose = self._build_pose(world_pos, rvec, tvec)
            self._best_pose = pose
            self._pub_best.publish(pose)
            self.get_logger().info(
                f'[SCOUT] New best detection #{self._detection_count}: '
                f'dist={dist_m:.2f}m  world=({world_pos[0]:.2f},{world_pos[1]:.2f},{world_pos[2]:.2f})')

    def _tvec_to_world(self, tvec: np.ndarray):
        try:
            pt = PointStamped()
            pt.header.frame_id = self._camera_frame
            pt.header.stamp    = rclpy.time.Time().to_msg()
            tv = tvec.flatten()
            pt.point.x = float(tv[0])
            pt.point.y = float(tv[1])
            pt.point.z = float(tv[2])
            pt_world = self._tf_buffer.transform(
                pt, self._world_frame, timeout=Duration(seconds=0.05))
            return np.array([pt_world.point.x, pt_world.point.y, pt_world.point.z])
        except Exception as e:
            self.get_logger().warn(f'TF cam→world failed: {e}', throttle_duration_sec=2.0)
            return None

    def _build_pose(self, world_pos: np.ndarray, rvec: np.ndarray, tvec: np.ndarray) -> PoseStamped:
        """Build a PoseStamped with position in world frame and orientation from rvec."""
        pose = PoseStamped()
        pose.header.frame_id = self._world_frame
        pose.header.stamp    = self.get_clock().now().to_msg()
        pose.pose.position.x = float(world_pos[0])
        pose.pose.position.y = float(world_pos[1])
        pose.pose.position.z = float(world_pos[2])

        # Convert rvec to quaternion via rotation matrix
        R, _ = cv2.Rodrigues(rvec)
        try:
            # Transform the marker rotation into world frame
            t = self._tf_buffer.lookup_transform(
                self._world_frame, self._camera_frame, rclpy.time.Time())
            q_tf = t.transform.rotation
            # Build rotation matrix from TF quaternion
            x, y, z, w = q_tf.x, q_tf.y, q_tf.z, q_tf.w
            R_cam2world = np.array([
                [1-2*(y*y+z*z), 2*(x*y-z*w),   2*(x*z+y*w)],
                [2*(x*y+z*w),   1-2*(x*x+z*z), 2*(y*z-x*w)],
                [2*(x*z-y*w),   2*(y*z+x*w),   1-2*(x*x+y*y)],
            ])
            R_world = R_cam2world @ R
            # Convert rotation matrix to quaternion
            qw = math.sqrt(max(0.0, 1.0 + R_world[0, 0] + R_world[1, 1] + R_world[2, 2])) / 2.0
            if qw > 1e-6:
                qx = (R_world[2, 1] - R_world[1, 2]) / (4.0 * qw)
                qy = (R_world[0, 2] - R_world[2, 0]) / (4.0 * qw)
                qz = (R_world[1, 0] - R_world[0, 1]) / (4.0 * qw)
            else:
                qx = qy = qz = 0.0
            pose.pose.orientation.x = qx
            pose.pose.orientation.y = qy
            pose.pose.orientation.z = qz
            pose.pose.orientation.w = qw
        except Exception:
            pose.pose.orientation.w = 1.0

        return pose

    def _heartbeat(self):
        if self._best_pose is not None:
            # Re-publish so late subscribers (TRANSIENT_LOCAL) catch it
            self._pub_best.publish(self._best_pose)
            self.get_logger().info(
                f'[SCOUT] Heartbeat: best detection at '
                f'({self._best_pose.pose.position.x:.2f},'
                f'{self._best_pose.pose.position.y:.2f},'
                f'{self._best_pose.pose.position.z:.2f})  '
                f'total detections={self._detection_count}',
                throttle_duration_sec=10.0)

    def save_best_pose(self):
        if self._best_pose is None:
            self.get_logger().warn('[SCOUT] No detection to save')
            return
        p = self._best_pose.pose.position
        o = self._best_pose.pose.orientation
        data = {
            'marker_id': self._marker_id,
            'detection_count': self._detection_count,
            'best_dist_m': self._best_dist,
            'world_frame': self._world_frame,
            'position': {'x': p.x, 'y': p.y, 'z': p.z},
            'orientation': {'x': o.x, 'y': o.y, 'z': o.z, 'w': o.w},
            'timestamp': time.time(),
        }
        try:
            with open(self._save_path, 'w') as f:
                json.dump(data, f, indent=2)
            self.get_logger().info(f'[SCOUT] Saved best pose to {self._save_path}')
        except Exception as e:
            self.get_logger().error(f'[SCOUT] Save failed: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = ArucoScoutNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.save_best_pose()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
