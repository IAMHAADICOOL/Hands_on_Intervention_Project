#!/usr/bin/env python3
"""
aruco_camera_node.py
Standalone ArUco camera overlay visualiser.

Replicates the full camera overlay from lab2_pick_place_vms_node_2.py but as a
dedicated node with no FSM, VMS, or arm control — just vision.

Overlays (matching the pick-place node exactly):
  - Detected marker outlines + IDs (green = target, orange = others)
  - PnP pose axes + distance label on the target marker
  - Centre-line alignment guide
  - Bearing-to-centre + face-angle readout (when target visible)
  - Detection status banner (top of frame)
  - HUD: FSM state from /pick_place/status, box top ENU, EE ENU, base pose
  - solvePnP tvec components (Tx, Ty, Tz) — useful for tuning geometry constants

Topics subscribed:
  <camera_topic>       (sensor_msgs/Image)   — colour image
  /pick_place/status   (std_msgs/String)      — FSM state from pick-place node
  /turtlebot/odom      (nav_msgs/Odometry)    — base pose for HUD

Parameters:
  camera_topic   (str,   default '/turtlebot/camera/color/image_color')
  marker_id      (int,   default 1)
  marker_size    (float, default 0.050)  — physical side length in metres
  window_w       (int,   default 960)
  window_h       (int,   default 540)
"""

import math

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge


# Camera intrinsics (turtlebot_featherstone.scn)
_IMG_W, _IMG_H = 1920, 1080
_HFOV_DEG      = 69.0
_FX = _FY      = (_IMG_W / 2.0) / math.tan(math.radians(_HFOV_DEG / 2.0))
_CX, _CY       = _IMG_W / 2.0, _IMG_H / 2.0
_DEFAULT_CAM_MTX = np.array([[_FX, 0.0, _CX],
                               [0.0, _FY, _CY],
                               [0.0, 0.0, 1.0]], dtype=np.float64)
_DEFAULT_DIST    = np.zeros(5, dtype=np.float64)

ALIGN_ANGLE_TOL = math.radians(4.0)


class ArucoCameraNode(Node):

    def __init__(self):
        super().__init__('aruco_camera_node')

        # ── Parameters ────────────────────────────────────────────────────────
        self.declare_parameter('camera_topic', '/turtlebot/camera/color/image_color')
        self.declare_parameter('marker_id',    1)
        self.declare_parameter('marker_size',  0.050)
        self.declare_parameter('window_w',     960)
        self.declare_parameter('window_h',     540)

        self._camera_topic = self.get_parameter('camera_topic').value
        self._marker_id    = self.get_parameter('marker_id').value
        self._marker_size  = self.get_parameter('marker_size').value
        self._window_w     = self.get_parameter('window_w').value
        self._window_h     = self.get_parameter('window_h').value

        # ── ArUco detector ────────────────────────────────────────────────────
        aruco_dict   = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
        aruco_params = cv2.aruco.DetectorParameters()
        self._detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

        h = self._marker_size / 2.0
        self._obj_pts = np.array([
            [-h,  h, 0.0], [ h,  h, 0.0],
            [ h, -h, 0.0], [-h, -h, 0.0],
        ], dtype=np.float64)

        # ── State ─────────────────────────────────────────────────────────────
        self._bridge       = CvBridge()
        self._vis_frame    = None
        self._fsm_status   = 'N/A'
        self._base_x       = 0.0
        self._base_y       = 0.0
        self._base_psi     = 0.0

        # Latest PnP result (kept across frames so HUD doesn't flicker)
        self._marker_rvec        = None
        self._marker_tvec        = None
        self._marker_cx_px       = None

        # ── OpenCV window ─────────────────────────────────────────────────────
        cv2.namedWindow('ArUco Camera', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('ArUco Camera', self._window_w, self._window_h)

        # ── Subscribers ───────────────────────────────────────────────────────
        self.create_subscription(Image, self._camera_topic, self._camera_cb, 5)
        self.create_subscription(String, '/pick_place/status', self._status_cb, 10)
        self.create_subscription(Odometry, '/turtlebot/odom', self._odom_cb, 10)

        # ── Vis timer (15 Hz display) ─────────────────────────────────────────
        self.create_timer(1.0 / 15.0, self._vis_tick)

        self.get_logger().info(
            f'ArucoCameraNode started — tracking ID {self._marker_id}  '
            f'topic={self._camera_topic}')

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _status_cb(self, msg: String):
        self._fsm_status = msg.data

    def _odom_cb(self, msg: Odometry):
        self._base_x = msg.pose.pose.position.x
        self._base_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self._base_psi = math.atan2(siny, cosy)

    def _camera_cb(self, msg: Image):
        try:
            frame = self._bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception:
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self._detector.detectMarkers(gray)
        annotated = frame.copy()
        h, w = annotated.shape[:2]

        # ── Marker detection ──────────────────────────────────────────────────
        target_found = False
        detected_ids = []

        if ids is not None:
            cv2.aruco.drawDetectedMarkers(annotated, corners, ids)
            detected_ids = ids.flatten().tolist()

            for i, mid in enumerate(detected_ids):
                mc    = corners[i]
                cx_px = int(mc[0, :, 0].mean())
                cy_px = int(mc[0, :, 1].mean())

                if mid == self._marker_id:
                    label_color  = (0, 255, 0)
                    target_found = True
                else:
                    label_color = (0, 165, 255)

                cv2.putText(annotated, f'ID {mid}',
                            (cx_px - 20, cy_px - 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, label_color, 2)

                if mid == self._marker_id:
                    ok, rvec, tvec = cv2.solvePnP(
                        self._obj_pts, mc.reshape(4, 2),
                        _DEFAULT_CAM_MTX, _DEFAULT_DIST)
                    if ok:
                        cv2.drawFrameAxes(annotated, _DEFAULT_CAM_MTX, _DEFAULT_DIST,
                                          rvec, tvec, self._marker_size * 0.7)
                        dist_m = float(np.linalg.norm(tvec))
                        cv2.putText(annotated, f'd = {dist_m:.3f} m',
                                    (cx_px - 45, cy_px + 25),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
                        self._marker_cx_px = cx_px
                        self._marker_rvec  = rvec
                        self._marker_tvec  = tvec

        # ── Centre-line + bearing readout ─────────────────────────────────────
        cx_img = w // 2
        cv2.line(annotated, (cx_img, 55), (cx_img, h - 5), (100, 100, 255), 1)

        if self._marker_rvec is not None and self._marker_tvec is not None:
            tv_ov   = self._marker_tvec.flatten()
            R_ov, _ = cv2.Rodrigues(self._marker_rvec)
            mz_ov   = R_ov @ np.array([0.0, 0.0, 1.0])
            face_ov   = math.atan2(float(mz_ov[0]), -float(mz_ov[2]))
            centre_ov = math.atan2(float(tv_ov[0]),  float(tv_ov[2]))
            d_ov      = float(tv_ov[2])
            color     = (0, 255, 0) if abs(centre_ov) < ALIGN_ANGLE_TOL else (0, 165, 255)
            label     = (f'ctr={math.degrees(centre_ov):+.1f}deg  '
                         f'face={math.degrees(face_ov):+.1f}deg  '
                         f'dist={d_ov:.3f}m')
            cv2.putText(annotated, label,
                        (cx_img - 420, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            # Pixel offset from image centre (alignment guide)
            if self._marker_cx_px is not None:
                px_err = self._marker_cx_px - cx_img
                cv2.putText(annotated, f'px_err={px_err:+d}px',
                            (cx_img - 420, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 1)

        # ── tvec readout (raw PnP output — useful for geometry tuning) ─────────
        if self._marker_tvec is not None:
            tv = self._marker_tvec.flatten()
            cv2.putText(annotated,
                        f'tvec  Tx={tv[0]:+.3f}  Ty={tv[1]:+.3f}  Tz={tv[2]:+.3f} m',
                        (10, h - 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180, 180, 255), 1)

        # ── Detection status banner (top of frame) ────────────────────────────
        banner_y = 36
        if ids is None or len(detected_ids) == 0:
            cv2.rectangle(annotated, (0, 0), (w, 50), (0, 0, 180), -1)
            cv2.putText(annotated, 'NO MARKERS DETECTED',
                        (10, banner_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        elif target_found:
            cv2.rectangle(annotated, (0, 0), (w, 50), (0, 140, 0), -1)
            cv2.putText(annotated,
                        f'TARGET ID {self._marker_id} DETECTED  |  IDs in view: {detected_ids}',
                        (10, banner_y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        else:
            cv2.rectangle(annotated, (0, 0), (w, 50), (0, 100, 200), -1)
            cv2.putText(annotated,
                        f'Looking for ID {self._marker_id}  |  Found IDs: {detected_ids}',
                        (10, banner_y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        # ── HUD (bottom-left) ─────────────────────────────────────────────────
        y = h - 110
        cv2.putText(annotated, f'FSM: {self._fsm_status}',
                    (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 2)
        y += 28
        cv2.putText(annotated,
                    f'Base: ({self._base_x:.2f}, {self._base_y:.2f})  '
                    f'psi={math.degrees(self._base_psi):.1f} deg',
                    (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 2)
        y += 26
        if self._marker_tvec is not None:
            tv = self._marker_tvec.flatten()
            dist_m = float(np.linalg.norm(tv))
            cv2.putText(annotated,
                        f'Marker dist: {dist_m:.3f} m',
                        (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 0), 2)

        self._vis_frame = annotated

    # ── Vis timer ──────────────────────────────────────────────────────────────

    def _vis_tick(self):
        if self._vis_frame is not None:
            cv2.imshow('ArUco Camera', self._vis_frame)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = ArucoCameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
