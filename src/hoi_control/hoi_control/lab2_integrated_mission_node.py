#!/usr/bin/env python3
"""
lab2_integrated_mission_node.py
Pick-and-place VMS node for integrated frontier-exploration missions.

Based on lab2_pick_place_vms_node_2.py (simulation).  Key differences:
  - Starts in IDLE, not SEARCH.  The mission coordinator activates it.
  - NAVIGATE_TO_GOAL removed — the global path planner handles navigation.
  - After PICK_ASCEND → IDLE_HOLDING (arm frozen, carrying box, awaiting place command).
  - /pick_place/start_pick  (Trigger) activates picking (marker pose via topic).
  - /pick_place/start_place (Trigger) activates placing (dest pose via topic).
  - /pick_place/status      (String) streams current FSM state name.
  - /pick_place/marker_pose (PoseStamped, TRANSIENT_LOCAL) — coordinator sets before start_pick.
  - /pick_place/place_dest  (PoseStamped, TRANSIENT_LOCAL) — coordinator sets before start_place.

FSM sequence
------------
IDLE                 -> waiting for /pick_place/start_pick service
ALIGN_DIST           -> drive base to stand-off point in front of marker
ALIGN_ANGLE          -> rotate in place until face-on to marker
APPROACH_BOX_VMS     -> VMS: drive EE to approach height above box
PICK_DESCEND         -> VMS: lower EE to box top (suction contact)
SUCTION_ON           -> activate suction cup, wait SUCTION_SETTLE_S
PICK_ASCEND          -> VMS: lift EE back to approach height
IDLE_HOLDING         -> arm frozen holding box; wait for /pick_place/start_place
PLACE_VMS_APPROACH   -> VMS: drive EE to approach height above floor drop point
PLACE_DESCEND        -> VMS: lower box to floor
SUCTION_OFF          -> deactivate suction, release box
PLACE_ASCEND         -> VMS: lift EE away from floor
DONE                 -> stop all, return to IDLE
"""

import math
import time
from enum import Enum, auto

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import JointState, Image
from std_msgs.msg import Float64MultiArray, String
from std_srvs.srv import SetBool, Trigger
from geometry_msgs.msg import Twist, PointStamped, Point, PoseStamped
from visualization_msgs.msg import Marker, MarkerArray
from cv_bridge import CvBridge
from tf2_ros import (Buffer, TransformListener,
                     LookupException, ConnectivityException, ExtrapolationException)

from hoi_control.swiftpro_robotics_rrc_2 import (
    swiftpro_fk_vms_5dof,
    DLS,
    weighted_DLS,
    scale_velocities,
    Q1_MIN, Q1_MAX,
    Q2_MIN, Q2_MAX,
    Q3_MIN, Q3_MAX,
    Q4_MIN, Q4_MAX,
    JOINT_NAMES_4DOF,
    VMSRobotState,
    VMSPositionTask,
    VMSJointLimitsTask,
    vms_task_priority_step,
)

# ── Topics / Frames ────────────────────────────────────────────────────────────
JOINT_STATE_TOPIC = '/turtlebot/joint_states'
JOINT_CMD_TOPIC   = '/turtlebot/swiftpro/joint_velocity_controller/command'
BASE_CMD_TOPIC    = '/turtlebot/cmd_vel'
CAMERA_TOPIC      = '/turtlebot/camera/color/image_color'
SUCTION_SRV       = '/turtlebot/swiftpro/vacuum_gripper/set_pump'
MARKER_TOPIC      = '/hoi/integrated_mission_markers'

WORLD_FRAME   = 'world_enu'
EE_FRAME      = 'end_effector'
J1_FRAME      = 'turtlebot/swiftpro/manipulator_base_link'
BASE_FRAME    = 'turtlebot/base_footprint'
CAMERA_FRAME  = 'camera_color_optical_frame'

CONTROL_HZ = 60.0
DT = 1.0 / CONTROL_HZ

# ── RViz marker IDs ────────────────────────────────────────────────────────────
ID_TARGET   = 0
ID_TF_EE    = 1
ID_FK_EE    = 2
ID_J1_BASE  = 3
ID_ERR_LINE = 4
ID_TEXT     = 5
ID_BOX_TOP    = 6
ID_ALIGN_TGT  = 7
ID_PATH_LINE  = 8
ID_PATH_WP    = 9

# ── ArUco detection ────────────────────────────────────────────────────────────
ARUCO_DICT_ID     = cv2.aruco.DICT_ARUCO_ORIGINAL
ARUCO_MARKER_ID   = 1
ARUCO_MARKER_SIZE = 0.050

# ── Box geometry ───────────────────────────────────────────────────────────────
BOX_HEIGHT   = 0.150
BOX_HALF_W   = 0.035
MARKER_TO_BOX_TOP_Z = 0.075

# ── Approach / pick geometry ───────────────────────────────────────────────────
APPROACH_HEIGHT_ABOVE   = 0.12
PICK_ASCEND_Z_EXTRA     = 0.00
EE_TOUCH_Z_OFFSET       = -0.003
EE_TOUCH_FORWARD_OFFSET = 0.0
SUCTION_SETTLE_S        = 1.2

# ── Place geometry ─────────────────────────────────────────────────────────────
FLOOR_PLACE_FORWARD    = 0.0
PLACE_APPROACH_Z_ABOVE = 0.10
PLACE_TOUCH_Z_OFFSET   = -0.005
NAV_EE_Z               = 0.30

# ── VMS controller ─────────────────────────────────────────────────────────────
VMS_K            = 0.5
VMS_DAMPING      = 0.08
BASE_MAX_LINEAR  = 0.15
BASE_MAX_ANGULAR = 0.45
VMS_PATH_PERIOD  = 30.0
ARM_MAX_VEL      = 0.50
W_BASE_COST      = 10.0
W_ARM_COST       = 1.0
EE_REACH_TOL     = 0.005

# ── Joint limits ───────────────────────────────────────────────────────────────
LIMIT_MARGIN           = 0.10
LIMIT_HYSTERESIS_RATIO = 1.5

# ── Search / align ─────────────────────────────────────────────────────────────
SEARCH_SWEEP_ANGLE       = math.radians(45)
SEARCH_SWEEP_ANGLE_ALIGN = math.radians(90)
SEARCH_SEQUENCE          = [0.0, SEARCH_SWEEP_ANGLE, 0.0, -SEARCH_SWEEP_ANGLE, 0.0]
SEARCH_SEQUENCE_ALIGN    = [0.0, SEARCH_SWEEP_ANGLE_ALIGN, 0.0, -SEARCH_SWEEP_ANGLE_ALIGN, 0.0]
SEARCH_OMEGA             = 0.35
SEARCH_HOLD_S            = 1.0
SEARCH_FWD_DIST          = 0.50
SEARCH_FWD_VEL           = 0.10

ALIGN_TARGET_DIST    = 0.80
ALIGN_DIST_NAV_TOL   = 0.08
ALIGN_DIST_K_HEAD    = 1.20
ALIGN_DIST_K_FWD     = 0.40
ALIGN_DIST_MAX_VX    = 0.15
ALIGN_DIST_MAX_OMEGA = 0.40
ALIGN_HEAD_TOL       = math.radians(15)
ALIGN_ANGLE_TOL      = math.radians(4.0)
ALIGN_K_CENTER       = 0.80
ALIGN_MAX_OMEGA      = 0.35

# ── Camera intrinsics (turtlebot_featherstone.scn) ────────────────────────────
_IMG_W, _IMG_H = 1920, 1080
_HFOV_DEG      = 69.0
_FX = _FY      = (_IMG_W / 2.0) / math.tan(math.radians(_HFOV_DEG / 2.0))
_CX, _CY       = _IMG_W / 2.0, _IMG_H / 2.0
CAMERA_MATRIX  = np.array([[_FX, 0.0, _CX],
                             [0.0, _FY, _CY],
                             [0.0, 0.0, 1.0]], dtype=np.float64)
DIST_COEFFS    = np.zeros(5, dtype=np.float64)


# ── FSM states ─────────────────────────────────────────────────────────────────
class State(Enum):
    IDLE               = auto()   # waiting for start_pick service call
    SEARCH             = auto()   # rotate and scan (fallback if no marker pose given)
    ALIGN_DIST         = auto()   # drive base to stand-off point
    ALIGN_ANGLE        = auto()   # rotate until face-on to marker
    APPROACH_BOX_VMS   = auto()   # VMS: EE above box
    PICK_DESCEND       = auto()   # VMS: descend to box top
    SUCTION_ON         = auto()   # activate suction
    PICK_ASCEND        = auto()   # VMS: lift EE back up
    IDLE_HOLDING       = auto()   # holding box, waiting for start_place
    PLACE_VMS_APPROACH = auto()   # VMS: EE above floor drop point
    PLACE_DESCEND      = auto()   # VMS: lower box to floor
    SUCTION_OFF        = auto()   # release box
    PLACE_ASCEND       = auto()   # VMS: lift EE away from floor
    DONE               = auto()   # cycle complete → return to IDLE


# ── QoS for latched pose topics ────────────────────────────────────────────────
_LATCHED_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)


class IntegratedMissionNode(Node):

    def __init__(self):
        super().__init__('integrated_mission_node')

        # ── FSM ───────────────────────────────────────────────────────────────
        self._state            = State.IDLE
        self._state_entry_time = None

        # ── Mission pose inputs ───────────────────────────────────────────────
        self._marker_pose_received = None   # PoseStamped from /pick_place/marker_pose
        self._place_dest_received  = None   # PoseStamped from /pick_place/place_dest

        # ── Detected box target ───────────────────────────────────────────────
        self._box_top_world = None
        self._box_locked    = False

        # ── Diagnostic log ────────────────────────────────────────────────────
        self._log_entries = []
        self._log_tick    = 0

        # ── Cached robot state ────────────────────────────────────────────────
        self._arm_q        = np.zeros(4)
        self._base_x       = 0.0
        self._base_y       = 0.0
        self._base_psi     = 0.0
        self._last_tf_ee   = None
        self._link1_world  = np.zeros(3)

        # ── Search / align sub-state ──────────────────────────────────────────
        self._search_initial_psi   = None
        self._search_idx           = 0
        self._search_hold_start    = None
        self._search_advancing     = False
        self._search_advance_start = None
        self._marker_cx_px         = None
        self._marker_rvec          = None
        self._marker_tvec          = None
        self._marker_detect_time   = None
        self._align_target_world   = None
        self._align_search_psi     = None
        self._align_search_idx     = 0
        self._align_search_hold    = None

        # ── Approach / place targets ──────────────────────────────────────────
        self._approach_target    = None
        self._place_floor_target = None

        # ── VMS path-tracking ─────────────────────────────────────────────────
        self._path_start   = None
        self._path_start_t = None
        self._path_desired = None

        # ── VMS infrastructure ────────────────────────────────────────────────
        self._vms_state = VMSRobotState()
        self._pos_task  = VMSPositionTask('pos', np.zeros(3))
        self._pos_task.setGain(np.eye(3) * VMS_K)
        self._joint_limit_tasks = [
            VMSJointLimitsTask('q1_lim', 2, Q1_MIN, Q1_MAX,
                               margin=LIMIT_MARGIN, hysteresis_ratio=LIMIT_HYSTERESIS_RATIO),
            VMSJointLimitsTask('q2_lim', 3, Q2_MIN, Q2_MAX,
                               margin=LIMIT_MARGIN, hysteresis_ratio=LIMIT_HYSTERESIS_RATIO),
            VMSJointLimitsTask('q3_lim', 4, Q3_MIN, Q3_MAX,
                               margin=LIMIT_MARGIN, hysteresis_ratio=LIMIT_HYSTERESIS_RATIO),
            VMSJointLimitsTask('q4_lim', 5, Q4_MIN, Q4_MAX,
                               margin=LIMIT_MARGIN, hysteresis_ratio=LIMIT_HYSTERESIS_RATIO),
        ]

        # ── ArUco detector ────────────────────────────────────────────────────
        aruco_dict   = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)
        aruco_params = cv2.aruco.DetectorParameters()
        self._aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
        h = ARUCO_MARKER_SIZE / 2.0
        self._marker_obj_pts = np.array([
            [-h,  h, 0.0], [ h,  h, 0.0],
            [ h, -h, 0.0], [-h, -h, 0.0],
        ], dtype=np.float64)

        # ── Camera visualisation ──────────────────────────────────────────────
        self._bridge    = CvBridge()
        self._vis_frame = None
        cv2.namedWindow('ArUco Detection', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('ArUco Detection', 960, 540)

        # ── TF ────────────────────────────────────────────────────────────────
        self._tf_buffer   = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        # ── Subscribers ───────────────────────────────────────────────────────
        self.create_subscription(JointState, JOINT_STATE_TOPIC, self._js_cb, 10)
        self.create_subscription(Image, CAMERA_TOPIC, self._camera_cb, 5)
        self.create_subscription(PoseStamped, '/pick_place/marker_pose',
                                 self._marker_pose_cb, _LATCHED_QOS)
        self.create_subscription(PoseStamped, '/pick_place/place_dest',
                                 self._place_dest_cb, _LATCHED_QOS)

        # ── Publishers ────────────────────────────────────────────────────────
        self._pub_arm     = self.create_publisher(Float64MultiArray, JOINT_CMD_TOPIC, 10)
        self._pub_base    = self.create_publisher(Twist, BASE_CMD_TOPIC, 10)
        self._pub_markers = self.create_publisher(MarkerArray, MARKER_TOPIC, 10)
        self._pub_status  = self.create_publisher(String, '/pick_place/status', 10)

        # ── Suction client ────────────────────────────────────────────────────
        self._suction_cli = self.create_client(SetBool, SUCTION_SRV)

        # ── Service servers ───────────────────────────────────────────────────
        self._srv_start_pick  = self.create_service(
            Trigger, '/pick_place/start_pick',  self._start_pick_cb)
        self._srv_start_place = self.create_service(
            Trigger, '/pick_place/start_place', self._start_place_cb)

        # ── Timers ────────────────────────────────────────────────────────────
        self._ctrl_timer = self.create_timer(DT, self._control_loop)
        self._vis_timer  = self.create_timer(1.0 / 15.0, self._vis_tick)

        self.get_logger().info('IntegratedMissionNode ready — State: IDLE')

    # ── Pose topic callbacks ───────────────────────────────────────────────────

    def _marker_pose_cb(self, msg: PoseStamped):
        self._marker_pose_received = msg
        self.get_logger().info(
            f'[MISSION] Marker pose received: '
            f'({msg.pose.position.x:.2f}, {msg.pose.position.y:.2f}, {msg.pose.position.z:.2f})',
            throttle_duration_sec=5.0)

    def _place_dest_cb(self, msg: PoseStamped):
        self._place_dest_received = msg
        self.get_logger().info(
            f'[MISSION] Place destination received: '
            f'({msg.pose.position.x:.2f}, {msg.pose.position.y:.2f})',
            throttle_duration_sec=5.0)

    # ── Service callbacks ──────────────────────────────────────────────────────

    def _start_pick_cb(self, request, response):
        if self._state != State.IDLE:
            response.success = False
            response.message = f'Not in IDLE (current: {self._state.name})'
            return response

        if self._marker_pose_received is not None:
            mp = self._marker_pose_received
            mx = mp.pose.position.x
            my = mp.pose.position.y
            mz = mp.pose.position.z
            # Box top: marker world XY + vertical correction for front-face marker
            self._box_top_world = np.array([mx, my, mz + MARKER_TO_BOX_TOP_Z])
            # Stand-off: approach from current robot position direction
            dx = self._base_x - mx
            dy = self._base_y - my
            dist = math.hypot(dx, dy)
            if dist > 0.01:
                norm_x, norm_y = dx / dist, dy / dist
                self._align_target_world = np.array([
                    mx + norm_x * ALIGN_TARGET_DIST,
                    my + norm_y * ALIGN_TARGET_DIST,
                ])
                self.get_logger().info(
                    f'[MISSION] start_pick: marker=({mx:.2f},{my:.2f}) '
                    f'standoff=({self._align_target_world[0]:.2f},{self._align_target_world[1]:.2f})')
            else:
                self._align_target_world = None
                self.get_logger().warn(
                    '[MISSION] start_pick: robot too close to marker, using SEARCH')
            self._transition(State.ALIGN_DIST)
        else:
            self.get_logger().info('[MISSION] start_pick: no marker pose, starting SEARCH')
            self._transition(State.SEARCH)

        response.success = True
        response.message = 'OK'
        return response

    def _start_place_cb(self, request, response):
        if self._state != State.IDLE_HOLDING:
            response.success = False
            response.message = f'Not in IDLE_HOLDING (current: {self._state.name})'
            return response

        if self._place_dest_received is not None:
            pd = self._place_dest_received
            self._place_floor_target = np.array([
                pd.pose.position.x,
                pd.pose.position.y,
                BOX_HEIGHT + PLACE_TOUCH_Z_OFFSET,
            ])
            self.get_logger().info(
                f'[MISSION] start_place: target=({self._place_floor_target[0]:.2f},'
                f'{self._place_floor_target[1]:.2f},'
                f'{self._place_floor_target[2]:.3f})')
        else:
            self.get_logger().warn('[MISSION] start_place: no place dest, using default')
            self._place_floor_target = self._compute_floor_target()

        self._path_start   = None
        self._path_start_t = None
        self._path_desired = None
        self._transition(State.PLACE_VMS_APPROACH)

        response.success = True
        response.message = 'OK'
        return response

    # ── Joint-state callback ───────────────────────────────────────────────────

    def _js_cb(self, msg):
        pos_map = dict(zip(msg.name, msg.position))
        for i, jn in enumerate(JOINT_NAMES_4DOF):
            if jn in pos_map:
                self._arm_q[i] = pos_map[jn]

    # ── Camera callback — ArUco detection ─────────────────────────────────────

    def _camera_cb(self, msg):
        try:
            frame = self._bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception:
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self._aruco_detector.detectMarkers(gray)
        annotated = frame.copy()
        h, w = annotated.shape[:2]

        target_found = False
        detected_ids = []
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(annotated, corners, ids)
            detected_ids = ids.flatten().tolist()
            for i, mid in enumerate(detected_ids):
                mc    = corners[i]
                cx_px = int(mc[0, :, 0].mean())
                cy_px = int(mc[0, :, 1].mean())
                if mid == ARUCO_MARKER_ID:
                    label_color  = (0, 255, 0)
                    target_found = True
                else:
                    label_color = (0, 165, 255)
                cv2.putText(annotated, f'ID {mid}',
                            (cx_px - 20, cy_px - 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, label_color, 2)
                if mid == ARUCO_MARKER_ID:
                    ok, rvec, tvec = cv2.solvePnP(
                        self._marker_obj_pts, mc.reshape(4, 2),
                        CAMERA_MATRIX, DIST_COEFFS)
                    if ok:
                        cv2.drawFrameAxes(annotated, CAMERA_MATRIX, DIST_COEFFS,
                                          rvec, tvec, ARUCO_MARKER_SIZE * 0.7)
                        dist_m = float(np.linalg.norm(tvec))
                        cv2.putText(annotated, f'd = {dist_m:.3f} m',
                                    (cx_px - 45, cy_px + 25),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
                        self._marker_cx_px       = cx_px
                        self._marker_rvec        = rvec
                        self._marker_tvec        = tvec
                        self._marker_detect_time = time.time()
                        if not self._box_locked:
                            box_top = self._camera_to_world(tvec, MARKER_TO_BOX_TOP_Z)
                            if box_top is not None:
                                self._box_top_world = box_top

        cx_img = w // 2
        cv2.line(annotated, (cx_img, 55), (cx_img, h - 5), (100, 100, 255), 1)
        if self._state in (State.ALIGN_DIST, State.ALIGN_ANGLE) \
                and self._marker_rvec is not None:
            tv_ov   = self._marker_tvec.flatten()
            R_ov, _ = cv2.Rodrigues(self._marker_rvec)
            mz_ov   = R_ov @ np.array([0.0, 0.0, 1.0])
            face_ov   = math.atan2(float(mz_ov[0]), -float(mz_ov[2]))
            centre_ov = math.atan2(float(tv_ov[0]), float(tv_ov[2]))
            d_ov      = float(tv_ov[2])
            if self._state == State.ALIGN_DIST:
                color = (0, 165, 255)
                label = (f'ALIGN_DIST  dist_to_goal='
                         f'{math.hypot(self._align_target_world[0] - self._base_x, self._align_target_world[1] - self._base_y):.2f}m'
                         if self._align_target_world is not None else 'ALIGN_DIST')
            else:
                color = (0, 255, 0) if abs(centre_ov) < ALIGN_ANGLE_TOL else (0, 165, 255)
                label = (f'ctr={math.degrees(centre_ov):+.1f}deg  '
                         f'face={math.degrees(face_ov):+.1f}deg  '
                         f'dist={d_ov:.3f}m')
            cv2.putText(annotated, label,
                        (cx_img - 420, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        banner_y = 36
        if ids is None or len(detected_ids) == 0:
            cv2.rectangle(annotated, (0, 0), (w, 50), (0, 0, 180), -1)
            cv2.putText(annotated, 'NO MARKERS DETECTED',
                        (10, banner_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        elif target_found:
            cv2.rectangle(annotated, (0, 0), (w, 50), (0, 140, 0), -1)
            cv2.putText(annotated,
                        f'TARGET ID {ARUCO_MARKER_ID} DETECTED  |  IDs in view: {detected_ids}',
                        (10, banner_y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        else:
            cv2.rectangle(annotated, (0, 0), (w, 50), (0, 100, 200), -1)
            cv2.putText(annotated,
                        f'Looking for ID {ARUCO_MARKER_ID}  |  Found IDs: {detected_ids}',
                        (10, banner_y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        y = h - 110
        cv2.putText(annotated, f'FSM: {self._state.name}',
                    (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 2)
        y += 28
        if self._box_top_world is not None:
            bx, by, bz = self._box_top_world
            locked_tag = ' [LOCKED]' if self._box_locked else ' [live]'
            cv2.putText(annotated,
                        f'Box top ENU: ({bx:.2f}, {by:.2f}, {bz:.2f}){locked_tag}',
                        (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 200, 0), 2)
        y += 26
        if self._last_tf_ee is not None:
            ex, ey, ez = self._last_tf_ee
            cv2.putText(annotated,
                        f'EE ENU:     ({ex:.2f}, {ey:.2f}, {ez:.2f})',
                        (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (180, 255, 180), 2)
        y += 26
        cv2.putText(annotated,
                    f'Base: ({self._base_x:.2f}, {self._base_y:.2f})  '
                    f'psi={math.degrees(self._base_psi):.1f} deg',
                    (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 2)

        self._vis_frame = annotated

    # ── Vis timer ──────────────────────────────────────────────────────────────

    def _vis_tick(self):
        if self._vis_frame is not None:
            cv2.imshow('ArUco Detection', self._vis_frame)
        cv2.waitKey(1)

    # ── TF readiness ──────────────────────────────────────────────────────────

    def _tf_ready(self) -> bool:
        required = [
            (WORLD_FRAME, BASE_FRAME),
            (WORLD_FRAME, EE_FRAME),
            (WORLD_FRAME, J1_FRAME),
            (WORLD_FRAME, CAMERA_FRAME),
        ]
        for parent, child in required:
            try:
                if not self._tf_buffer.can_transform(parent, child, rclpy.time.Time()):
                    self.get_logger().info(
                        f'Waiting for TF: {child} → {parent}',
                        throttle_duration_sec=2.0)
                    return False
            except Exception:
                self.get_logger().info(
                    f'Waiting for TF: {child} → {parent}',
                    throttle_duration_sec=2.0)
                return False
        return True

    # ── Main control loop ──────────────────────────────────────────────────────

    def _control_loop(self):
        self._log_tick += 1

        # Publish status every tick
        status_msg = String()
        status_msg.data = self._state.name
        self._pub_status.publish(status_msg)

        if not self._tf_ready():
            return

        self._update_vms_state()

        dispatch = {
            State.IDLE:               self._run_idle,
            State.SEARCH:             self._run_search,
            State.ALIGN_DIST:         self._run_align_dist,
            State.ALIGN_ANGLE:        self._run_align_angle,
            State.APPROACH_BOX_VMS:   self._run_approach_box_vms,
            State.PICK_DESCEND:       self._run_pick_descend,
            State.SUCTION_ON:         self._run_suction_on,
            State.PICK_ASCEND:        self._run_pick_ascend,
            State.IDLE_HOLDING:       self._run_idle_holding,
            State.PLACE_VMS_APPROACH: self._run_place_vms_approach,
            State.PLACE_DESCEND:      self._run_place_descend,
            State.SUCTION_OFF:        self._run_suction_off,
            State.PLACE_ASCEND:       self._run_place_ascend,
            State.DONE:               self._run_done,
        }
        dispatch[self._state]()
        self._publish_markers()
        self._terminal_log()

    # ── Terminal log ───────────────────────────────────────────────────────────

    def _terminal_log(self):
        if self._log_tick % 20 != 0:
            return
        q  = self._arm_q
        ee = self._last_tf_ee
        pd = self._path_desired

        tgt = None
        if self._state == State.ALIGN_DIST and self._align_target_world is not None:
            tgt = np.array([self._align_target_world[0], self._align_target_world[1], 0.0])
        elif self._state == State.ALIGN_ANGLE and self._box_top_world is not None:
            tgt = self._box_top_world
        elif self._state == State.APPROACH_BOX_VMS:
            tgt = self._approach_target
        elif self._state in (State.PICK_DESCEND, State.SUCTION_ON) \
                and self._box_top_world is not None:
            bx, by, bz = self._box_top_world
            tgt = np.array([bx, by, bz + EE_TOUCH_Z_OFFSET])
        elif self._state == State.PICK_ASCEND and self._approach_target is not None:
            tgt = self._approach_target + np.array([0., 0., PICK_ASCEND_Z_EXTRA])
        elif self._state in (State.PLACE_VMS_APPROACH, State.PLACE_ASCEND) \
                and self._place_floor_target is not None:
            tgt = self._place_floor_target + np.array([0., 0., PLACE_APPROACH_Z_ABOVE])
        elif self._state in (State.PLACE_DESCEND, State.SUCTION_OFF) \
                and self._place_floor_target is not None:
            tgt = self._place_floor_target

        err    = float(np.linalg.norm(tgt - ee)) if (tgt is not None and ee is not None) else float('nan')
        err_wp = float(np.linalg.norm(pd  - ee)) if (pd  is not None and ee is not None) else float('nan')
        ee_s   = f'[{ee[0]:.3f}, {ee[1]:.3f}, {ee[2]:.3f}]' if ee is not None else 'N/A'
        tgt_s  = f'[{tgt[0]:.3f}, {tgt[1]:.3f}, {tgt[2]:.3f}]' if tgt is not None else 'N/A'
        wp_s   = f'[{pd[0]:.3f}, {pd[1]:.3f}, {pd[2]:.3f}]' if pd is not None else 'N/A'
        box_s  = (f'[{self._box_top_world[0]:.3f}, {self._box_top_world[1]:.3f}, '
                  f'{self._box_top_world[2]:.3f}]') if self._box_top_world is not None else 'N/A'
        lim_active = [t.name for t in self._joint_limit_tasks if t.isActive()]

        if self._path_start_t is not None:
            elapsed_s = (self.get_clock().now() - self._path_start_t).nanoseconds / 1e9
            alpha = float(np.clip(elapsed_s / VMS_PATH_PERIOD, 0.0, 1.0))
        else:
            alpha = float('nan')

        log = (
            f'\n{"─"*68}\n'
            f'  FSM state   : {self._state.name}\n'
            f'  BASE        : x={self._base_x:.3f}  y={self._base_y:.3f}  '
            f'psi={math.degrees(self._base_psi):.1f} deg\n'
            f'  ARM q[1-4]  : [{q[0]:.3f}, {q[1]:.3f}, {q[2]:.3f}, {q[3]:.3f}] rad\n'
            f'  EE (world)  : {ee_s}\n'
            f'  Target      : {tgt_s}\n'
            f'  Waypoint    : {wp_s}  (alpha={alpha:.3f})\n'
            f'  err_to_tgt  : {err:.4f} m\n'
            f'  err_to_wp   : {err_wp:.4f} m\n'
            f'  box_top     : {box_s}  locked={self._box_locked}\n'
            f'  active_lims : {lim_active if lim_active else "none"}\n'
            f'{"─"*68}'
        )
        self.get_logger().info(log)

    # ── FSM: IDLE ──────────────────────────────────────────────────────────────

    def _run_idle(self):
        self._send_base(0.0, 0.0)
        self._send_arm(np.zeros(4))
        self.get_logger().info(
            'IDLE — waiting for /pick_place/start_pick service call',
            throttle_duration_sec=10.0)

    # ── FSM: IDLE_HOLDING ─────────────────────────────────────────────────────

    def _run_idle_holding(self):
        self._send_base(0.0, 0.0)
        self._send_arm(np.zeros(4))
        self.get_logger().info(
            'IDLE_HOLDING — holding box, waiting for /pick_place/start_place',
            throttle_duration_sec=10.0)

    # ── FSM: SEARCH ───────────────────────────────────────────────────────────

    def _run_search(self):
        if self._marker_rvec is not None and self._marker_tvec is not None:
            target = self._compute_align_target()
            if target is not None:
                self._align_target_world = target
                self.get_logger().info(
                    f'Marker detected -> stand-off target {np.round(target, 3)} -> ALIGN_DIST')
                self._send_base(0.0, 0.0)
                self._transition(State.ALIGN_DIST)
                return

        if self._search_advancing:
            if self._search_advance_start is None:
                self._search_advance_start = time.time()
            elapsed      = time.time() - self._search_advance_start
            advance_time = SEARCH_FWD_DIST / SEARCH_FWD_VEL
            if elapsed < advance_time:
                self._send_base(SEARCH_FWD_VEL, 0.0)
            else:
                self._search_advancing     = False
                self._search_advance_start = None
                self._search_initial_psi   = self._base_psi
                self._search_idx           = 0
            return

        if self._search_initial_psi is None:
            self._search_initial_psi = self._base_psi

        target_psi = (self._search_initial_psi
                      + SEARCH_SEQUENCE[self._search_idx % len(SEARCH_SEQUENCE)])
        psi_err = _angle_wrap(target_psi - self._base_psi)

        if abs(psi_err) < 0.05:
            if self._search_hold_start is None:
                self._search_hold_start = time.time()
            self._send_base(0.0, 0.0)
            if time.time() - self._search_hold_start >= SEARCH_HOLD_S:
                self._search_idx       += 1
                self._search_hold_start = None
                if self._search_idx >= len(SEARCH_SEQUENCE):
                    self._search_idx      = 0
                    self._search_advancing = True
        else:
            omega = float(np.clip(SEARCH_OMEGA * np.sign(psi_err),
                                  -SEARCH_OMEGA, SEARCH_OMEGA))
            self._search_hold_start = None
            self._send_base(0.0, omega)

    # ── FSM: ALIGN_DIST ───────────────────────────────────────────────────────

    def _run_align_dist(self):
        if self._align_target_world is None:
            self._transition(State.ALIGN_ANGLE)
            return

        tx, ty = self._align_target_world[0], self._align_target_world[1]
        dx = tx - self._base_x
        dy = ty - self._base_y
        dist = math.hypot(dx, dy)

        if dist < ALIGN_DIST_NAV_TOL:
            self.get_logger().info(f'At stand-off (dist={dist:.3f}m) -> ALIGN_ANGLE')
            self._send_base(0.0, 0.0)
            self._transition(State.ALIGN_ANGLE)
            return

        heading_to_target = math.atan2(dy, dx)
        heading_err       = _angle_wrap(heading_to_target - self._base_psi)
        omega = float(np.clip(ALIGN_DIST_K_HEAD * heading_err,
                              -ALIGN_DIST_MAX_OMEGA, ALIGN_DIST_MAX_OMEGA))
        vx = 0.0
        if abs(heading_err) < ALIGN_HEAD_TOL:
            vx = float(np.clip(ALIGN_DIST_K_FWD * dist, 0.0, ALIGN_DIST_MAX_VX))
        self._send_base(vx, omega)
        self._send_arm(np.zeros(4))

    # ── FSM: ALIGN_ANGLE ──────────────────────────────────────────────────────

    def _run_align_angle(self):
        if self._align_search_psi is None:
            self._marker_rvec        = None
            self._marker_tvec        = None
            self._marker_detect_time = None
            self._align_search_psi   = self._base_psi
            self._align_search_idx   = 0
            self.get_logger().info(
                f'ALIGN_ANGLE entered, flushed stale PnP, psi={math.degrees(self._base_psi):.1f}°')

        if self._marker_rvec is not None:
            tv = self._marker_tvec.flatten()
            R, _         = cv2.Rodrigues(self._marker_rvec)
            mz           = R @ np.array([0.0, 0.0, 1.0])
            center_angle = math.atan2(float(tv[0]), float(tv[2]))
            face_angle   = math.atan2(float(mz[0]), -float(mz[2]))

            self.get_logger().info(
                f'ALIGN_ANGLE: ctr={math.degrees(center_angle):+.1f}° '
                f'face={math.degrees(face_angle):+.1f}° '
                f'tol={math.degrees(ALIGN_ANGLE_TOL):.1f}°',
                throttle_duration_sec=0.5)

            if abs(center_angle) < ALIGN_ANGLE_TOL:
                self.get_logger().info('Centered -> APPROACH_BOX_VMS')
                self._send_base(0.0, 0.0)
                self._transition(State.APPROACH_BOX_VMS)
                return

            omega = float(np.clip(-ALIGN_K_CENTER * center_angle,
                                  -ALIGN_MAX_OMEGA, ALIGN_MAX_OMEGA))
            self._send_base(0.0, omega)
            self._send_arm(np.zeros(4))
            self._align_search_idx  = 0
            self._align_search_hold = None
        else:
            target_psi = (self._align_search_psi
                          + SEARCH_SEQUENCE_ALIGN[self._align_search_idx % len(SEARCH_SEQUENCE_ALIGN)])
            psi_err = _angle_wrap(target_psi - self._base_psi)
            if abs(psi_err) < 0.05:
                if self._align_search_hold is None:
                    self._align_search_hold = time.time()
                self._send_base(0.0, 0.0)
                if time.time() - self._align_search_hold >= SEARCH_HOLD_S:
                    self._align_search_idx  += 1
                    self._align_search_hold  = None
            else:
                omega = float(np.clip(SEARCH_OMEGA * np.sign(psi_err),
                                      -SEARCH_OMEGA, SEARCH_OMEGA))
                self._align_search_hold = None
                self._send_base(0.0, omega)
            self._send_arm(np.zeros(4))

    # ── FSM: APPROACH_BOX_VMS ─────────────────────────────────────────────────

    def _run_approach_box_vms(self):
        if self._approach_target is None:
            bx, by, bz = self._box_top_world
            fwd = np.array([math.cos(self._base_psi), math.sin(self._base_psi), 0.0])
            self._approach_target = (np.array([bx, by, bz + APPROACH_HEIGHT_ABOVE])
                                     + EE_TOUCH_FORWARD_OFFSET * fwd)
            self._box_locked = True
            self.get_logger().info(f'Approach target: {np.round(self._approach_target, 3)}')

        W = np.diag([W_BASE_COST, W_BASE_COST, W_ARM_COST, W_ARM_COST, W_ARM_COST, W_ARM_COST])
        self._vms_nav_step(self._approach_target, weight_matrix=W)

        if self._at_position(self._approach_target, EE_REACH_TOL):
            self.get_logger().info('Approach reached -> PICK_DESCEND')
            self._transition(State.PICK_DESCEND)

    # ── FSM: PICK_DESCEND ─────────────────────────────────────────────────────

    def _run_pick_descend(self):
        bx, by, bz = self._box_top_world
        fwd = np.array([math.cos(self._base_psi), math.sin(self._base_psi), 0.0])
        pick_target = np.array([bx, by, bz + EE_TOUCH_Z_OFFSET]) + EE_TOUCH_FORWARD_OFFSET * fwd
        W = np.diag([W_BASE_COST, W_BASE_COST, W_ARM_COST, W_ARM_COST, W_ARM_COST, W_ARM_COST])
        self._vms_nav_step(pick_target, weight_matrix=W)
        if self._at_position(pick_target, EE_REACH_TOL):
            self.get_logger().info('Pick position reached -> SUCTION_ON')
            self._transition(State.SUCTION_ON)

    # ── FSM: SUCTION_ON ───────────────────────────────────────────────────────

    def _run_suction_on(self):
        if self._state_entry_time is None:
            self._call_suction(True)
            self.get_logger().info('Suction ON — waiting for settle')
        self._send_base(0.0, 0.0)
        self._send_arm(np.zeros(4))
        if self._elapsed() >= SUCTION_SETTLE_S:
            self.get_logger().info('Suction settled -> PICK_ASCEND')
            self._transition(State.PICK_ASCEND)

    # ── FSM: PICK_ASCEND ──────────────────────────────────────────────────────

    def _run_pick_ascend(self):
        ascend_target = self._approach_target + np.array([0., 0., PICK_ASCEND_Z_EXTRA])
        W = np.diag([W_BASE_COST, W_BASE_COST, W_ARM_COST, W_ARM_COST, W_ARM_COST, W_ARM_COST])
        self._vms_nav_step(ascend_target, weight_matrix=W)
        if self._at_position(ascend_target, EE_REACH_TOL):
            self.get_logger().info('Ascended -> IDLE_HOLDING (waiting for coordinator)')
            self._transition(State.IDLE_HOLDING)

    # ── FSM: PLACE_VMS_APPROACH ───────────────────────────────────────────────

    def _run_place_vms_approach(self):
        if self._place_floor_target is None:
            self._place_floor_target = self._compute_floor_target()
            self.get_logger().info(
                f'PLACE_VMS_APPROACH target latched: {np.round(self._place_floor_target, 3)}')

        approach = self._place_floor_target + np.array([0., 0., PLACE_APPROACH_Z_ABOVE])
        W = np.diag([W_BASE_COST, W_BASE_COST, W_ARM_COST, W_ARM_COST, W_ARM_COST, W_ARM_COST])
        self._vms_nav_step(approach, weight_matrix=W)
        if self._at_position(approach, EE_REACH_TOL):
            self.get_logger().info('Above floor drop point -> PLACE_DESCEND')
            self._transition(State.PLACE_DESCEND)

    # ── FSM: PLACE_DESCEND ────────────────────────────────────────────────────

    def _run_place_descend(self):
        target = self._place_floor_target
        if target is None:
            return
        W = np.diag([W_BASE_COST, W_BASE_COST, W_ARM_COST, W_ARM_COST, W_ARM_COST, W_ARM_COST])
        self._vms_nav_step(target, weight_matrix=W)
        if self._at_position(target, EE_REACH_TOL):
            self.get_logger().info('Box at floor level -> SUCTION_OFF')
            self._transition(State.SUCTION_OFF)

    # ── FSM: SUCTION_OFF ──────────────────────────────────────────────────────

    def _run_suction_off(self):
        if self._state_entry_time is None:
            self._call_suction(False)
            self.get_logger().info('Suction OFF — waiting for release')
        self._send_base(0.0, 0.0)
        self._send_arm(np.zeros(4))
        if self._elapsed() >= SUCTION_SETTLE_S:
            self.get_logger().info('Box released -> PLACE_ASCEND')
            self._transition(State.PLACE_ASCEND)

    # ── FSM: PLACE_ASCEND ─────────────────────────────────────────────────────

    def _run_place_ascend(self):
        if self._place_floor_target is None:
            self._transition(State.DONE)
            return
        approach = self._place_floor_target + np.array([0., 0., PLACE_APPROACH_Z_ABOVE])
        W = np.diag([W_BASE_COST, W_BASE_COST, W_ARM_COST, W_ARM_COST, W_ARM_COST, W_ARM_COST])
        self._vms_nav_step(approach, weight_matrix=W)
        if self._at_position(approach, EE_REACH_TOL):
            self.get_logger().info('Lifted from floor -> DONE')
            self._transition(State.DONE)

    # ── FSM: DONE ─────────────────────────────────────────────────────────────

    def _run_done(self):
        self._send_base(0.0, 0.0)
        self._send_arm(np.zeros(4))
        self.get_logger().info('Mission complete — returning to IDLE', throttle_duration_sec=5.0)
        self._reset_mission_state()
        self._transition(State.IDLE)

    # ── Mission state reset ────────────────────────────────────────────────────

    def _reset_mission_state(self):
        self._box_top_world        = None
        self._box_locked           = False
        self._marker_rvec          = None
        self._marker_tvec          = None
        self._marker_detect_time   = None
        self._align_target_world   = None
        self._align_search_psi     = None
        self._align_search_idx     = 0
        self._align_search_hold    = None
        self._approach_target      = None
        self._place_floor_target   = None
        self._search_initial_psi   = None
        self._search_idx           = 0
        self._search_hold_start    = None
        self._search_advancing     = False
        self._search_advance_start = None

    # ── VMS weighted task-priority ─────────────────────────────────────────────

    def _task_priority_weighted(self, W: np.ndarray) -> np.ndarray:
        n    = self._vms_state.getDOF()
        P    = np.eye(n)
        zeta = np.zeros((n, 1))
        for task in self._joint_limit_tasks:
            task.update(self._vms_state)
            if not task.isActive():
                continue
            Ji     = task.getJacobian()
            xi_dot = task.getGain() @ task.getError() + task.getFF()
            Ji_bar = Ji @ P
            Ji_bar_inv = DLS(Ji_bar, VMS_DAMPING)
            zeta   = zeta + Ji_bar_inv @ (xi_dot - Ji @ zeta)
            P      = P - np.linalg.pinv(Ji_bar) @ Ji_bar
        self._pos_task.update(self._vms_state)
        Ji     = self._pos_task.getJacobian()
        xi_dot = self._pos_task.getGain() @ self._pos_task.getError() + self._pos_task.getFF()
        Ji_bar = Ji @ P
        Ji_bar_inv = weighted_DLS(Ji_bar, VMS_DAMPING, W)
        zeta   = zeta + Ji_bar_inv @ (xi_dot - Ji @ zeta)
        return zeta

    # ── VMS nav step ───────────────────────────────────────────────────────────

    def _vms_nav_step(self, target_world: np.ndarray, weight_matrix=None):
        ee = self._last_tf_ee
        if self._path_start is None:
            self._path_start   = ee.copy() if ee is not None else target_world.copy()
            self._path_start_t = self.get_clock().now()

        elapsed      = (self.get_clock().now() - self._path_start_t).nanoseconds / 1e9
        alpha        = float(np.clip(elapsed / VMS_PATH_PERIOD, 0.0, 1.0))
        path_desired = self._path_start + alpha * (target_world - self._path_start)
        self._path_desired = path_desired

        ff_vel = ((target_world - self._path_start) / VMS_PATH_PERIOD
                  if alpha < 1.0 else np.zeros(3))

        self._pos_task.setDesired(path_desired.reshape(3, 1))
        self._pos_task.setGain(np.eye(3) * VMS_K)
        self._pos_task.setFF(ff_vel.reshape(3, 1))

        if weight_matrix is not None:
            zeta = self._task_priority_weighted(weight_matrix).flatten()
        else:
            tasks = self._joint_limit_tasks + [self._pos_task]
            zeta  = vms_task_priority_step(
                tasks, self._vms_state, damping=VMS_DAMPING, method=2).flatten()

        vx    = float(np.clip(zeta[0], -BASE_MAX_LINEAR,  BASE_MAX_LINEAR))
        omega = float(np.clip(zeta[1], -BASE_MAX_ANGULAR, BASE_MAX_ANGULAR))
        dq    = np.clip(zeta[2:6], -ARM_MAX_VEL, ARM_MAX_VEL)
        self._send_base(vx, omega)
        self._send_arm(dq)

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _at_position(self, target_world: np.ndarray, tol: float) -> bool:
        ee = self._last_tf_ee
        if ee is None:
            return False
        return float(np.linalg.norm(target_world - ee)) < tol

    def _send_base(self, vx: float, omega: float):
        msg = Twist()
        msg.linear.x  = float(vx)
        msg.angular.z = float(omega)
        self._pub_base.publish(msg)

    def _send_arm(self, dq: np.ndarray):
        msg = Float64MultiArray()
        msg.data = [float(v) for v in dq[:4]]
        self._pub_arm.publish(msg)

    def _call_suction(self, enable: bool):
        if not self._suction_cli.wait_for_service(timeout_sec=0.5):
            self.get_logger().warn('Suction service not available', throttle_duration_sec=2.0)
            return
        req = SetBool.Request()
        req.data = enable
        self._suction_cli.call_async(req)

    def _transition(self, new_state: State):
        self.get_logger().info(f'{self._state.name} -> {new_state.name}')
        self._state            = new_state
        self._state_entry_time = None
        self._path_start       = None
        self._path_start_t     = None
        self._path_desired     = None
        # _place_floor_target intentionally NOT reset (must persist through place sequence)

    def _elapsed(self) -> float:
        if self._state_entry_time is None:
            self._state_entry_time = time.time()
        return time.time() - self._state_entry_time

    def _update_vms_state(self):
        ee = self._tf_pos(EE_FRAME)
        if ee is not None:
            self._last_tf_ee = ee
        j1 = self._tf_pos(J1_FRAME)
        if j1 is not None:
            self._link1_world = j1
        pose = self._tf_pose(BASE_FRAME)
        if pose is not None:
            self._base_x, self._base_y, self._base_psi = pose
        ee_for_jac = self._last_tf_ee if self._last_tf_ee is not None else np.zeros(3)
        self._vms_state.update(
            ee_for_jac, [self._base_x, self._base_y],
            self._base_psi, self._arm_q, self._tf_buffer)

    def _tf_pos(self, frame: str):
        try:
            t  = self._tf_buffer.lookup_transform(WORLD_FRAME, frame, rclpy.time.Time())
            tr = t.transform.translation
            return np.array([tr.x, tr.y, tr.z])
        except (LookupException, ConnectivityException, ExtrapolationException):
            return None

    def _tf_pose(self, frame: str):
        try:
            t   = self._tf_buffer.lookup_transform(WORLD_FRAME, frame, rclpy.time.Time())
            tr  = t.transform.translation
            rot = t.transform.rotation
            siny = 2.0 * (rot.w * rot.z + rot.x * rot.y)
            cosy = 1.0 - 2.0 * (rot.y * rot.y + rot.z * rot.z)
            yaw  = math.atan2(siny, cosy)
            return float(tr.x), float(tr.y), float(yaw)
        except (LookupException, ConnectivityException, ExtrapolationException):
            return None

    def _camera_to_world(self, tvec: np.ndarray, extra_z: float = 0.0):
        try:
            pt = PointStamped()
            pt.header.frame_id = CAMERA_FRAME
            pt.header.stamp    = rclpy.time.Time().to_msg()
            tv = tvec.flatten()
            pt.point.x = float(tv[0])
            pt.point.y = float(tv[1])
            pt.point.z = float(tv[2])
            pt_world = self._tf_buffer.transform(pt, WORLD_FRAME, timeout=Duration(seconds=0.05))
            pos = np.array([pt_world.point.x, pt_world.point.y, pt_world.point.z])
            pos[2] += extra_z
            return pos
        except Exception as e:
            self.get_logger().warn(f'TF cam→world failed: {e}', throttle_duration_sec=1.0)
            return None

    def _compute_align_target(self):
        if self._marker_rvec is None or self._marker_tvec is None:
            return None
        R, _       = cv2.Rodrigues(self._marker_rvec)
        mz_cam     = R @ np.array([0.0, 0.0, 1.0])
        tv         = self._marker_tvec.flatten()
        standoff_cam = tv + ALIGN_TARGET_DIST * mz_cam
        world = self._camera_to_world(standoff_cam, 0.0)
        if world is None:
            return None
        return world[:2]

    def _compute_floor_target(self):
        """Default place target — override via /pick_place/place_dest topic."""
        return np.array([0.0, 1.8 + FLOOR_PLACE_FORWARD, BOX_HEIGHT + PLACE_TOUCH_Z_OFFSET])

    def _publish_markers(self):
        now = self.get_clock().now().to_msg()
        ma  = MarkerArray()

        def mk(mid, mtype):
            m = Marker()
            m.header.frame_id    = WORLD_FRAME
            m.header.stamp       = now
            m.ns                 = 'integrated_mission'
            m.id                 = mid
            m.type               = mtype
            m.action             = Marker.ADD
            m.pose.orientation.w = 1.0
            return m

        def sphere(mid, pos, r, g, b, size=0.025):
            m = mk(mid, Marker.SPHERE)
            m.pose.position.x = float(pos[0])
            m.pose.position.y = float(pos[1])
            m.pose.position.z = float(pos[2])
            m.scale.x = m.scale.y = m.scale.z = size
            m.color.r = r; m.color.g = g; m.color.b = b; m.color.a = 1.0
            return m

        target = None
        if self._state == State.ALIGN_DIST and self._align_target_world is not None:
            target = np.array([self._align_target_world[0], self._align_target_world[1], 0.0])
        elif self._state == State.ALIGN_ANGLE:
            target = self._box_top_world
        elif self._state == State.APPROACH_BOX_VMS:
            target = self._approach_target
        elif self._state == State.PICK_ASCEND and self._approach_target is not None:
            target = self._approach_target + np.array([0., 0., PICK_ASCEND_Z_EXTRA])
        elif self._state in (State.PICK_DESCEND, State.SUCTION_ON) \
                and self._box_top_world is not None:
            bx, by, bz = self._box_top_world
            target = np.array([bx, by, bz + EE_TOUCH_Z_OFFSET])
        elif self._state in (State.PLACE_VMS_APPROACH, State.PLACE_ASCEND) \
                and self._place_floor_target is not None:
            target = self._place_floor_target + np.array([0., 0., PLACE_APPROACH_Z_ABOVE])
        elif self._state in (State.PLACE_DESCEND, State.SUCTION_OFF) \
                and self._place_floor_target is not None:
            target = self._place_floor_target

        if target is not None:
            ma.markers.append(sphere(ID_TARGET, target, 1.0, 0.0, 0.0, 0.030))

        ee = self._last_tf_ee
        if ee is not None:
            err_norm  = float(np.linalg.norm(target - ee)) if target is not None else 0.0
            closeness = float(np.clip(1.0 - err_norm / 0.10, 0.0, 1.0))
            m = mk(ID_TF_EE, Marker.SPHERE)
            m.pose.position.x = float(ee[0])
            m.pose.position.y = float(ee[1])
            m.pose.position.z = float(ee[2])
            m.scale.x = m.scale.y = m.scale.z = 0.022
            m.color.r = 1.0 - closeness
            m.color.g = 1.0
            m.color.b = 0.0
            m.color.a = 1.0
            ma.markers.append(m)

        fk_ee = swiftpro_fk_vms_5dof(
            self._link1_world, self._arm_q, self._base_psi, self._tf_buffer)
        if fk_ee is not None:
            ma.markers.append(sphere(ID_FK_EE, fk_ee, 0.0, 0.8, 1.0, 0.015))

        ma.markers.append(sphere(ID_J1_BASE,
                                  np.array([self._base_x, self._base_y, 0.0]),
                                  1.0, 1.0, 0.0, 0.018))

        if ee is not None and target is not None:
            ml = mk(ID_ERR_LINE, Marker.LINE_STRIP)
            ml.scale.x = 0.004
            ml.color.r = ml.color.g = ml.color.b = 1.0
            ml.color.a = 0.9
            p1 = Point(); p1.x = float(ee[0]);     p1.y = float(ee[1]);     p1.z = float(ee[2])
            p2 = Point(); p2.x = float(target[0]); p2.y = float(target[1]); p2.z = float(target[2])
            ml.points = [p1, p2]
            ma.markers.append(ml)

        txt_pos = target if target is not None else np.array([self._base_x, self._base_y, 0.5])
        mt = mk(ID_TEXT, Marker.TEXT_VIEW_FACING)
        mt.pose.position.x = float(txt_pos[0])
        mt.pose.position.y = float(txt_pos[1])
        mt.pose.position.z = float(txt_pos[2]) + 0.07
        mt.scale.z = 0.025
        mt.color.r = mt.color.g = mt.color.b = mt.color.a = 1.0
        if ee is not None and target is not None:
            detail = f'  err={np.linalg.norm(target - ee):.3f}m'
        else:
            detail = ''
        mt.text = f'[{self._state.name}]{detail}'
        ma.markers.append(mt)

        if self._box_top_world is not None:
            ma.markers.append(sphere(ID_BOX_TOP, self._box_top_world, 1.0, 0.0, 1.0, 0.028))

        if self._align_target_world is not None:
            at = np.array([self._align_target_world[0], self._align_target_world[1], 0.05])
            ma.markers.append(sphere(ID_ALIGN_TGT, at, 0.0, 1.0, 1.0, 0.035))

        if self._path_start is not None and target is not None:
            ml = mk(ID_PATH_LINE, Marker.LINE_STRIP)
            ml.scale.x = 0.006
            ml.color.r = 1.0; ml.color.g = 0.5; ml.color.b = 0.0; ml.color.a = 0.8
            ps = Point()
            ps.x = float(self._path_start[0]); ps.y = float(self._path_start[1])
            ps.z = float(self._path_start[2])
            pe = Point()
            pe.x = float(target[0]); pe.y = float(target[1]); pe.z = float(target[2])
            ml.points = [ps, pe]
            ma.markers.append(ml)

        if self._path_desired is not None:
            ma.markers.append(sphere(ID_PATH_WP, self._path_desired, 1.0, 0.5, 0.0, 0.018))

        self._pub_markers.publish(ma)


# ── Utility ────────────────────────────────────────────────────────────────────

def _angle_wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def main(args=None):
    rclpy.init(args=args)
    node = IntegratedMissionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        log_path = '/home/haadi/ROS2_Crash_Course/ros2_ws/src/hoi_control/logs/integrated_mission_log.txt'
        try:
            with open(log_path, 'w') as f:
                f.write('\n'.join(node._log_entries))
        except Exception:
            pass
        try:
            node._send_base(0.0, 0.0)
            node._send_arm(np.zeros(4))
        except Exception:
            pass
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
