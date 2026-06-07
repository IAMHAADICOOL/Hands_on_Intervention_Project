#!/usr/bin/env python3
"""
lab2_pick_place_vms_node.py
Autonomous ArUco box pick-and-place using VMS (vehicle-manipulator system)
and arm-only resolved-rate FK control.

FSM sequence
------------
SEARCH               -> rotate base, scan for ArUco marker
ALIGN_DIST           -> drive base to stand-off point in front of marker
ALIGN_ANGLE          -> rotate in place until face-on to marker
APPROACH_BOX_VMS     -> VMS: drive EE to approach height above box
PICK_DESCEND         -> arm-only: lower EE to box top (suction contact)
SUCTION_ON           -> activate suction cup, wait SUCTION_SETTLE_S
PICK_ASCEND          -> arm-only: lift EE back to approach height
NAVIGATE_TO_GOAL     -> VMS: drive robot to destination carrying box
PLACE_VMS_APPROACH   -> VMS: drive EE to approach height above floor drop point
PLACE_DESCEND        -> arm-only: lower box to floor
SUCTION_OFF          -> deactivate suction, release box
PLACE_ASCEND         -> arm-only: lift EE away from floor
DONE                 -> stop

All geometry offsets are tunable constants at the top of this file.
"""

from ast import If
import math
import time
from enum import Enum, auto

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from sensor_msgs.msg import JointState, Image
from std_msgs.msg import Float64MultiArray
from std_srvs.srv import SetBool
from geometry_msgs.msg import Twist, PointStamped, Point
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

# Topics / Frames 
JOINT_STATE_TOPIC = '/turtlebot/joint_states'
JOINT_CMD_TOPIC   = '/turtlebot/swiftpro/joint_velocity_controller/command'
BASE_CMD_TOPIC    = '/turtlebot/cmd_vel'
CAMERA_TOPIC      = '/turtlebot/camera/color/image_color'
SUCTION_SRV       = '/turtlebot/swiftpro/vacuum_gripper/set_pump'
MARKER_TOPIC      = '/hoi/pick_place_markers'

WORLD_FRAME   = 'world_enu'
EE_FRAME      = 'end_effector'
J1_FRAME      = 'turtlebot/swiftpro/manipulator_base_link'
BASE_FRAME    = 'turtlebot/base_footprint'
CAMERA_FRAME  = 'camera_color_optical_frame'

CONTROL_HZ = 60.0
DT = 1.0 / CONTROL_HZ

# RViz marker IDs 
ID_TARGET   = 0   # RED    sphere — current VMS/arm target position
ID_TF_EE    = 1   # GREEN/YELLOW sphere — true EE from TF (yellow=far, green=close)
ID_FK_EE    = 2   # CYAN   sphere — VMS FK estimate of EE
ID_J1_BASE  = 3   # YELLOW sphere — J1 arm base position
ID_ERR_LINE = 4   # WHITE  line   — error vector EE → target
ID_TEXT     = 5   # WHITE  text   — FSM state + error magnitude
ID_BOX_TOP    = 6   # MAGENTA sphere — detected box top position (world_enu)
ID_ALIGN_TGT  = 7   # CYAN   sphere — ALIGN_DIST stand-off target (world XY)
ID_PATH_LINE  = 8   # ORANGE line   — straight-line path start → VMS goal
ID_PATH_WP    = 9   # ORANGE sphere — current interpolated waypoint being tracked


# ArUco detection
# The texture files are named aruco_original*.png -> DICT_ARUCO_ORIGINAL is the
# most likely match for the Stonefish aruco_box texture.
# If detection still fails, try uncommenting a different dictionary below:
#   cv2.aruco.DICT_4X4_50
#   cv2.aruco.DICT_4X4_100
#   cv2.aruco.DICT_4X4_250
#   cv2.aruco.DICT_5X5_50
#   cv2.aruco.DICT_5X5_100
#   cv2.aruco.DICT_5X5_250
#   cv2.aruco.DICT_6X6_250
#   cv2.aruco.DICT_7X7_250
ARUCO_DICT_ID     = cv2.aruco.DICT_ARUCO_ORIGINAL
ARUCO_MARKER_ID   = 1       # which marker ID to track (tune once detection works)
ARUCO_MARKER_SIZE = 0.050   # physical marker side length (m)

# Tunable geometry offsets 
# Box geometry (from aruco_box.obj: 70×70×150 mm)
BOX_HEIGHT   = 0.150    # m  total box height
BOX_HALF_W   = 0.035    # m  half-width (x and y)

# When PnP detects the marker centre, how far above it is the box top?
# Set to 0 if marker is on the TOP face.
# Set to ~BOX_HEIGHT/2 if marker is on the FRONT face (most likely).
MARKER_TO_BOX_TOP_Z = 0.075   # m  (tune: 0 = top face, 0.075 = front face mid-height)

# Approach clearance above the box top before descending to pick
APPROACH_HEIGHT_ABOVE = 0.12   # m above box top

# Extra Z added on top of the approach target during PICK_ASCEND so the arm
# lifts slightly higher before the next state takes over.
PICK_ASCEND_Z_EXTRA = 0.00     # m (tune: 0 = same as approach, positive = higher)

# How close the EE needs to be to the box top for suction contact
# (positive = EE slightly above, negative = pressed in)
EE_TOUCH_Z_OFFSET = -0.003      # m above box top surface

# Forward offset applied along the robot's heading direction (not world-X).
# Positive = shift pick point further from the robot (away from base),
# Negative = shift pick point closer to the robot.
# Components are automatically decomposed: delta_x = offset·cos((sigh)), delta_y = offset·sin((sigh)).
EE_TOUCH_FORWARD_OFFSET = 0.0   # m along robot heading (tune: range -0.05 – +0.05)
# Suction settle time (seconds) after toggling the pump
SUCTION_SETTLE_S = 1.2

# Navigation goal (world_enu coordinates the BASE should reach)
GOAL_BASE_X  = 0.0     # m  (East)
GOAL_BASE_Y  = 1.8     # m  (North, i.e. 1.8 m forward from start)
GOAL_TOL_XY  = 0.15    # m  base position tolerance to declare "at goal"
NAV_EE_Z     = 0.30    # m  EE height during VMS navigation

# Floor placement after navigation: place this far in front of the arm base
FLOOR_PLACE_FORWARD = 0.0    # m in front of arm base (tune for arm reach)

# Place-phase Z offsets (tune these two for floor placement)
# PLACE_APPROACH_Z_ABOVE : clearance above the drop point for PLACE_VMS_APPROACH
#   and PLACE_ASCEND.  Reduce to shorten PLACE_DESCEND travel; increase if arm
#   needs more clearance on the way down.
# PLACE_TOUCH_Z_OFFSET   : added to BOX_HEIGHT to get the PLACE_DESCEND target Z.
#   Positive -> EE releases box slightly above floor (box drops gently, no pressing).
#   Zero     -> EE exactly at box top when box rests on floor.
#   Negative -> EE presses box into ground (avoid).
#   Tune: start +0.03, reduce toward 0 once placement looks stable.
PLACE_APPROACH_Z_ABOVE = 0.10   # m  (typical range 0.06 – 0.15)
# _place_floor_target is the floor-level position where the box bottom should touch the ground. 
# PLACE_APPROACH_Z_ABOVE is a fixed vertical offset added on top of that to get a safe hover height.
PLACE_TOUCH_Z_OFFSET   = -0.005   # m  (range -0.01 – +0.05)

# VMS controller gains / limits
VMS_K            = 0.5    # proportional gain for VMSPositionTask
VMS_DAMPING      = 0.08   # DLS damping λ
BASE_MAX_LINEAR  = 0.15   # m/s base linear speed cap
BASE_MAX_ANGULAR = 0.45   # rad/s base yaw-rate cap

VMS_PATH_PERIOD = 30.0   # s  path period for VMS states (approach, navigation)
# PLACE_APPROACH_PATH_PERIOD = 20.0   # s  (Bézier arc — unused)
# VMS_CURVE_HEIGHT     = 0.0    # m  (Bézier apex lift — unused)
# VMS_CURVE_LATERAL    = -0.40  # m  (Bézier sideways pull — unused)
# VMS_CURVE_P2_PULLBACK = 0.15  # m  (Bézier P2 pullback — unused)

ARM_MAX_VEL = 0.50   # rad/s per joint (used for arm velocity capping in vms_step)

# Weighted DLS cost factors for the swing phase of PICK_DESCEND.
# Higher W_BASE_COST -> optimizer strongly prefers arm joints over base movement.
# Ratio of 10:1 means the base is 10× more "expensive" than arm joints.
W_BASE_COST = 10.0   # cost on vx and ω (indices 0, 1 in quasi-velocity space)
W_ARM_COST  = 1.0    # cost on dq1–dq4 (indices 2–5)

# Goal-reached tolerance for arm states (EE position error in metres)
EE_REACH_TOL = 0.005   # m
# EE_REACH_TOL = 0.005   # m
# XY alignment threshold for PICK_DESCEND swing phase:
# once the EE is within this horizontal distance of the box centre, descend.
# XY_ALIGN_TOL = 0.030   # m

# Joint limit avoidance — matches lab2_rrc_methods_vms_node settings
LIMIT_MARGIN           = 0.10   # rad — activation threshold alpha
LIMIT_HYSTERESIS_RATIO = 1.5    # delta = margin x ratio; must be > 1 to avoid chatter

# Approach standoff — EE approach target is this far behind the box so the
# base parks at a safe distance and only the arm moves for the final pick.
# Roughly equal to the arm's max forward reach (~0.20 m).
# APPROACH_STANDOFF  = 0.20   # m behind box (in robot heading direction)

# Search sweep 
SEARCH_SWEEP_ANGLE = math.radians(45)  # rad each side (tune: 30°–90°)
SEARCH_SWEEP_ANGLE_ALIGN = math.radians(90)  # rad each side (tune: 30°–90°)
SEARCH_SEQUENCE    = [0.0, SEARCH_SWEEP_ANGLE, 0.0, -SEARCH_SWEEP_ANGLE, 0.0]
SEARCH_SEQUENCE_ALIGN    = [0.0, SEARCH_SWEEP_ANGLE_ALIGN, 0.0, -SEARCH_SWEEP_ANGLE_ALIGN, 0.0]
SEARCH_OMEGA       = 0.35    # rad/s angular speed during search rotation
SEARCH_HOLD_S      = 1.0     # seconds to hold each orientation before checking
SEARCH_FWD_DIST    = 0.50    # m to advance forward after each complete sweep
SEARCH_FWD_VEL     = 0.10    # m/s during forward advance

# ALIGN_DIST: drive base to stand-off point computed from marker pose 
# Robot drives to a world-frame XY target (marker_pos + ALIGN_TARGET_DIST *
# marker_normal). No angular correction during this phase.
ALIGN_TARGET_DIST    = 0.80   # m  stand-off from marker face (tune)
ALIGN_DIST_NAV_TOL   = 0.08   # m  XY arrival tolerance
ALIGN_DIST_K_HEAD    = 1.20   # rad/s per rad  heading-to-target P-gain
ALIGN_DIST_K_FWD     = 0.40   # m/s  per m     forward speed P-gain
ALIGN_DIST_MAX_VX    = 0.15   # m/s cap
ALIGN_DIST_MAX_OMEGA = 0.40   # rad/s cap
ALIGN_HEAD_TOL       = math.radians(15)  # must face target before driving fwd

# ALIGN_ANGLE: rotate in place until face-on 
# From the stand-off point, sweep to re-find marker then align.
ALIGN_ANGLE_TOL    = math.radians(4.0)  # rad  face-on + centering tolerance
# ALIGN_K_ANGLE      = 0.30              # rad/s per rad — face-on error (rvec)
ALIGN_K_CENTER     = 0.80              # rad/s per rad — bearing-to-centre error
ALIGN_MAX_OMEGA    = 0.35              # rad/s cap


# Finite State Machine── 
class State(Enum):
    # auto() auto-assigns integer values (1, 2, ) — 
    # we don't care what the numbers are, we only ever compare 
    # by name like self._state == State.PICK_DESCEND
    SEARCH             = auto()
    ALIGN_DIST         = auto()   # drive base to stand-off point in front of marker
    ALIGN_ANGLE        = auto()   # rotate in place until face-on to marker
    APPROACH_BOX_VMS   = auto()   # VMS: EE above box (pick approach)
    PICK_DESCEND       = auto()   # arm-only: descend to box top
    SUCTION_ON         = auto()   # activate suction
    PICK_ASCEND        = auto()   # arm-only: lift EE back up
    NAVIGATE_TO_GOAL   = auto()   # VMS: drive to destination carrying box
    PLACE_VMS_APPROACH = auto()   # VMS: EE above floor drop point at destination
    PLACE_DESCEND      = auto()   # arm-only: lower box to floor
    SUCTION_OFF        = auto()   # deactivate suction (release box)
    PLACE_ASCEND       = auto()   # arm-only: lift EE away from floor
    DONE               = auto()
    # old states — kept for reference, not active in current FSM 
    # PLACE_APPROACH    = auto()   # old: approach robot back (unreachable)
    # PLACE_DROP        = auto()   # old: lower onto robot back
    # SUCTION_OFF_1     = auto()
    # RETREAT_FROM_DROP = auto()
    # UNLOAD_APPROACH   = auto()
    # UNLOAD_GRAB       = auto()
    # SUCTION_ON_2      = auto()
    # UNLOAD_ASCEND     = auto()
    # FLOOR_APPROACH    = auto()
    # FLOOR_DROP        = auto()
    # SUCTION_OFF_FINAL = auto()


# Camera intrinsics (from turtlebot_featherstone.scn) 
_IMG_W, _IMG_H = 1920, 1080
_HFOV_DEG      = 69.0
_FX = _FY      = (_IMG_W / 2.0) / math.tan(math.radians(_HFOV_DEG / 2.0))
_CX, _CY       = _IMG_W / 2.0, _IMG_H / 2.0
CAMERA_MATRIX  = np.array([[_FX, 0.0, _CX],
                             [0.0, _FY, _CY],
                             [0.0, 0.0, 1.0]], dtype=np.float64)
DIST_COEFFS    = np.zeros(5, dtype=np.float64)  # simulation: no distortion


# Bézier helpers (was trying something, didn't delete it in case it's useful for future path-tracking improvements)
# def _bezier(t: float, P0: np.ndarray, P1: np.ndarray, P2: np.ndarray) -> np.ndarray:
#     """Position on quadratic Bézier at parameter t ∈ [0, 1]."""
#     return (1.0 - t)**2 * P0 + 2.0 * (1.0 - t) * t * P1 + t**2 * P2
#
# def _bezier_vel(t: float, P0: np.ndarray, P1: np.ndarray, P2: np.ndarray) -> np.ndarray:
#     """Tangent (derivative w.r.t. t) of quadratic Bézier."""
#     return 2.0 * (1.0 - t) * (P1 - P0) + 2.0 * t * (P2 - P1)
#
# def _bezier_cubic(t: float,
#                   P0: np.ndarray, P1: np.ndarray,
#                   P2: np.ndarray, P3: np.ndarray) -> np.ndarray:
#     """Position on cubic Bézier at parameter t ∈ [0, 1]."""
#     u = 1.0 - t
#     return u**3*P0 + 3.0*u**2*t*P1 + 3.0*u*t**2*P2 + t**3*P3
#
# def _bezier_cubic_vel(t: float,
#                       P0: np.ndarray, P1: np.ndarray,
#                       P2: np.ndarray, P3: np.ndarray) -> np.ndarray:
#     """Tangent (derivative w.r.t. t) of cubic Bézier."""
#     u = 1.0 - t
#     return 3.0*(u**2*(P1-P0) + 2.0*u*t*(P2-P1) + t**2*(P3-P2))


class PickPlaceVMSNode(Node):

    def __init__(self):
        super().__init__('pick_place_vms_node')

        # state machine 
        self._state            = State.SEARCH
        self._state_entry_time = None   # wall-clock time of last transition

        # detected box target (world_enu) 
        self._box_top_world = None   # [x, y, z] of detected box top in world_enu
        self._box_locked    = False  # True once we commit to a pick position

        # diagnostic log (saved to file on Ctrl+C) 
        self._log_entries   = []
        self._log_tick      = 0     # incremented each control tick

        # cached robot state 
        self._arm_q            = np.zeros(4)
        self._base_x           = 0.0
        self._base_y           = 0.0
        self._base_psi         = 0.0
        self._last_tf_ee       = None
        self._link1_world      = np.zeros(3)

        # search / align sub-state 
        self._search_initial_psi   = None
        self._search_idx           = 0
        self._search_hold_start    = None
        self._search_advancing     = False   # True while driving forward between sweeps
        self._search_advance_start = None    # wall time when forward advance began
        self._marker_cx_px         = None    # x pixel of target marker centre (overlay only)
        self._marker_rvec          = None    # rvec from latest solvePnP
        self._marker_tvec          = None    # tvec from latest solvePnP
        self._marker_detect_time   = None    # wall time of last solvePnP success
        self._align_target_world   = None    # world XY stand-off point (ALIGN_DIST target)
        self._align_search_psi     = None    # heading reference for ALIGN_ANGLE sweep
        self._align_search_idx     = 0       # current step in sweep sequence
        self._align_search_hold    = None    # hold-start time at each sweep waypoint

        # approach cached targets 
        self._approach_target     = None  # standoff position above box
        self._place_floor_target  = None  # latched in PLACE_VMS_APPROACH (fixed world pos)

        # VMS path-tracking state 
        # Reset on every state transition; latched on the first VMS tick.
        self._path_start       = None   # EE position at path start (np.array(3))
        self._path_start_t     = None   # rclpy.time.Time when path began
        self._path_desired     = None   # current Bézier waypoint (for visualisation)
        # self._path_control_pt  = None   # cubic Bézier P1 (unused)
        # self._path_control_pt2 = None   # cubic Bézier P2 (unused)

        # VMS infrastructure 
        self._vms_state = VMSRobotState()

        self._pos_task = VMSPositionTask('pos', np.zeros(3))
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

        # ArUco detector
        aruco_dict   = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_ID)
        aruco_params = cv2.aruco.DetectorParameters()
        self._aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

        # Marker corner object points (marker centred at origin, in marker plane)
        h = ARUCO_MARKER_SIZE / 2.0
        self._marker_obj_pts = np.array([
            [-h,  h, 0.0],
            [ h,  h, 0.0],
            [ h, -h, 0.0],
            [-h, -h, 0.0],
        ], dtype=np.float64)

        # camera visualisation 
        self._bridge        = CvBridge()
        self._vis_frame     = None   # latest annotated frame (updated in camera cb)
        cv2.namedWindow('ArUco Detection', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('ArUco Detection', 960, 540)

        # ROS I/O 
        self._tf_buffer   = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self._sub_js  = self.create_subscription(
            JointState, JOINT_STATE_TOPIC, self._js_cb, 10)
        self._sub_img = self.create_subscription(
            Image, CAMERA_TOPIC, self._camera_cb, 5)

        self._pub_arm     = self.create_publisher(
            Float64MultiArray, JOINT_CMD_TOPIC, 10)
        self._pub_base    = self.create_publisher(Twist, BASE_CMD_TOPIC, 10)
        self._pub_markers = self.create_publisher(MarkerArray, MARKER_TOPIC, 10)

        self._suction_cli = self.create_client(SetBool, SUCTION_SRV)

        # timers
        self._ctrl_timer = self.create_timer(DT, self._control_loop)
        self._vis_timer  = self.create_timer(1.0 / 15.0, self._vis_tick)

        self.get_logger().info('PickPlaceVMSNode started — State: SEARCH')

    # Joint-state callback 
    def _js_cb(self, msg):
        pos_map = dict(zip(msg.name, msg.position))
        for i, jn in enumerate(JOINT_NAMES_4DOF):
            if jn in pos_map:
                self._arm_q[i] = pos_map[jn]

    # Camera callback — ArUco detection + visualisation 
    def _camera_cb(self, msg):
        try:
            frame = self._bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception:
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self._aruco_detector.detectMarkers(gray)
        # corners is a list of 4-corner arrays (one per detected marker), ids is an array of their integer IDs, _ discards rejected candidates.
        annotated = frame.copy()
        h, w = annotated.shape[:2]

        # Draw ALL detected markers so you can see which ID is on the box 
        target_found = False
        detected_ids = []
        if ids is not None:
        # ids is None when zero markers are detected, so this guard prevents the loop from running on an empty frame.
            # Draw outlines + ID labels for every detected marker
            cv2.aruco.drawDetectedMarkers(annotated, corners, ids)
            # Draws green outlines around every detected marker (not just our target).
            detected_ids = ids.flatten().tolist()
            # ids comes out as a 2D array [[1], [5], ...], .flatten().tolist() turns it into [1, 5, ...] for easy indexing
            for i, mid in enumerate(detected_ids):
                mc     = corners[i]
                cx_px  = int(mc[0, :, 0].mean())
                cy_px  = int(mc[0, :, 1].mean())
                # cx_px/cy_px compute the pixel centre by averaging the x and y coordinates of the 4 corners
                
                # Highlight our target ID with a different color label
                if mid == ARUCO_MARKER_ID:
                    label_color = (0, 255, 0)      # green  — this is the one we want
                    target_found = True
                else:
                    label_color = (0, 165, 255)    # orange — detected but not our target

                cv2.putText(annotated, f'ID {mid}',
                            (cx_px - 20, cy_px - 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, label_color, 2)

                # Full PnP + axes only for our target marker
                if mid == ARUCO_MARKER_ID:
                    ok, rvec, tvec = cv2.solvePnP(
                        self._marker_obj_pts,
                        mc.reshape(4, 2),
                        CAMERA_MATRIX,
                        DIST_COEFFS,
                    )
                    # tvec = translation (3D position of marker centre in camera frame)
                    if ok:
                        cv2.drawFrameAxes(annotated, CAMERA_MATRIX, DIST_COEFFS,
                                          rvec, tvec, ARUCO_MARKER_SIZE * 0.7)
                        dist_m = float(np.linalg.norm(tvec))
                        cv2.putText(annotated,
                                    f'd = {dist_m:.3f} m',
                                    (cx_px - 45, cy_px + 25),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

                        # Track pixel centre (overlay) and raw PnP result
                        self._marker_cx_px      = cx_px
                        self._marker_rvec       = rvec
                        self._marker_tvec       = tvec
                        self._marker_detect_time = time.time()
                        # Saves the detection result to instance variables so 
                        # other parts of the FSM (e.g. ALIGN_ANGLE) can read 
                        # the latest pose without re-running PnP
                        
                        if not self._box_locked:
                            box_top = self._camera_to_world(tvec, MARKER_TO_BOX_TOP_Z)
                            if box_top is not None:
                                self._box_top_world = box_top

        # Alignment guide: vertical centre line + pixel error (ALIGN state)
        cx_img = w // 2
        cv2.line(annotated, (cx_img, 55), (cx_img, h - 5), (100, 100, 255), 1)
        # (cx_img, 55) — start point: horizontally centered, 55 pixels from the top (skips the top strip where text overlays are)
        # (cx_img, h - 5) — end point: same X, 5 pixels from the bottom (small margin so the line doesn't touch the edge)
        if self._state in (State.ALIGN_DIST, State.ALIGN_ANGLE) \
                and self._marker_rvec is not None:
        # Only draw this overlay during the two alignment states, and only if a PnP result exists (marker was detected at least once).
            tv_ov       = self._marker_tvec.flatten()
            R_ov, _     = cv2.Rodrigues(self._marker_rvec)
            # rvec is a compact 3-element rotation vector (axis-angle). 
            # cv2.Rodrigues converts it into a full 3×3 rotation matrix R_ov — 
            # needed to extract the marker's facing direction
            mz_ov       = R_ov @ np.array([0.0, 0.0, 1.0])
            # R_ov rotates from the ArUco local frame -> camera frame
            # [0, 0, 1] is the Z-axis expressed in the marker's local frame
            # Applying R_ov transforms it into camera frame coordinates
            # Result: "where does the marker's Z-axis point, as seen by the camera"
            face_ov     = math.atan2(float(mz_ov[0]), -float(mz_ov[2]))
            # Computes the angle between the marker normal and the camera's optical axis. 
            # mz_ov[0] is the sideways component, -mz_ov[2] is the forward component 
            # (negated because camera Z points into the scene).
            centre_ov   = math.atan2(float(tv_ov[0]), float(tv_ov[2]))
            # tx (index 0) = left/right offset of marker from camera centre
            # ty (index 1) = up/down offset
            # tz (index 2) = forward depth
            # atan2(tx, tz) gives the horizontal angle (yaw) from the camera's forward 
            # axis to the marker — exactly the angle the base needs to rotate to centre the marker.

            # ty is the vertical offset — how high or low the marker is in the image. 
            # The base can't tilt up/down, so vertical misalignment is irrelevant here and ignored.
            d_ov        = float(tv_ov[2])
            # Forward distance from camera to marker in metres (the Z component of tvec)
            if self._state == State.ALIGN_DIST:
                color = (0, 165, 255)  # orange — navigating
                label = (f'ALIGN_DIST  dist_to_goal='
                         f'{math.hypot(self._align_target_world[0] - self._base_x, self._align_target_world[1] - self._base_y):.2f}m'
                         if self._align_target_world is not None else 'ALIGN_DIST')
            else:
                color  = (0, 255, 0) if abs(centre_ov) < ALIGN_ANGLE_TOL else (0, 165, 255)
                label  = (f'ctr={math.degrees(centre_ov):+.1f}deg  '
                          f'face={math.degrees(face_ov):+.1f}deg  '
                          f'dist={d_ov:.3f}m')
            cv2.putText(annotated, label,
                        (cx_img - 420, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        # Detection status banner (top of frame) 
        banner_y = 36
        if ids is None or len(detected_ids) == 0:
            # Nothing at all detected
            cv2.rectangle(annotated, (0, 0), (w, 50), (0, 0, 180), -1)
            cv2.putText(annotated, 'NO MARKERS DETECTED',
                        (10, banner_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        elif target_found:
            # Our marker is visible
            cv2.rectangle(annotated, (0, 0), (w, 50), (0, 140, 0), -1)
            cv2.putText(annotated,
                        f'TARGET ID {ARUCO_MARKER_ID} DETECTED  |  IDs in view: {detected_ids}',
                        (10, banner_y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        else:
            # Markers detected but not our target ID
            cv2.rectangle(annotated, (0, 0), (w, 50), (0, 100, 200), -1)
            cv2.putText(annotated,
                        f'Looking for ID {ARUCO_MARKER_ID}  |  Found IDs: {detected_ids}',
                        (10, banner_y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        # HUD overlays (bottom-left) 
        y = h - 110
        cv2.putText(annotated,
                    f'FSM: {self._state.name}',
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

    # Visualisation timer 
    def _vis_tick(self):
        if self._vis_frame is not None:
            cv2.imshow('ArUco Detection', self._vis_frame)
        cv2.waitKey(1)

    # TF readiness check 
    def _tf_ready(self) -> bool:
        """Returns True once all required TF frames are available."""
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

    # Main control loop (CONTROL_HZ) 
    def _control_loop(self):
        self._log_tick += 1

        if not self._tf_ready():
            return

        # Refresh TF-based robot state
        self._update_vms_state()

        dispatch = {
            State.SEARCH:             self._run_search,
            State.ALIGN_DIST:         self._run_align_dist,
            State.ALIGN_ANGLE:        self._run_align_angle,
            State.APPROACH_BOX_VMS:   self._run_approach_box_vms,
            State.PICK_DESCEND:       self._run_pick_descend,
            State.SUCTION_ON:         self._run_suction_on,
            State.PICK_ASCEND:        self._run_pick_ascend,
            State.NAVIGATE_TO_GOAL:   self._run_navigate_to_goal,
            State.PLACE_VMS_APPROACH: self._run_place_vms_approach,
            State.PLACE_DESCEND:      self._run_place_descend,
            State.SUCTION_OFF:        self._run_suction_off,
            State.PLACE_ASCEND:       self._run_place_ascend,
            State.DONE:               self._run_done,
        }
        dispatch[self._state]()
        self._publish_markers()
        self._terminal_log()

    # Terminal log — printed every 20 ticks (≈1 s at 20 Hz) 
    def _terminal_log(self):
        if self._log_tick % 20 != 0:
            return

        q   = self._arm_q
        ee  = self._last_tf_ee
        pd  = self._path_desired

        # Current target from publish_markers logic (reuse the same logic)
        tgt = None
        if self._state == State.ALIGN_DIST and self._align_target_world is not None:
            tgt = np.array([self._align_target_world[0],
                            self._align_target_world[1], 0.0])
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
        # (State.PLACE_APPROACH removed — was unreachable old state)
        elif self._state == State.NAVIGATE_TO_GOAL:
            nav_z = (self._approach_target[2] + PICK_ASCEND_Z_EXTRA
                     if self._approach_target is not None else NAV_EE_Z)
            tgt = np.array([GOAL_BASE_X, GOAL_BASE_Y + FLOOR_PLACE_FORWARD, nav_z])
        elif self._state in (State.PLACE_VMS_APPROACH, State.PLACE_ASCEND) \
                and self._place_floor_target is not None:
            tgt = self._place_floor_target + np.array([0., 0., PLACE_APPROACH_Z_ABOVE])
        elif self._state in (State.PLACE_DESCEND, State.SUCTION_OFF) \
                and self._place_floor_target is not None:
            tgt = self._place_floor_target  # Z = BOX_HEIGHT + PLACE_TOUCH_Z_OFFSET

        err     = float(np.linalg.norm(tgt - ee)) if (tgt is not None and ee is not None) else float('nan')
        err_wp  = float(np.linalg.norm(pd  - ee)) if (pd  is not None and ee is not None) else float('nan')
        ee_s = f'[{ee[0]:.3f}, {ee[1]:.3f}, {ee[2]:.3f}]' if ee is not None else 'N/A'
        tgt_z_note = ''
        if self._state in (State.PLACE_VMS_APPROACH, State.PLACE_ASCEND) \
                and self._place_floor_target is not None:
            tgt_z_note = f'  (floor+{PLACE_APPROACH_Z_ABOVE:.3f}m approach)'
        elif self._state == State.PLACE_DESCEND and self._place_floor_target is not None:
            tgt_z_note = f'  (BOX_HEIGHT={BOX_HEIGHT:.3f} + PLACE_TOUCH_Z_OFFSET={PLACE_TOUCH_Z_OFFSET:.3f})'
        tgt_s = (f'[{tgt[0]:.3f}, {tgt[1]:.3f}, {tgt[2]:.3f}]{tgt_z_note}'
                 if tgt is not None else 'N/A')
        wp_s  = f'[{pd[0]:.3f}, {pd[1]:.3f}, {pd[2]:.3f}]' if pd is not None else 'N/A'
        box_s = (f'[{self._box_top_world[0]:.3f}, {self._box_top_world[1]:.3f}, '
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
            f'  err_to_tgt  : {err:.4f} m  (EE -> final target)\n'
            f'  err_to_wp   : {err_wp:.4f} m  (EE -> moving waypoint)\n'
            f'  box_top     : {box_s}  locked={self._box_locked}\n'
            f'  active_lims : {lim_active if lim_active else "none"}\n'
            f'{"─"*68}'
        )
        self.get_logger().info(log)

    # FSM: SEARCH 
    def _run_search(self):
        # Check for marker on every tick — even during forward advance
        if self._marker_rvec is not None and self._marker_tvec is not None:
            target = self._compute_align_target()
            if target is not None:
                self._align_target_world = target
                self.get_logger().info(
                    f'Marker detected -> stand-off target {np.round(target, 3)} -> ALIGN_DIST')
                self._send_base(0.0, 0.0)
                self._transition(State.ALIGN_DIST)
                return

        # Forward advance phase (after a full sweep) 
        if self._search_advancing:
            if self._search_advance_start is None:
                self._search_advance_start = time.time()
                self.get_logger().info(
                    f'Search: advancing forward {SEARCH_FWD_DIST:.2f} m')

            elapsed      = time.time() - self._search_advance_start
            advance_time = SEARCH_FWD_DIST / SEARCH_FWD_VEL

            if elapsed < advance_time:
                self._send_base(SEARCH_FWD_VEL, 0.0)
            else:
                # Advance complete — start a fresh sweep from current heading
                self._search_advancing     = False
                self._search_advance_start = None
                self._search_initial_psi   = self._base_psi
                self._search_idx           = 0
                self.get_logger().info('Search: advance done, starting next sweep')
            return

        # Rotation sweep phase 
        if self._search_initial_psi is None:
            self._search_initial_psi = self._base_psi
            self.get_logger().info(
                f'Search sweep started, yaw = {math.degrees(self._base_psi):.1f}°')

        target_psi = (self._search_initial_psi
                      + SEARCH_SEQUENCE[self._search_idx % len(SEARCH_SEQUENCE)])
        psi_err    = _angle_wrap(target_psi - self._base_psi)

        if abs(psi_err) < 0.05:
            # At target heading — hold and check
            if self._search_hold_start is None:
                self._search_hold_start = time.time()
            self._send_base(0.0, 0.0)

            if time.time() - self._search_hold_start >= SEARCH_HOLD_S:
                self._search_idx       += 1
                self._search_hold_start = None
                if self._search_idx >= len(SEARCH_SEQUENCE):
                    # Full sweep done, nothing found — move forward then sweep again
                    self._search_idx      = 0
                    self._search_advancing = True
        else:
            # Rotate toward target heading
            omega = float(np.clip(SEARCH_OMEGA * np.sign(psi_err),
                                  -SEARCH_OMEGA, SEARCH_OMEGA))
            self._search_hold_start = None
            self._send_base(0.0, omega)

    # FSM: ALIGN_DIST 
    def _run_align_dist(self):
        """
        Drive the robot base to the stand-off target computed in SEARCH.
        No angular correction — just navigate to the XY point using a
        heading-first differential-drive P-controller.
        Transitions to ALIGN_ANGLE once within ALIGN_DIST_NAV_TOL.
        """
        if self._align_target_world is None:
            self._transition(State.ALIGN_ANGLE)
            return

        tx, ty = self._align_target_world[0], self._align_target_world[1]
        dx     = tx - self._base_x
        dy     = ty - self._base_y
        dist   = math.hypot(dx, dy)

        if dist < ALIGN_DIST_NAV_TOL:
            self.get_logger().info(
                f'At stand-off point (dist={dist:.3f}m) -> ALIGN_ANGLE')
            self._send_base(0.0, 0.0)
            self._transition(State.ALIGN_ANGLE)
            return

        heading_to_target = math.atan2(dy, dx)
        heading_err       = _angle_wrap(heading_to_target - self._base_psi)

        omega = float(np.clip(ALIGN_DIST_K_HEAD * heading_err,
                              -ALIGN_DIST_MAX_OMEGA, ALIGN_DIST_MAX_OMEGA))
        vx    = 0.0
        if abs(heading_err) < ALIGN_HEAD_TOL:
            vx = float(np.clip(ALIGN_DIST_K_FWD * dist,
                               0.0, ALIGN_DIST_MAX_VX))

        self._send_base(vx, omega)
        self._send_arm(np.zeros(4))

    # FSM: ALIGN_ANGLE 
    def _run_align_angle(self):
        """
        Rotate in place to align the camera face-on to the marker.

        On state entry (first tick): clears stale rvec/tvec so only fresh
        camera detections drive the controller — avoids reacting to data from
        before the robot reached the stand-off point.

        While marker not visible: sweeps ±SEARCH_SWEEP_ANGLE to find it.
        While marker visible:     P-controller on center_angle + face_angle.
        Done when both within ALIGN_ANGLE_TOL.
        """
        # State entry: flush stale PnP data 
        # When the FSM transitions into ALIGN_ANGLE, the most recently stored _marker_tvec/_marker_rvec 
        # could be stale — detected while the robot was still driving during ALIGN_DIST. At that point 
        # the robot was moving, so the pose estimate is for a different robot position/orientation than
        # where it actually stopped. If we use that stale data immediately, the base might rotate based 
        # on an incorrect bearing to the marker.
        if self._align_search_psi is None:
            # The guard if self._align_search_psi is None
            # This only runs once — on the very first tick after the state transition. 
            # _align_search_psi is reset to None in _transition(), so it's None only at entry. 
            # After this block sets it, subsequent ticks skip the flush entirely.
            self._marker_rvec        = None
            self._marker_tvec        = None
            self._marker_detect_time = None
            self._align_search_psi   = self._base_psi   # latch entry heading
            self._align_search_idx   = 0
            self.get_logger().info(
                f'ALIGN_ANGLE entered, flushed stale PnP, '
                f'psi={math.degrees(self._base_psi):.1f}°')

        if self._marker_rvec is not None:
            # Aruco Marker visible — align 
            tv = self._marker_tvec.flatten()
            R, _         = cv2.Rodrigues(self._marker_rvec)
            mz           = R @ np.array([0.0, 0.0, 1.0])
            center_angle = math.atan2(float(tv[0]), float(tv[2]))
            face_angle   = math.atan2(float(mz[0]), -float(mz[2]))  # logged only

            self.get_logger().info(
                f'ALIGN_ANGLE: ctr={math.degrees(center_angle):+.1f}° '
                f'face={math.degrees(face_angle):+.1f}° '
                f'tol={math.degrees(ALIGN_ANGLE_TOL):.1f}°',
                throttle_duration_sec=0.5)

            if abs(center_angle) < ALIGN_ANGLE_TOL:
                self.get_logger().info(
                    f'Centered (ctr={math.degrees(center_angle):+.1f}° '
                    f'face={math.degrees(face_angle):+.1f}°) -> APPROACH_BOX_VMS')
                self._send_base(0.0, 0.0)
                self._transition(State.APPROACH_BOX_VMS)
                return

            # Drive omega from center_angle only — face_angle excluded because
            # at the stand-off point the two terms have opposite signs and cancel.
            omega = float(np.clip(
                -ALIGN_K_CENTER * center_angle,
                -ALIGN_MAX_OMEGA, ALIGN_MAX_OMEGA))
            self._send_base(0.0, omega)
            self._send_arm(np.zeros(4))
            # Reset sweep state so a loss-and-reacquire starts a fresh sweep
            self._align_search_idx  = 0
            self._align_search_hold = None
        else:
            # Aruco Marker not visible — sweep ±SEARCH_SWEEP_ANGLE to find it
            target_psi = (self._align_search_psi
                          + SEARCH_SEQUENCE_ALIGN[self._align_search_idx % len(SEARCH_SEQUENCE_ALIGN)])
            psi_err    = _angle_wrap(target_psi - self._base_psi)

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

    # FSM: APPROACH_BOX_VMS 
    def _run_approach_box_vms(self):
        """
        Single-target weighted VMS: drive EE directly above the box at
        approach height.  Base DOFs are very expensive so the arm handles
        almost all the motion; the base only nudges if the arm truly cannot
        reach alone.  Tune W_BASE_COST to control how much the base moves.
        """
        if self._approach_target is None:
            bx, by, bz = self._box_top_world
            fwd = np.array([math.cos(self._base_psi),
                            math.sin(self._base_psi), 0.0])
            fwd_offset = EE_TOUCH_FORWARD_OFFSET * fwd
            self._approach_target = np.array([bx, by, bz + APPROACH_HEIGHT_ABOVE]) + fwd_offset
            self._box_locked = True
            self.get_logger().info(
                f'Approach target (above box, psi={math.degrees(self._base_psi):.1f}°): '
                f'{np.round(self._approach_target, 3)}')

        W = np.diag([W_BASE_COST, W_BASE_COST,
                     W_ARM_COST,  W_ARM_COST,
                     W_ARM_COST,  W_ARM_COST])
        self._vms_nav_step(self._approach_target, weight_matrix=W)

        if self._at_position(self._approach_target, EE_REACH_TOL):
            self.get_logger().info('Approach reached -> PICK_DESCEND')
            self._transition(State.PICK_DESCEND)

    # FSM: PICK_DESCEND 
    def _run_pick_descend(self):
        bx, by, bz  = self._box_top_world
        fwd = np.array([math.cos(self._base_psi),
                            math.sin(self._base_psi), 0.0])
        fwd_offset = EE_TOUCH_FORWARD_OFFSET * fwd
        pick_target = np.array([bx, by, bz + EE_TOUCH_Z_OFFSET]) + fwd_offset

        # # Arm-only descent (base frozen)
        # self._arm_step(pick_target)

        # Weighted VMS descent — same logic as APPROACH_BOX_VMS, high base cost
        W = np.diag([W_BASE_COST, W_BASE_COST,
                     W_ARM_COST,  W_ARM_COST,
                     W_ARM_COST,  W_ARM_COST])
        self._vms_nav_step(pick_target, weight_matrix=W)

        if self._at_position(pick_target, EE_REACH_TOL):
            self.get_logger().info('Pick position reached -> SUCTION_ON')
            self._transition(State.SUCTION_ON)

    # FSM: SUCTION_ON 
    def _run_suction_on(self):
        if self._state_entry_time is None:
            self._call_suction(True)
            self.get_logger().info('Suction ON — waiting for settle')
        self._send_base(0.0, 0.0)
        self._send_arm(np.zeros(4))
        if self._elapsed() >= SUCTION_SETTLE_S:
            self.get_logger().info('Suction settled -> PICK_ASCEND')
            self._transition(State.PICK_ASCEND)
            # self.get_logger().info('Suction settled → DONE (iterative stop)')
            # self._transition(State.DONE)

    # FSM: PICK_ASCEND 
    def _run_pick_ascend(self):
        """Lift EE to approach position + PICK_ASCEND_Z_EXTRA — weighted VMS."""
        ascend_target = self._approach_target + np.array([0., 0., PICK_ASCEND_Z_EXTRA])
        W = np.diag([W_BASE_COST, W_BASE_COST,
                     W_ARM_COST,  W_ARM_COST,
                     W_ARM_COST,  W_ARM_COST])
        self._vms_nav_step(ascend_target, weight_matrix=W)
        if self._at_position(ascend_target, EE_REACH_TOL):
            self.get_logger().info('Ascended -> NAVIGATE_TO_GOAL')
            self._transition(State.NAVIGATE_TO_GOAL)

    # FSM: NAVIGATE_TO_GOAL 
    def _run_navigate_to_goal(self):
        """
        VMS: drive robot to goal position carrying the box.
        Z is kept the same as the end of PICK_ASCEND (approach_z + PICK_ASCEND_Z_EXTRA)
        so the box stays at a safe height throughout navigation.
        """
        # Clear any stale floor target so PLACE_VMS_APPROACH latches a fresh one.
        if self._path_start is None:
            self._place_floor_target = None

        nav_z = (self._approach_target[2] + PICK_ASCEND_Z_EXTRA
                 if self._approach_target is not None else NAV_EE_Z)
        # EE target is FLOOR_PLACE_FORWARD ahead of GOAL_BASE_Y so the arm's
        # natural forward reach forces the base to actually reach GOAL_BASE_Y.
        # Without this offset the arm extends to cover the gap and the base
        # parks ~0.3m short, making PLACE_VMS_APPROACH target behind the EE.
        vms_goal = np.array([GOAL_BASE_X,
                             GOAL_BASE_Y + FLOOR_PLACE_FORWARD,
                             nav_z])
        self._vms_nav_step(vms_goal)

        if self._at_position(vms_goal, EE_REACH_TOL):
            self.get_logger().info(
                f'Goal reached (EE at goal) -> PLACE_VMS_APPROACH')
            self._transition(State.PLACE_VMS_APPROACH)

    # FSM: PLACE_VMS_APPROACH
    def _run_place_vms_approach(self):
        """
        VMS: drive EE to approach height above the floor drop point.
        _place_floor_target is latched once (fixed world_enu position) so the
        base navigating does not shift the target.
        Orange path line visible via _vms_nav_step.
        """
        if self._place_floor_target is None:
            self._place_floor_target = self._compute_floor_target()
            self.get_logger().info(
                f'PLACE_VMS_APPROACH target latched: '
                f'{np.round(self._place_floor_target, 3)}')

        approach = self._place_floor_target + np.array([0., 0., PLACE_APPROACH_Z_ABOVE])
        W = np.diag([W_BASE_COST, W_BASE_COST,
                     W_ARM_COST,  W_ARM_COST,
                     W_ARM_COST,  W_ARM_COST])
        self._vms_nav_step(approach, weight_matrix=W)

        if self._at_position(approach, EE_REACH_TOL):
            self.get_logger().info(
                f'Above floor drop point (approach={np.round(approach, 3)}) -> PLACE_DESCEND')
            self._transition(State.PLACE_DESCEND)

    # FSM: PLACE_DESCEND
    def _run_place_descend(self):
        """
        Arm-only: descend to PLACE_DESCEND target (BOX_HEIGHT + PLACE_TOUCH_Z_OFFSET).
        Target Z encodes how high the EE is when releasing the box — increase
        PLACE_TOUCH_Z_OFFSET if box is pressed into ground.
        """
        target = self._place_floor_target   # Z = BOX_HEIGHT + PLACE_TOUCH_Z_OFFSET
        if target is None:
            return
        # self._arm_step(target)
        W = np.diag([W_BASE_COST, W_BASE_COST,
                     W_ARM_COST,  W_ARM_COST,
                     W_ARM_COST,  W_ARM_COST])
        self._vms_nav_step(target, weight_matrix=W)
        if self._at_position(target, EE_REACH_TOL):
            self.get_logger().info(
                f'Box at floor level (target={np.round(target, 3)}) -> SUCTION_OFF')
            self._transition(State.SUCTION_OFF)

    # FSM: SUCTION_OFF 
    def _run_suction_off(self):
        """Deactivate suction to release the box, then lift away."""
        if self._state_entry_time is None:
            self._call_suction(False)
            self.get_logger().info('Suction OFF — waiting for release')
        self._send_base(0.0, 0.0)
        self._send_arm(np.zeros(4))
        if self._elapsed() >= SUCTION_SETTLE_S:
            self.get_logger().info('Box released -> PLACE_ASCEND')
            self._transition(State.PLACE_ASCEND)

    # FSM: PLACE_ASCEND 
    def _run_place_ascend(self):
        """
        Arm-only: lift EE back to approach height above floor drop point.
        Mirror of PICK_ASCEND. Orange path line visible via _vms_nav_step.
        """
        if self._place_floor_target is None:
            self._transition(State.DONE)
            return
        approach = self._place_floor_target + np.array([0., 0., PLACE_APPROACH_Z_ABOVE])
        W = np.diag([W_BASE_COST, W_BASE_COST,
                     W_ARM_COST,  W_ARM_COST,
                     W_ARM_COST,  W_ARM_COST])
        self._vms_nav_step(approach, weight_matrix=W)
        if self._at_position(approach, EE_REACH_TOL):
            self.get_logger().info('Lifted from floor -> DONE')
            self._transition(State.DONE)

    # FSM: DONE 
    def _run_done(self):
        self._send_base(0.0, 0.0)
        self._send_arm(np.zeros(4))

    # Weighted task-priority loop 
    def _task_priority_weighted(self, W: np.ndarray) -> np.ndarray:
        """
        Task priority step where:
          - Joint limit tasks  → standard DLS   (highest priority)
          - Position task      → weighted DLS(W) (lower priority, in null-space)

        W is a 6×6 positive-definite diagonal weight matrix.
        Higher W_ii → DOF i is more expensive → optimizer avoids it.
        """
        n    = self._vms_state.getDOF()
        P    = np.eye(n)
        zeta = np.zeros((n, 1))

        # High-priority joint limit tasks — standard DLS
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

        # Position task — weighted DLS (prefers arm joints over base)
        self._pos_task.update(self._vms_state)
        Ji     = self._pos_task.getJacobian()
        xi_dot = self._pos_task.getGain() @ self._pos_task.getError() + self._pos_task.getFF()
        Ji_bar = Ji @ P
        Ji_bar_inv = weighted_DLS(Ji_bar, VMS_DAMPING, W)
        zeta   = zeta + Ji_bar_inv @ (xi_dot - Ji @ zeta)

        return zeta

    # VMS nav step — straight-line path (optional weighted DLS) 
    def _vms_nav_step(self, target_world: np.ndarray, weight_matrix=None):
        """
        Drive EE toward target_world using full VMS with a straight-line
        interpolated path and feedforward velocity.

        weight_matrix : optional 6x6 diagonal np.ndarray.
            If provided, the position task uses weighted_DLS(W) so that
            expensive DOFs (e.g. base) are avoided in favour of cheap ones
            (e.g. arm joints).  None -> standard DLS for all tasks.
        """
        ee = self._last_tf_ee

        if self._path_start is None:
            self._path_start   = (ee.copy() if ee is not None
                                  else target_world.copy())
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


    # Helpers 
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
        self._state_entry_time = None   # set on first tick of new state
        # Reset Bézier path so the next VMS state latches a fresh start
        self._path_start       = None
        self._path_start_t     = None
        self._path_desired     = None
        # self._path_control_pt  = None   # (unused — Bézier)
        # self._path_control_pt2 = None   # (unused — Bézier)
        # NOTE: _place_floor_target is intentionally NOT reset here.
        # It is latched in PLACE_VMS_APPROACH and must persist through
        # PLACE_DESCEND → SUCTION_OFF → PLACE_ASCEND.

    def _elapsed(self) -> float:
        """Seconds since state entry (sets entry time on first call)."""
        if self._state_entry_time is None:
            self._state_entry_time = time.time()
        return time.time() - self._state_entry_time

    def _update_vms_state(self):
        """Refresh all TF-based state fields and VMSRobotState."""
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
            ee_for_jac,
            [self._base_x, self._base_y],
            self._base_psi,
            self._arm_q,
            self._tf_buffer,
        )

    def _tf_pos(self, frame: str):
        """Look up frame origin in world_enu. Returns np.array(3) or None."""
        try:
            t  = self._tf_buffer.lookup_transform(
                WORLD_FRAME, frame, rclpy.time.Time())
            tr = t.transform.translation
            return np.array([tr.x, tr.y, tr.z])
        except (LookupException, ConnectivityException, ExtrapolationException):
            return None

    def _tf_pose(self, frame: str):
        """Look up (x, y, yaw) of frame in world_enu. Returns tuple or None."""
        try:
            t   = self._tf_buffer.lookup_transform(
                WORLD_FRAME, frame, rclpy.time.Time())
            tr  = t.transform.translation
            rot = t.transform.rotation
            siny = 2.0 * (rot.w * rot.z + rot.x * rot.y)
            cosy = 1.0 - 2.0 * (rot.y * rot.y + rot.z * rot.z)
            yaw  = math.atan2(siny, cosy)
            return float(tr.x), float(tr.y), float(yaw)
        except (LookupException, ConnectivityException, ExtrapolationException):
            return None

    def _camera_to_world(self, tvec: np.ndarray, extra_z: float = 0.0):
        """
        Transform a position from the camera optical frame to world_enu via TF.
        extra_z is added to the z component BEFORE the transform
        (useful to shift from marker centre to box top in the camera frame).
        Note: 'extra_z' is added in the camera frame's z direction (optical axis).
        After transform, the resulting world_enu point is returned.
        """
        try:
            pt = PointStamped()
            pt.header.frame_id = CAMERA_FRAME
            pt.header.stamp    = rclpy.time.Time().to_msg()   # time=0 → use latest available
            tv = tvec.flatten()
            pt.point.x = float(tv[0])
            pt.point.y = float(tv[1])
            pt.point.z = float(tv[2])
            # Transform marker centre to world_enu
            pt_world = self._tf_buffer.transform(
                pt, WORLD_FRAME,
                timeout=Duration(seconds=0.05))
            pos = np.array([pt_world.point.x,
                            pt_world.point.y,
                            pt_world.point.z])
            # Shift vertically in world_enu by extra_z
            # (MARKER_TO_BOX_TOP_Z compensates for marker-not-on-top)
            pos[2] += extra_z
            return pos
        except Exception as e:
            self.get_logger().warn(
                f'TF cam→world failed: {e}', throttle_duration_sec=1.0)
            return None

    def _publish_markers(self):
        """
        Publish RViz MarkerArray each control tick:
          RED     sphere — current VMS/arm target
          GREEN/YELLOW sphere — TF EE (yellow=far, green=close)
          CYAN    sphere — VMS FK estimate of EE
          YELLOW  sphere — J1 arm base
          WHITE   line   — error vector EE → target
          WHITE   text   — FSM state + error magnitude
          MAGENTA sphere — detected box top (world_enu)
        """
        now = self.get_clock().now().to_msg()
        ma  = MarkerArray()

        def mk(mid, mtype):
            m = Marker()
            m.header.frame_id    = WORLD_FRAME
            m.header.stamp       = now
            m.ns                 = 'pick_place_vms'
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

        # ── Determine current target from FSM state ───────────────────────────
        target = None

        if self._state == State.SEARCH:
            target = None                                   # no target yet

        elif self._state == State.ALIGN_DIST and self._align_target_world is not None:
            target = np.array([self._align_target_world[0],
                               self._align_target_world[1], 0.0])

        elif self._state == State.ALIGN_ANGLE:
            target = self._box_top_world   # show box location while aligning

        elif self._state == State.APPROACH_BOX_VMS:
            target = self._approach_target   # above-box weighted VMS target

        elif self._state == State.PICK_ASCEND and self._approach_target is not None:
            target = self._approach_target + np.array([0., 0., PICK_ASCEND_Z_EXTRA])

        elif self._state in (State.PICK_DESCEND, State.SUCTION_ON) \
                and self._box_top_world is not None:
            bx, by, bz = self._box_top_world
            target = np.array([bx, by, bz + EE_TOUCH_Z_OFFSET])

        elif self._state == State.NAVIGATE_TO_GOAL:
            nav_z = (self._approach_target[2] + PICK_ASCEND_Z_EXTRA
                     if self._approach_target is not None else NAV_EE_Z)
            target = np.array([GOAL_BASE_X, GOAL_BASE_Y + FLOOR_PLACE_FORWARD, nav_z])

        elif self._state in (State.PLACE_VMS_APPROACH, State.PLACE_ASCEND) \
                and self._place_floor_target is not None:
            target = self._place_floor_target + np.array([0., 0., PLACE_APPROACH_Z_ABOVE])

        elif self._state in (State.PLACE_DESCEND, State.SUCTION_OFF) \
                and self._place_floor_target is not None:
            target = self._place_floor_target

        # (old robot-back states removed from enum — see commented block in State class)

        # RED sphere — current target
        if target is not None:
            ma.markers.append(sphere(ID_TARGET, target, 1.0, 0.0, 0.0, 0.030))

        # GREEN/YELLOW sphere — TF EE (colour encodes error proximity)
        ee = self._last_tf_ee
        if ee is not None:
            err_norm = float(np.linalg.norm(target - ee)) if target is not None else 0.0
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

        # CYAN sphere — VMS FK estimate
        fk_ee = swiftpro_fk_vms_5dof(
            self._link1_world, self._arm_q, self._base_psi, self._tf_buffer)
        if fk_ee is not None:
            ma.markers.append(sphere(ID_FK_EE, fk_ee, 0.0, 0.8, 1.0, 0.015))

        # YELLOW sphere — J1 arm base
        ma.markers.append(sphere(ID_J1_BASE,
                                  np.array([self._base_x, self._base_y, 0.0]),
                                  1.0, 1.0, 0.0, 0.018))

        # WHITE line — error vector EE → target
        if ee is not None and target is not None:
            ml = mk(ID_ERR_LINE, Marker.LINE_STRIP)
            ml.scale.x = 0.004
            ml.color.r = ml.color.g = ml.color.b = 1.0
            ml.color.a = 0.9
            p1 = Point(); p1.x = float(ee[0]);     p1.y = float(ee[1]);     p1.z = float(ee[2])
            p2 = Point(); p2.x = float(target[0]); p2.y = float(target[1]); p2.z = float(target[2])
            ml.points = [p1, p2]
            ma.markers.append(ml)

        # WHITE text — FSM state + relevant error metric
        txt_pos = target if target is not None else np.array([self._base_x, self._base_y, 0.5])
        mt = mk(ID_TEXT, Marker.TEXT_VIEW_FACING)
        mt.pose.position.x = float(txt_pos[0])
        mt.pose.position.y = float(txt_pos[1])
        mt.pose.position.z = float(txt_pos[2]) + 0.07
        mt.scale.z = 0.025
        mt.color.r = mt.color.g = mt.color.b = mt.color.a = 1.0
        if self._state == State.ALIGN_DIST and self._align_target_world is not None:
            d_nav = math.hypot(self._align_target_world[0] - self._base_x,
                               self._align_target_world[1] - self._base_y)
            detail = f'  nav_dist={d_nav:.3f}m / tol={ALIGN_DIST_NAV_TOL:.2f}m'
        elif self._state == State.ALIGN_ANGLE and self._marker_rvec is not None:
            tv_t     = self._marker_tvec.flatten()
            R_t, _   = cv2.Rodrigues(self._marker_rvec)
            mz_t     = R_t @ np.array([0.0, 0.0, 1.0])
            face_t   = math.atan2(float(mz_t[0]), -float(mz_t[2]))
            centre_t = math.atan2(float(tv_t[0]), float(tv_t[2]))
            detail   = (f'  face={math.degrees(face_t):+.1f}deg'
                        f'  ctr={math.degrees(centre_t):+.1f}deg')
        elif ee is not None and target is not None:
            detail = f'  err={np.linalg.norm(target - ee):.3f}m'
        else:
            detail = ''
        mt.text = f'[{self._state.name}]{detail}'
        ma.markers.append(mt)

        # MAGENTA sphere — detected box top
        if self._box_top_world is not None:
            ma.markers.append(sphere(ID_BOX_TOP, self._box_top_world,
                                      1.0, 0.0, 1.0, 0.028))

        # CYAN sphere — ALIGN_DIST stand-off target (world XY, shown at ground level)
        if self._align_target_world is not None:
            at = np.array([self._align_target_world[0],
                           self._align_target_world[1], 0.05])
            ma.markers.append(sphere(ID_ALIGN_TGT, at, 0.0, 1.0, 1.0, 0.035))

        # ORANGE line — path visualisation (VMS states only)
        if self._path_start is not None and target is not None:
            ml = mk(ID_PATH_LINE, Marker.LINE_STRIP)
            ml.scale.x = 0.006
            ml.color.r = 1.0; ml.color.g = 0.5; ml.color.b = 0.0; ml.color.a = 0.8

            # if self._path_control_pt is not None and self._path_control_pt2 is not None:
            #     # Cubic Bézier (_vms_step)
            #     P0, P1, P2, P3 = (self._path_start, self._path_control_pt,
            #                        self._path_control_pt2, target)
            #     for i in range(31):
            #         t  = i / 30.0
            #         pt = _bezier_cubic(t, P0, P1, P2, P3)
            #         p  = Point(); p.x = float(pt[0]); p.y = float(pt[1]); p.z = float(pt[2])
            #         ml.points.append(p)
            # elif self._path_control_pt is not None:
            #     # Quadratic Bézier (_arm_bezier_step)
            #     P0, P1, P2 = self._path_start, self._path_control_pt, target
            #     for i in range(21):
            #         t  = i / 20.0
            #         pt = _bezier(t, P0, P1, P2)
            #         p  = Point(); p.x = float(pt[0]); p.y = float(pt[1]); p.z = float(pt[2])
            #         ml.points.append(p)
            # else:
            #     # Straight-line — two endpoints only
            #     ...

            # Straight-line (_vms_nav_step / _arm_step) — two endpoints only
            ps = Point()
            ps.x = float(self._path_start[0])
            ps.y = float(self._path_start[1])
            ps.z = float(self._path_start[2])
            pe = Point()
            pe.x = float(target[0])
            pe.y = float(target[1])
            pe.z = float(target[2])
            ml.points = [ps, pe]

            ma.markers.append(ml)

        # ORANGE sphere — current interpolated waypoint sliding along the path
        if self._path_desired is not None:
            ma.markers.append(sphere(ID_PATH_WP, self._path_desired,
                                      1.0, 0.5, 0.0, 0.018))

        self._pub_markers.publish(ma)

    def _compute_align_target(self):
        """
        Compute world-frame XY stand-off point directly in front of the marker face.
        Point = marker_world_pos + ALIGN_TARGET_DIST * marker_Z_in_world
        (marker Z points toward the camera, so we follow it back from the marker).
        Returns np.array([x, y]) or None on TF failure.
        """
        if self._marker_rvec is None or self._marker_tvec is None:
            return None
        R, _    = cv2.Rodrigues(self._marker_rvec)
        mz_cam  = R @ np.array([0.0, 0.0, 1.0])   # marker Z in camera frame
        # Stand-off point in camera frame: move along marker Z (toward camera)
        tv      = self._marker_tvec.flatten()
        standoff_cam = tv + ALIGN_TARGET_DIST * mz_cam
        world   = self._camera_to_world(standoff_cam, 0.0)
        if world is None:
            return None
        return world[:2]   # only XY — base navigates on the ground plane

    def _compute_floor_target(self):
        """
        Fixed world-frame EE target for placing the box on the floor.

        XY is defined as (GOAL_BASE_X, GOAL_BASE_Y + FLOOR_PLACE_FORWARD) —
        the same world point the NAVIGATE_TO_GOAL EE target uses — so
        PLACE_VMS_APPROACH never moves the robot backward regardless of
        the heading the robot arrives with.
        """
        return np.array([GOAL_BASE_X,
                         GOAL_BASE_Y + FLOOR_PLACE_FORWARD,
                         BOX_HEIGHT + PLACE_TOUCH_Z_OFFSET])


# ── Utility ───────────────────────────────────────────────────────────────────
def _angle_wrap(a: float) -> float:
    """Wrap angle to (-π, π]."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def main(args=None):
    rclpy.init(args=args)
    node = PickPlaceVMSNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Save diagnostic log FIRST before any cleanup that might fail
        log_path = '/home/haadi/ROS2_Crash_Course/ros2_ws/src/hoi_control/logs/place_approach_log.txt'
        try:
            with open(log_path, 'w') as f:
                f.write('\n'.join(node._log_entries))
            print(f'\nLog saved → {log_path}  ({len(node._log_entries)} entries)')
        except Exception as e:
            print(f'Failed to save log: {e}')
        try:
            node._send_base(0.0, 0.0)
            node._send_arm(np.zeros(4))
        except Exception:
            pass
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()    # # ── Arm-only cubic Bézier step — curved trajectory, base frozen ──────────
    # def _arm_bezier_step(self, target_world: np.ndarray):
    # """
    # Follow a cubic Bézier curved path to target_world using arm joints only.
    # Same control point geometry as _vms_step (VMS_CURVE_LATERAL for outward
    # swing, VMS_CURVE_HEIGHT for lift) but base velocity is zeroed — only
    # arm FK/Jacobian drives the motion.

    # Used for PLACE_APPROACH so the arm arcs outward from the pick position
    # to the drop position without colliding with the robot body.
    # Visualization: orange cubic curve + sliding orange sphere (automatic).
    # """
    # ee = self._last_tf_ee
    # if ee is None:
    # self._send_base(0.0, 0.0)
    # self._send_arm(np.zeros(4))
    # return

    # # Latch path start and control points on first call
    # if self._path_start is None:
    # self._path_start   = ee.copy()
    # self._path_start_t = self.get_clock().now()

    # P0, P3 = self._path_start, target_world
    # path_xy  = P3[:2] - P0[:2]
    # path_len = float(np.linalg.norm(path_xy))
    # if path_len > 0.01:
    # perp_xy = np.array([-path_xy[1], path_xy[0]]) / path_len
    # fwd_xy  = path_xy / path_len
    # else:
    # perp_xy = np.zeros(2)
    # fwd_xy  = np.zeros(2)

    # lat  = -VMS_CURVE_LATERAL   # opposite side from the VMS approach arc
    # lift = VMS_CURVE_HEIGHT

    # # Single quadratic control point at path midpoint, pulled sideways + lifted
    # midpoint = (P0 + P3) / 2.0
    # self._path_control_pt  = midpoint + np.array([
    # lat * perp_xy[0], lat * perp_xy[1], lift])
    # self._path_control_pt2 = None   # quadratic — no second control point

    # P0 = self._path_start
    # P1 = self._path_control_pt
    # P2 = target_world

    # elapsed      = (self.get_clock().now() - self._path_start_t).nanoseconds / 1e9
    # alpha        = float(np.clip(elapsed / PLACE_APPROACH_PATH_PERIOD, 0.0, 1.0))
    # path_desired = _bezier(alpha, P0, P1, P2)
    # self._path_desired = path_desired

    # ff_vel = (_bezier_vel(alpha, P0, P1, P2) / PLACE_APPROACH_PATH_PERIOD
    # if alpha < 1.0 else np.zeros(3))

    # self._pos_task.setDesired(path_desired.reshape(3, 1))
    # self._pos_task.setGain(np.eye(3) * VMS_K)
    # self._pos_task.setFF(ff_vel.reshape(3, 1))

    # tasks = self._joint_limit_tasks + [self._pos_task]
    # zeta  = vms_task_priority_step(
    # tasks, self._vms_state, damping=VMS_DAMPING, method=2).flatten()

    # # Zero base — arm joints only
    # dq = np.clip(zeta[2:6], -ARM_MAX_VEL, ARM_MAX_VEL)
    # self._send_base(0.0, 0.0)
    # self._send_arm(dq)


if __name__ == '__main__':
    main()
