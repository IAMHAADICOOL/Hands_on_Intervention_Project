#!/usr/bin/env python3
import math
import datetime
import os

import rclpy
from rclpy.node import Node           
from rclpy.parameter import Parameter 
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, Float64
from geometry_msgs.msg import Twist, Point
from visualization_msgs.msg import Marker, MarkerArray
import numpy as np
from tf2_ros import Buffer, TransformListener, LookupException, \
    ConnectivityException, ExtrapolationException
from hoi_control.swiftpro_robotics_rrc import (
    # SwiftProManipulator4DOF,      # 4-DOF arm state + kinematics queries
    # swiftpro_fk,                  # Arm-local FK: [q1, q2, q3] -> position relative to link1
    swiftpro_fk_vms_5dof,         # 5-DOF VMS FK: base (x,y,psi) + arm (q1,q2,q3)
    swiftpro_jacobian_vms_5dof,   # 5-DOF VMS Jacobian: 3x5 matrix (base + arm DOF)
    swiftpro_jacobian_vms_6dof,   # 6-DOF VMS Jacobian: 4x6 matrix (pos + yaw, includes q4)
    # swiftpro_fk_vms_5dof_ned,     # FK variant for world_ned frame 
    # swiftpro_jacobian_vms_5dof_ned, # Jacobian variant for world_ned frame 
    # swiftpro_fk_vms_5dof_ned,     # 5-DOF VMS FK (world_ned -> world_enu via TF) 
    # swiftpro_jacobian_vms_5dof_ned,   # 5-DOF VMS Jacobian (world_ned -> world_enu via TF) 
    # DLS,                          # Damped Least-Squares pseudo-inverse
    scale_velocities,             # uniform velocity scaling to stay within max_vel
    Q1_MIN, Q1_MAX,               # joint1 base yaw URDF limits  (+-pi/2)
    Q2_MIN, Q2_MAX,               # joint2 shoulder URDF limits  (-pi/2, +0.05)
    Q3_MIN, Q3_MAX,               # joint3 elbow    URDF limits  (-pi/2, +0.05)
    Q4_MIN, Q4_MAX,               # joint4 EE yaw   URDF limits  (+-pi/2)
    # Task-priority infrastructure
    VMSRobotState,
    VMSPositionTask,
    VMSConfigurationTask,
    # VMSYawQ4Task,         # replaced by VMSConfigurationTask for orient goals
    # VMSJointCenteringTask,
    VMSJointLimitsTask,
    VMSJointPositionTask,
    VMSBaseOrientationTask,
    vms_task_priority_step,
)

# ---------------------------------------------------------------------------
# Topics and TF frames
# ---------------------------------------------------------------------------
# The joint velocity controller listens on this topic; it expects a
# Float64MultiArray with one value per active joint: [dq1, dq2, dq3, dq4]
JOINT_CMD_TOPIC   = '/turtlebot/swiftpro/joint_velocity_controller/command'

# The simulator publishes all joint positions and velocities on this topic
JOINT_STATE_TOPIC = '/turtlebot/joint_states'

# Mobile base velocity controller (TurtleBot differential drive)
BASE_CMD_TOPIC    = '/turtlebot/cmd_vel'  # Twist message: linear.x, angular.z 

# marker topic 
MARKER_TOPIC      = '/hoi/rrc_methods_vms_markers'

# TF frame in which all world-frame positions are expressed (ENU = East-North-Up)
WORLD_FRAME       = 'world_enu'

# TF frame of the physical end-effector (published by the static_transform_publisher in the launch file)
EE_FRAME          = 'end_effector'

# TF frame of the arm's first joint / base (link1); used to convert between
# world_enu and arm-local ENU by subtracting J1's world position
J1_FRAME          = 'turtlebot/swiftpro/manipulator_base_link'

# Control loop period: 60 Hz matches the simulator physics timestep
DT = 1.0 / 60.0   # seconds per control tick


# Marker IDs — each ID corresponds to a persistent RViz marker object.
# Using fixed IDs means successive publishes UPDATE the marker in place
# rather than creating a new one each tick.
ID_TARGET    = 0   # RED    sphere  — desired target position
ID_TF_EE     = 1   # GREEN  sphere  — true EE from TF (colour interpolates to yellow as error grows)
ID_FK_EE     = 2   # CYAN   sphere  — FK estimate of EE (shows FK accuracy vs TF)
ID_J1_BASE   = 3   # YELLOW sphere  — J1 arm base (origin of arm-local ENU)
ID_ERR_LINE  = 4   # WHITE  line    — vector from current EE to target (visual error)
ID_TEXT      = 5   # WHITE  text    — method name, error magnitude, and goal index
ID_PATH_LINE = 6   # ORANGE line    — straight-line path from start to goal (full intended path)
ID_PATH_WP   = 7   # ORANGE sphere  — current interpolated waypoint being tracked


# Automatic goal cycling (robot-relative, not arm-relative)
# GOAL_SEQUENCE is a flat list with two kinds of entries that can appear in any order:
#
#   Position entry  (original format, unchanged):
#       (np.array([x, y, z]), yaw)
#       yaw = float('nan') -> position-only sub-phase
#       yaw = finite float -> position sub-phase then yaw sub-phase (decoupled)
#
#   Joint-position entry (one per joint, placed anywhere in the list):
#       ('joint', arm_joint_idx, desired_angle_rad)
#       arm_joint_idx : 0=q1, 1=q2, 2=q3, 3=q4
#
#   Base-orientation entry (one, placed anywhere in the list):
#       ('base_yaw', desired_heading_rad)
#       Controls base yaw (sigh) via the (omega) quasi-velocity DOF (index 1 in dzeta).
#       Internally stored with key -1 in the same persistent dicts as joint tasks.
#
# Scanning rule (computed once at startup by _parse_goal_sequence):
#   An entry that appears BEFORE its adjacent position entry -> ABOVE
#     (higher priority than EE position — position acts in its null-space)
#   An entry that appears AFTER  its adjacent position entry -> BELOW
#     (lower priority — runs in null-space of EE position)
#
# Task hierarchy for the position sub-phase of each goal:
#   joint_limits  >  joint tasks ABOVE  >  EE position  >  joint tasks BELOW
#
# Joint tasks are NOT applied during the yaw sub-phase (sub==1) — preserving
# the position/yaw decoupling of configuration goals.
GOAL_SEQUENCE = [
    # Goal 0 — position-only, no joint tasks
    # ('base_yaw', 0.0), # (sigh) -> 0.0 rad    [Below]
    (np.array([ 1.20,  0.00,  0.30]), float('nan')),

    # Goal 1 — q2 (shoulder) pinned ABOVE the position task
    # Q2_MAX=0.05, margin=0.07 -> safe upper bound = 0.05−0.07×1.5 = −0.055
    # ('joint', 1, -0.10),                                  # q2->-0.10  [ABOVE]
    (np.array([ 0.00,  1.20,  0.35]),  0.5),              # orient goal

    # Goal 2 — q1 biased BELOW position; base yaw biased toward 0.0 (face East) in null-space
    (np.array([ 1.00,  0.60,  0.28]), float('nan')),      # position-only
    # ('joint',    0,  0.3),                                # q1->0.3       [BELOW]
                                       

    # Goal 3 — orient, no joint tasks
    (np.array([ 0.00, -1.20,  0.30]), -0.5),

    # Goal 4 — q3 (elbow) fixed ABOVE position; q1 biased BELOW
    # Q3_MAX=0.05, margin=0.07 -> safe upper bound = 0.05−0.07×1.5 = −0.055
    # ('joint', 2, -0.10),                                  # q3->-0.10  [ABOVE]
    (np.array([ 0.80,  0.80,  0.36]),  0.0),              # orient goal
    ('joint', 0, 0.5),                                    # q1->0.5  [BELOW]

    # Goal 5 — orient, no joint tasks
    (np.array([ 1.10, -0.40,  0.28]),  0.8),
]


def _parse_goal_sequence(seq):
    """
    Scan GOAL_SEQUENCE once at startup and return a list of parsed goal tuples:
        (position, yaw, above_specs, below_specs)

    position   : np.array([x, y, z])
    yaw        : float or nan
    above_specs: list of (idx, desired_angle) — tasks above EE position
    below_specs: list of (idx, desired_angle) — tasks below EE position

    idx encoding:
       0-3  : arm joint index (q1-q4) from ('joint', idx, angle) entries
        -1  : base orientation       from ('base_yaw', angle) entries

    Algorithm:
      1. Find indices of all position entries (first element is not a task tag).
      2. For each position entry at flat index i (goal g):
           above = task entries between (prev_pos_idx + 1) and i  (i exclusive)
           below = task entries between i and (next_pos_idx)      (i exclusive)
    """
    _TASK_TAGS = ('joint', 'base_yaw') # Any tuple whose first element is one of these strings is a task, not a position goal.
    is_pos = lambda e: not (isinstance(e[0], str) and e[0] in _TASK_TAGS)
    pos_indices = [i for i, e in enumerate(seq) if is_pos(e)]

    def _to_spec(e):
        # For creating the data in a format that will help us to define the tasks
        """Convert a task entry to (idx, angle)."""
        if e[0] == 'joint':
            return (e[1], e[2])    # (arm_joint_idx, desired_angle)
        elif e[0] == 'base_yaw':
            return (-1, e[1])      # (-1 = base orientation, desired_heading)

    goals = []
    for g, pi in enumerate(pos_indices):
        position, yaw = seq[pi][0], seq[pi][1]

        prev_pi = pos_indices[g - 1] if g > 0 else -1
        next_pi = pos_indices[g + 1] if g < len(pos_indices) - 1 else len(seq)

        above = [_to_spec(e) for e in seq[prev_pi + 1 : pi] if not is_pos(e)]
        below = [_to_spec(e) for e in seq[pi + 1 : next_pi] if not is_pos(e)]

        goals.append((position, yaw, above, below))
        # For goal 4 which has both above and below:
        # (np.array([0.80, 0.80, 0.36]),   # position
            # 0.0,                            # yaw
            # [(2, -0.10)],                   # above  — q3 pinned to -0.10
            # [(0,  0.5)]                     # below  — q1 biased to 0.5
            # )

    return goals

# How long (seconds) the arm tries to reach each goal before cycling to the next
DEFAULT_GOAL_CYCLE_PERIOD = 40.0

# Safety margin applied inside the URDF hard limits when clamping velocities.
# Stops the joint slightly before the hard limit to give the controller time
# to decelerate
LIMIT_MARGIN = 0.07   # radians — activation threshold α
LIMIT_HYSTERESIS_RATIO = 1.5  # δ = LIMIT_MARGIN * ratio; must be > 1 to avoid chatter

# Weighted DLS cost matrix — penalises base DOFs (vx, ω) more than arm joints (dq1-dq4).
# Higher cost → that DOF is used less by the optimiser → arm preferred over base motion.
W_BASE_COST = 10.0   # cost for vx and ω
W_ARM_COST  = 1.0    # cost for dq1–dq4
VMS_WEIGHT_MATRIX = np.diag([W_BASE_COST, W_BASE_COST,
                              W_ARM_COST,  W_ARM_COST,
                              W_ARM_COST,  W_ARM_COST])


class Lab2RRCMethodsVMSNode(Node):

    def __init__(self):
        super().__init__('lab2_rrc_methods_vms_config')

        # Declare runtime-tunable parameters 

        # Parse the flat GOAL_SEQUENCE (which may contain 'joint' entries mixed with
        # position entries) into a structured list of goal tuples once at startup.
        # self._goals[g] = (position, yaw, above_specs, below_specs)
        # _parse_goal_sequence skips 'joint' entries and finds the first actual
        # position entry, so self._goals[0] is always a valid position goal.
        self._goals = _parse_goal_sequence(GOAL_SEQUENCE)

        # Goal position in world_enu frame.
        # Use the first position entry from the parsed goals
        _g0_pos, _g0_yaw = self._goals[0][0], self._goals[0][1]
        self.declare_parameter('target_x',   float(_g0_pos[0]))
        self.declare_parameter('target_y',   float(_g0_pos[1]))
        self.declare_parameter('target_z',   float(_g0_pos[2]))

        # Desired end-effector yaw angle (radians) for the suction cup joint (q4).
        # NaN   -> position-only mode: q4 is not controlled.
        # float -> q4 is driven to this angle after the position sub-task completes.
        self.declare_parameter('target_yaw', float(_g0_yaw))

        # Proportional gain K for the position task: larger = faster convergence,
        self.declare_parameter('gain_k',    0.5)

        # Proportional gain for the q4 yaw sub-task. Kept lower than gain_k because yaw correction is a secondary objective.
        self.declare_parameter('yaw_gain_k', 0.3)

        # Maximum arm joint speed after uniform scaling (rad/s).
        # All joint velocities are clipped so the largest does not exceed this.
        self.declare_parameter('max_vel',   0.3)

        # DLS damping factor lambda. Larger -> smoother motion near singularities
        # but larger steady-state position error. Smaller -> more accurate but
        # can produce large velocities at singular configurations.
        self.declare_parameter('damping',   0.05)

        # Inverse kinematics method selector:
        #   0 = Jacobian transpose  (fast, no matrix inversion, may not converge)
        #   1 = Pseudo-inverse      (exact at non-singular configs, unstable near singularities)
        #   2 = DLS (Damped Least Squares) — recommended, stable near singularities
        self.declare_parameter('method',    2)

        # Master enable switch. False -> publish zero velocities, arm holds still.
        self.declare_parameter('enabled',   True)

        # Goal cycling: auto-advance through GOAL_SEQUENCE every goal_cycle_period seconds.
        self.declare_parameter('goal_auto_cycle',   True)
        # Time (s) each sub-task runs before the timer fires and advances to the next.
        self.declare_parameter('goal_cycle_period', DEFAULT_GOAL_CYCLE_PERIOD)

        # Maximum base linear speed (m/s). Lower values prevent orbit by reducing
        # the step size of each DLS update relative to the workspace curvature.
        self.declare_parameter('base_linear_max',  0.3)

        # Maximum base angular speed (rad/s). Keep proportional to base_linear_max
        # to avoid generating circular trajectories (orbit) near goals.
        self.declare_parameter('base_angular_max', 0.5)

        # Position error below which the base stops moving (m).
        # Prevents micro-oscillations when the EE is very close to the target.
        self.declare_parameter('error_stop',       0.03)

        # Yaw error below which the yaw sub-task is considered complete (rad).
        self.declare_parameter('yaw_stop',         0.08)
        
        # The above two parameters are not used here.

        # Task-priority infrastructure — state container + task objects.
        # Joint limit tasks have highest priority (indices 2-5 = q1-q4 in dzeta).
        # Position task is next; yaw task is lowest priority (null-space of position).
        self._vms_state = VMSRobotState()
        self._joint_limit_tasks = [
            VMSJointLimitsTask('q1_limits', 2, Q1_MIN, Q1_MAX, margin=LIMIT_MARGIN, hysteresis_ratio=LIMIT_HYSTERESIS_RATIO),
            VMSJointLimitsTask('q2_limits', 3, Q2_MIN, Q2_MAX, margin=LIMIT_MARGIN, hysteresis_ratio=LIMIT_HYSTERESIS_RATIO),
            VMSJointLimitsTask('q3_limits', 4, Q3_MIN, Q3_MAX, margin=LIMIT_MARGIN, hysteresis_ratio=LIMIT_HYSTERESIS_RATIO),
            VMSJointLimitsTask('q4_limits', 5, Q4_MIN, Q4_MAX, margin=LIMIT_MARGIN, hysteresis_ratio=LIMIT_HYSTERESIS_RATIO),
        ]
        self._pos_task    = VMSPositionTask('ee_position', np.zeros((3, 1)))
        self._config_task = VMSConfigurationTask('ee_config', np.zeros((4, 1)))  # pos+yaw, 4×6 J

        # Persistent joint-position task state — survives across goal transitions.
        # Keyed by arm_joint_idx (0=q1 .. 3=q4); value = desired angle (rad).
        # New goals MERGE into these dicts (add / override); joints not mentioned
        # in a new goal keep their previous desired angle and priority side.
        # If a goal moves a joint from 'above' to 'below' (or vice versa), the
        # old entry is removed from its previous side automatically.
        self._persist_above: dict = {}   # {arm_joint_idx: desired_angle}
        self._persist_below: dict = {}   # {arm_joint_idx: desired_angle}

        # Cached VMSJointPositionTask lists rebuilt from the persistent dicts
        # whenever the goal index changes.
        self._joint_tasks_above: list = []
        self._joint_tasks_below: list = []
        self._cached_goal_idx = -1   # sentinel: forces rebuild on first tick
        
        # READ THE explanation_to_some_complex_ideas.txt TO UNDERSTAND THE ABOVE PERSISTENCE LOGIC AND GOAL SCANNING ALGORITHM — IT'S NOT OBVIOUS!

        # Guard flag: skip the control loop until the first /joint_states message
        # arrives so we never compute with stale zero-initialised joint angles
        self._got_js = False

        # Cached robot base pose in world_enu frame (extracted from TF: base_footprint)
        # Used by 5-DOF VMS FK/Jacobian
        self._base_x = 0.0       # robot base x position (East, m)
        self._base_y = 0.0       # robot base y position (North, m)
        self._base_psi = 0.0     # robot base yaw angle (rad)
        # self._omega_lpf = 0.0    # low-pass filter state for base angular velocity

        # Initial base yaw — latched on first TF reading.
        # swiftpro_fk() is calibrated for the scenario's initial robot yaw.
        # delta_yaw = base_psi - initial_base_psi corrects for any base rotation.
        # self._initial_base_psi = None   # no longer needed — FK uses q1_mod = q1-(ψ-π/2)
        
        # Cached arm link1 position in world_enu (used for FK offset)
        self._link1_world_enu = np.array([0.0, 0.0, 0.0])

        # Most recent TF-based EE position (world_enu); retained between ticks
        # so the debug log can show it even when a TF lookup temporarily fails
        self._last_tf_ee = None

        # Incrementing counter used to throttle the verbose terminal log:
        # we only print every 30 ticks (~0.5 s) to avoid terminal flooding
        self._loop_count = 0

        # Index into GOAL_SEQUENCE pointing to the currently active goal.
        # Starts at 0 (first goal) and wraps around after the last goal.
        self._goal_idx = 0

        # A dedicated ROS timer fires _advance_goal() every DEFAULT_GOAL_CYCLE_PERIOD
        # seconds, independently of the control-loop rate.  
        self._goal_cycle_timer = self.create_timer(DEFAULT_GOAL_CYCLE_PERIOD, self._advance_goal)

        # No sub-phase index — configuration task handles pos+yaw simultaneously.

        # Straight-line path tracking for the position sub-task.
        # _path_start  : EE position (world_enu) when the position sub-task began.
        # _path_start_t: ROS time when the position sub-task began.
        # The desired position is interpolated linearly from _path_start to the
        # goal position over goal_cycle_period seconds, so the EE follows a
        # straight line in task space rather than curving directly toward the goal.
        self._path_start   = None   # (3,) ndarray or None
        self._path_start_t = None   # rclpy.time.Time or None

        # Accumulated log entries — written to file on Ctrl+C.
        self._log_entries = []

        # Buffer stores a sliding window of recent transforms from /tf and /tf_static
        self._tf_buffer = Buffer()

        self._tf_listener = TransformListener(self._tf_buffer, self)

        
        self.sub_js = self.create_subscription(JointState, JOINT_STATE_TOPIC, self._js_cb, 10)

        # Publisher for joint velocity commands; the controller reads this at ~60 Hz
        # TODO What does the above statement mean?
        self.pub_cmd = self.create_publisher(Float64MultiArray, JOINT_CMD_TOPIC, 10)

        # Publisher for mobile base velocity commands (Twist: linear.x, angular.z)
        self.pub_base_cmd = self.create_publisher(Twist, BASE_CMD_TOPIC, 10)

        # Publisher for RViz debug markers (spheres, lines, text)
        self.pub_markers = self.create_publisher(MarkerArray, MARKER_TOPIC, 10)

        # Publishers for joint angles (PlotJuggler visualization)
        self.pub_q1 = self.create_publisher(Float64, '/hoi/joint_angle_q1', 10)
        self.pub_q2 = self.create_publisher(Float64, '/hoi/joint_angle_q2', 10)
        self.pub_q3 = self.create_publisher(Float64, '/hoi/joint_angle_q3', 10)
        self.pub_q4 = self.create_publisher(Float64, '/hoi/joint_angle_q4', 10)

        # Publishers for joint limit status (1 = active, 0 = inactive)
        self.pub_q1_limit = self.create_publisher(Float64, '/hoi/joint_limit_q1', 10)
        self.pub_q2_limit = self.create_publisher(Float64, '/hoi/joint_limit_q2', 10)
        self.pub_q3_limit = self.create_publisher(Float64, '/hoi/joint_limit_q3', 10)
        self.pub_q4_limit = self.create_publisher(Float64, '/hoi/joint_limit_q4', 10)

        # Publishers for task error norms (PlotJuggler visualization)
        self.pub_err_to_goal = self.create_publisher(Float64, '/hoi/err_to_goal', 10)
        self.pub_err_to_wp   = self.create_publisher(Float64, '/hoi/err_to_waypoint', 10)

        # Publisher for base yaw (PlotJuggler visualization)
        self.pub_base_yaw = self.create_publisher(Float64, '/hoi/base_yaw', 10)
        
        # Main control timer: fires _control_loop at exactly 60 Hz (every DT seconds)
        self.timer = self.create_timer(DT, self._control_loop)

        # Startup log 
        self.get_logger().info('=' * 62)
        self.get_logger().info('Lab2 RRC Methods VMS node started (4-DOF on Turtlebot)')
        self.get_logger().info('Tune: ros2 param set /lab2_rrc_methods_vms <param> <value>')
        self.get_logger().info('Arm params:  target_x/y/z, target_yaw, gain_k, max_vel, '
                               'damping, method, enabled, goal_auto_cycle, goal_cycle_period')
        self.get_logger().info('Base params: base_vel_enabled, base_linear_max, base_angular_max, '
                               'error_threshold')
        self.get_logger().info('Publishing: arm commands -> /turtlebot/.../joint_velocity_controller/command')
        self.get_logger().info('Publishing: base commands -> /turtlebot/cmd_vel  (namespaced!)')
        self.get_logger().info('RViz:  Fixed Frame=world_enu  '
                               'MarkerArray from ' + MARKER_TOPIC)
        self.get_logger().info(
            f'Goals: {len(self._goals)} DISTANT targets in world_enu, cycling every '
            f'{DEFAULT_GOAL_CYCLE_PERIOD:.0f} s  (goal_auto_cycle=True)')
        self.get_logger().info('FK/Jacobian: Arm-local (proven), errors in world_enu frame')
        self.get_logger().info('Method: DLS (default=2), World EE = J1_world + FK_arm_local')
        self.get_logger().info('Base moves when error > error_threshold using base_linear_max')
        self.get_logger().info('=' * 62)

    # Joint task cache 

    def _build_joint_tasks(self, goal_idx):
        """
        Merge this goal's joint-position specs into the persistent dicts and
        rebuild the VMSJointPositionTask lists.  Called only on goal change.

        Persistence rule:
          • Joints defined in this goal are added / updated in the persistent dict.
          • If a joint switches sides (above <--> below), it is removed from the old side.
          • Joints NOT mentioned in this goal keep their previous desired angle and side.

        This means a joint task introduced in goal N continues to apply in goals
        N+1, N+2, ... (with its last desired angle) until a later goal explicitly
        changes or cancels it.
        """
        _, _, above_specs, below_specs = self._goals[goal_idx]

        # Merge 'above' specs — remove from below if switching sides
        for qi, angle in above_specs:
            self._persist_below.pop(qi, None)
            self._persist_above[qi] = angle

        # Merge 'below' specs — remove from above if switching sides
        for qi, angle in below_specs:
            self._persist_above.pop(qi, None)
            self._persist_below[qi] = angle

        # Rebuild task objects from the now-updated persistent dicts.
        # idx == -1  -> VMSBaseOrientationTask (controls base (sigh) via (omega) DOF)
        # idx == 0-3 -> VMSJointPositionTask   (controls arm joint q_{idx+1})
        def _make_task(side, qi, angle):
            if qi == -1:
                return VMSBaseOrientationTask(f'base_yaw_{side}', angle)
            return VMSJointPositionTask(f'jp_{side}_q{qi+1}', angle, qi)

        def _label(qi, angle):
            return f'(sigh)->{angle:.3f}rad' if qi == -1 else f'q{qi+1}→{angle:.3f}rad'

        self._joint_tasks_above = [
            _make_task('above', qi, angle) for qi, angle in self._persist_above.items()
        ]
        self._joint_tasks_below = [
            _make_task('below', qi, angle) for qi, angle in self._persist_below.items()
        ]
        self._cached_goal_idx = goal_idx

        above_str = [_label(qi, a) for qi, a in self._persist_above.items()]
        below_str = [_label(qi, a) for qi, a in self._persist_below.items()]
        self.get_logger().info(
            f'[JOINT TASKS] goal {goal_idx} — persistent state: '
            f'above={above_str}  below={below_str}')

    # Callbacks
    def _js_cb(self, msg: JointState):
        """Update arm joint angles from the latest joint_states message."""
        self._vms_state.update_from_joint_states(list(msg.name), list(msg.position))

        # Signal that at least one valid joint-state has arrived; the control
        # loop blocks until this is True
        self._got_js = True

    # For robot ee
    def _tf_pos(self, child_frame):
        """Look up position of child_frame in world_enu. Returns (3,) or None."""
        try:
            t = self._tf_buffer.lookup_transform(
                WORLD_FRAME, child_frame, rclpy.time.Time())
            tr = t.transform.translation
            return np.array([tr.x, tr.y, tr.z])
        except (LookupException, ConnectivityException, ExtrapolationException):
            return None
    
    # For mobile base pose (x, y, yaw)
    def _tf_pose(self, frame_id):
        """Look up pose (position + yaw) of frame_id in world_enu. Returns [x, y, psi] or None."""
        try:
            t = self._tf_buffer.lookup_transform(
                WORLD_FRAME, frame_id, rclpy.time.Time())
            tr = t.transform.translation
            quat = t.transform.rotation
            qx, qy, qz, qw = quat.x, quat.y, quat.z, quat.w
            yaw = np.arctan2(2.0 * (qw*qz + qx*qy), 1.0 - 2.0 * (qy*qy + qz*qz))
            return np.array([tr.x, tr.y, yaw])
        except (LookupException, ConnectivityException, ExtrapolationException):
            return None

    # Goal cycling 

    def _advance_goal(self):
        """
        Timer callback: rotate to the next goal in GOAL_SEQUENCE every
        `goal_cycle_period` seconds (when goal_auto_cycle is enabled).

        The new target is written back into the ROS parameters so that
        external observers (ros2 param get / dynamic_reconfigure) see the
        current goal, and so the control loop reads them the same way it
        reads manually-set parameters.
        """
        # If cycling is disabled at runtime, fire but do nothing
        if not self.get_parameter('goal_auto_cycle').value:
            return

        # Re-read the period in case it was changed at runtime via ros2 param set.
        # timer_period_ns is the period this timer was created with (nanoseconds).
        period = float(self.get_parameter('goal_cycle_period').value)
        if abs(self._goal_cycle_timer.timer_period_ns / 1e9 - period) > 0.5:
            # Period was changed by more than 0.5 s — recreate the timer.
            # ROS 2 timers cannot be reconfigured after creation, so we destroy
            # the old one and create a fresh timer with the new period.
            self._goal_cycle_timer.destroy()
            self._goal_cycle_timer = self.create_timer(period, self._advance_goal)

        # Configuration task: pos+yaw simultaneously — advance goal directly each timer fire.
        self._path_start   = self._last_tf_ee.copy() if self._last_tf_ee is not None else None
        self._path_start_t = self.get_clock().now()
        self._goal_idx = (self._goal_idx + 1) % len(self._goals)

        goal_pos, goal_yaw = self._goals[self._goal_idx][0], self._goals[self._goal_idx][1]

        # Write the new target back into the ROS parameter server.
        self.set_parameters([
            Parameter('target_x',   Parameter.Type.DOUBLE, float(goal_pos[0])),
            Parameter('target_y',   Parameter.Type.DOUBLE, float(goal_pos[1])),
            Parameter('target_z',   Parameter.Type.DOUBLE, float(goal_pos[2])),
            Parameter('target_yaw', Parameter.Type.DOUBLE, float(goal_yaw)),
        ])

        yaw_str = f'{goal_yaw:.2f} rad ({math.degrees(goal_yaw):.1f} deg)' \
                  if math.isfinite(goal_yaw) else 'position-only'
        msg = (f'[GOAL CYCLE] -> goal {self._goal_idx}/{len(self._goals)-1}: '
               f'[{goal_pos[0]:.3f}, {goal_pos[1]:.3f}, {goal_pos[2]:.3f}] world_enu  '
               f'yaw={yaw_str}')
        self.get_logger().info(msg)
        self._log_entries.append(msg)

    def _control_loop(self):
        # Block until the first /joint_states message has been received.
        if not self._got_js:
            return

        # Increment tick counter; used below to throttle the terminal log
        self._loop_count += 1

        # Read runtime parameters 
        # All parameters are re-read every tick so that ros2 param set changes
        # take effect within one control cycle (~16 ms at 60 Hz).
        tx     = self.get_parameter('target_x').value   
        ty     = self.get_parameter('target_y').value    
        tz     = self.get_parameter('target_z').value    
        t_yaw  = self.get_parameter('target_yaw').value  # desired EE yaw (rad); NaN = position-only
        K_gain = self.get_parameter('gain_k').value      # scalar proportional gain applied to error
        mv          = self.get_parameter('max_vel').value          # joint velocity ceiling (rad/s)
        dam         = self.get_parameter('damping').value          # DLS damping factor lambda
        mth         = int(self.get_parameter('method').value)      # integer selector: 0/1/2
        ena         = self.get_parameter('enabled').value          # bool: if False, publish zeros
        base_lin_max = float(self.get_parameter('base_linear_max').value)   # m/s
        base_ang_max = float(self.get_parameter('base_angular_max').value)  # rad/s

        
        target_world_enu = np.array([tx, ty, tz])

        # Convert the integer method selector to a human-readable label for logging
        method_name = ['transpose', 'pseudoinverse', 'DLS'][mth % 3]

        # Orientation mode is active when target_yaw is a real number (not NaN).
        # In position-only mode the 4th column of the Jacobian is [0,0,0], so q4
        # receives no velocity command and the arm behaves as a 3-DOF position task.
        orient_mode = math.isfinite(t_yaw)

        # TF lookups: robot base pose, link1 position, and end-effector position
        ee_world_enu = self._tf_pos(EE_FRAME)
        link1_world_enu = self._tf_pos(J1_FRAME)  # arm base in world (needed for FK)
        base_pose = self._tf_pose('turtlebot/base_footprint')
        
        if base_pose is not None:
            self._base_x, self._base_y, self._base_psi = base_pose

        # Latch initial base yaw on first valid TF reading.
        # delta_yaw corrects for any base rotation since then.
        # if self._initial_base_psi is None and base_pose is not None:
        #     self._initial_base_psi = self._base_psi
        #     self.get_logger().info(
        #         f'Initial base yaw latched: {self._initial_base_psi:.4f} rad')

        if link1_world_enu is not None:
            self._link1_world_enu = link1_world_enu

        if ee_world_enu is not None:
            self._last_tf_ee = ee_world_enu

        # 5-DOF VMS FK 
        q_arm = np.array(self._vms_state.arm_q)   # [q1, q2, q3, q4] — full 4-DOF
        fk_world_enu = swiftpro_fk_vms_5dof(
            self._link1_world_enu, q_arm, self._base_psi, self._tf_buffer)
        
        if fk_world_enu is None:
            self.get_logger().warn('FK computation failed, skipping control tick',
                                   throttle_duration_sec=1.0)
            return

        # Path-tracking defaults — overwritten inside the position sub-task branch.
        path_alpha   = float('nan')
        path_desired = target_world_enu.copy()

        # Position error: TF-based if available, otherwise FK-based
        if ee_world_enu is not None:
            pos_err = (target_world_enu - ee_world_enu).reshape(3, 1)
            err_norm = float(np.linalg.norm(pos_err))
            fk_vs_tf_mm = float(np.linalg.norm(fk_world_enu - ee_world_enu)) * 1000
            position_source = 'TF'
            ee_for_jac = ee_world_enu   # use ground-truth EE for Jacobian omega column

        else:
            pos_err = (target_world_enu - fk_world_enu).reshape(3, 1)
            err_norm = float(np.linalg.norm(pos_err))
            fk_vs_tf_mm = float('nan')
            position_source = 'FK_fallback'
            ee_for_jac = fk_world_enu
            self.get_logger().warn('TF unavailable — using 5-DOF FK for error',
                                   throttle_duration_sec=1.0)

        # Task-Priority Control 
        # Update the shared robot state container used by all task objects.
        self._vms_state.update(
            ee_for_jac,
            [self._base_x, self._base_y],
            self._base_psi,
            q_arm,
            self._tf_buffer)

        yaw_k = float(self.get_parameter('yaw_gain_k').value)

        # Latch path start on first tick — skip until TF is available.
        if self._path_start is None:
            if ee_world_enu is not None:
                self._path_start   = ee_world_enu.copy()
                self._path_start_t = self.get_clock().now()
            else:
                return

        elapsed    = (self.get_clock().now() - self._path_start_t).nanoseconds / 1e9
        period     = float(self.get_parameter('goal_cycle_period').value)
        path_alpha = float(np.clip(elapsed / period, 0.0, 1.0))

        # Interpolated position waypoint (straight line start → goal).
        path_desired = self._path_start + path_alpha * (target_world_enu - self._path_start)
        ff_pos = (target_world_enu - self._path_start) / period if path_alpha < 1.0 else np.zeros(3)

        # Rebuild joint-position task objects only when goal changes.
        if self._goal_idx != self._cached_goal_idx:
            self._build_joint_tasks(self._goal_idx)

        if orient_mode:
            # Configuration task: position + yaw simultaneously via 4×6 Jacobian.
            # FF is position-only (3 components); yaw FF = 0 (feedback sufficient).
            desired_config = np.array([path_desired[0], path_desired[1],
                                       path_desired[2], t_yaw]).reshape(4, 1)
            ff_config = np.vstack([ff_pos.reshape(3, 1), [[0.0]]])
            self._config_task.setDesired(desired_config)
            self._config_task.setGain(np.diag([K_gain, K_gain, K_gain, yaw_k]))
            self._config_task.setFF(ff_config)
            tasks = (self._joint_limit_tasks
                     + self._joint_tasks_above
                     + [self._config_task]
                     + self._joint_tasks_below)
        else:
            # Position-only task — no yaw correction.
            self._pos_task.setDesired(path_desired.reshape(3, 1))
            self._pos_task.setGain(np.diag([K_gain] * 3))
            self._pos_task.setFF(ff_pos.reshape(3, 1))
            tasks = (self._joint_limit_tasks
                     + self._joint_tasks_above
                     + [self._pos_task]
                     + self._joint_tasks_below)

        # Run one step of the recursive Task-Priority algorithm (weighted DLS).
        if ena:
            dzeta = vms_task_priority_step(
                tasks, self._vms_state, damping=dam, method=mth,
                weight_matrix=VMS_WEIGHT_MATRIX).flatten()
        else:
            vms_task_priority_step(
                tasks, self._vms_state, damping=dam, method=mth,
                weight_matrix=VMS_WEIGHT_MATRIX)
            dzeta = np.zeros(6)

        if orient_mode:
            self._config_task.update(self._vms_state)
            pos_err = self._config_task.getError()[:3]   # position rows only
            yaw_err = float(self._config_task.getError()[3, 0])
            J_cond  = float(np.linalg.cond(self._config_task.getJacobian()))
        else:
            self._pos_task.update(self._vms_state)
            pos_err = self._pos_task.getError()
            yaw_err = float('nan')
            J_cond  = float(np.linalg.cond(self._pos_task.getJacobian()))

        # err_norm to waypoint (internal, used by controller)
        err_norm = float(np.linalg.norm(pos_err))

        # display_err_norm — distance from EE to the FINAL target (not the moving waypoint).
        # always sees how far the EE is from the actual goal, not from the interpolated point.
        _ee_now = ee_world_enu if ee_world_enu is not None else fk_world_enu
        display_err_norm = float(np.linalg.norm(target_world_enu - _ee_now))

        # Active joint-limit status for log
        active_limits = [t.name for t in self._joint_limit_tasks if t.isActive()]

        # Extract velocity components from the 6-DOF dzeta vector.
        v_base_linear  = dzeta[0]
        v_base_angular = dzeta[1]
        dq_arm         = dzeta[2:5]   # [dq1, dq2, dq3]
        dq4            = dzeta[5]

        # Scale base velocities independently (different units and limits).
        v_base_linear  = float(scale_velocities(np.array([v_base_linear]),  base_lin_max)[0])
        v_base_angular = float(scale_velocities(np.array([v_base_angular]), base_ang_max)[0])


        # Scale arm joints uniformly so direction is preserved.
        all_dq = scale_velocities(np.array([dq_arm[0], dq_arm[1], dq_arm[2], dq4]), mv)
        dq_arm = all_dq[:3]
        dq4    = all_dq[3]

        # Publish base velocity.
        base_cmd = Twist()
        base_cmd.linear.x  = v_base_linear
        base_cmd.angular.z = v_base_angular
        self.pub_base_cmd.publish(base_cmd)

        dq    = np.array([dq_arm[0], dq_arm[1], dq_arm[2], dq4])
        dq_arr = dq.flatten()

        if ena:
            cmd = Float64MultiArray()
            cmd.data = dq_arr.tolist()
            self.pub_cmd.publish(cmd)
        else:
            cmd = Float64MultiArray()
            cmd.data = [0.0, 0.0, 0.0, 0.0]
            self.pub_cmd.publish(cmd)

        # Verbose debug log (every 30 loops ≈ 0.5 s)
        if self._loop_count % 30 == 0:
            q = self._vms_state.arm_q
            if orient_mode:
                mode_str = f'config (pos+yaw simultaneous, target_yaw={t_yaw:.2f} rad)'
            else:
                mode_str = 'position-only'


            if orient_mode and math.isfinite(yaw_err):
                yaw_ok = abs(yaw_err) < 0.05
                pos_ok = err_norm < 0.01
                if pos_ok and yaw_ok:
                    orient_status = 'ACHIEVED (pos + yaw)'
                elif yaw_ok:
                    orient_status = 'YAW OK — pos converging'
                elif pos_ok:
                    orient_status = 'POS OK — yaw converging'
                else:
                    orient_status = 'converging...'
                orient_lines = (
                    f'  yaw target  : {t_yaw:.3f} rad  ({math.degrees(t_yaw):.1f} deg)\n'
                    f'  EE yaw      : {self._base_psi + q[0] + q[3]:.3f} rad  '
                    f'(psi={self._base_psi:.3f} + q1={q[0]:.3f} + q4={q[3]:.3f})\n'
                    f'  yaw_err     : {math.degrees(yaw_err):.2f} deg  ({yaw_err:.4f} rad)\n'
                    f'  orient status: {orient_status}\n'
                )
            else:
                orient_lines = ''

            log_str = (
                f'\n{"─"*70}\n'
                f'  solver      : task-priority | method: {method_name}  |  enabled: {ena}  |  mode: {mode_str}\n'
                f'  goal_idx    : {self._goal_idx}/{len(self._goals)-1}\n'
                f'  BASE POSE   : x={self._base_x:.3f}  y={self._base_y:.3f}  '
                f'psi={self._base_psi:.3f}\n'
                f'  ARM q[1-4]  : [{q[0]:.3f}, {q[1]:.3f}, {q[2]:.3f}, {q[3]:.3f}] rad\n'
                f'  target(wenu): [{tx:.3f}, {ty:.3f}, {tz:.3f}]\n'
                f'  TF EE(wenu) : {np.round(ee_world_enu,3) if ee_world_enu is not None else "N/A"}\n'
                f'  FK EE(wenu) : {np.round(fk_world_enu,3)}\n'
                f'  FK_vs_TF    : {fk_vs_tf_mm:.3f} mm\n'
                f'  pos_source  : {position_source}\n'
                f'  err_to_goal : {display_err_norm:.4f} m  (EE -> final target)\n'
                f'  err_to_wp   : {err_norm:.4f} m  (EE -> moving waypoint)\n'
                f'  err_vec     : {np.round(pos_err.flatten(),4)}\n'
                f'  path_alpha  : {path_alpha:.3f}  (waypoint {np.round(path_desired,3)})\n'
                + orient_lines +
                f'  [vx, omega] : [{v_base_linear:.4f}, {v_base_angular:.4f}]\n'
                f'  dq[1-4]     : {np.round(dq_arr,4)}\n'
                f'  J_cond(pos) : {J_cond:.1f}\n'
                f'  active_lims : {active_limits if active_limits else "none"}\n'
                f'  params      : K={K_gain}  yaw_k={yaw_k:.2f}  mv={mv}  dam={dam}\n'
                f'{"─"*70}'
            )
            self.get_logger().info(log_str)
            self._log_entries.append(log_str)

        # Markers
        self._publish_markers(
            target_world_enu, ee_world_enu, fk_world_enu,
            np.array([self._base_x, self._base_y, 0.0]), display_err_norm, method_name, yaw_err,
            path_start=self._path_start, path_desired=path_desired)
        # Publish joint angles, limit status, and error norms for PlotJuggler visualization
        self._publish_joint_data(q_arm, display_err_norm, err_norm)
    # Joint data publisher

    def _publish_joint_data(self, q_arm, err_to_goal=float('nan'), err_to_waypoint=float('nan')):
        """
        Publish joint angles, joint limit status, and task error norms for PlotJuggler.

        Topics
        ------
        /hoi/joint_angle_q{1-4}   : current arm joint angles (rad)
        /hoi/joint_limit_q{1-4}   : 1.0 if limit task active, 0.0 otherwise
        /hoi/err_to_goal          : EE distance to final target (m)
        /hoi/err_to_waypoint      : EE distance to moving interpolated waypoint (m)
        """
        # Extract individual joint angles
        q1, q2, q3, q4 = q_arm[0], q_arm[1], q_arm[2], q_arm[3]
        
        # Publish joint angles
        msg_q1 = Float64(data=float(q1))
        msg_q2 = Float64(data=float(q2))
        msg_q3 = Float64(data=float(q3))
        msg_q4 = Float64(data=float(q4))
        
        self.pub_q1.publish(msg_q1)
        self.pub_q2.publish(msg_q2)
        self.pub_q3.publish(msg_q3)
        self.pub_q4.publish(msg_q4)
        
        # Publish joint limit status (1 = active, 0 = inactive)
        # Check each joint limit task to see if it's active
        q1_limit_status = 1.0 if self._joint_limit_tasks[0].isActive() else 0.0
        q2_limit_status = 1.0 if self._joint_limit_tasks[1].isActive() else 0.0
        q3_limit_status = 1.0 if self._joint_limit_tasks[2].isActive() else 0.0
        q4_limit_status = 1.0 if self._joint_limit_tasks[3].isActive() else 0.0
        
        msg_q1_limit = Float64(data=q1_limit_status)
        msg_q2_limit = Float64(data=q2_limit_status)
        msg_q3_limit = Float64(data=q3_limit_status)
        msg_q4_limit = Float64(data=q4_limit_status)
        
        self.pub_q1_limit.publish(msg_q1_limit)
        self.pub_q2_limit.publish(msg_q2_limit)
        self.pub_q3_limit.publish(msg_q3_limit)
        self.pub_q4_limit.publish(msg_q4_limit)

        # Publish error norms
        self.pub_err_to_goal.publish(Float64(data=float(err_to_goal)))
        self.pub_err_to_wp.publish(Float64(data=float(err_to_waypoint)))

        # Publish base yaw
        self.pub_base_yaw.publish(Float64(data=float(self._base_psi)))

    # Marker publisher 

    def _publish_markers(self, target, tf_ee, fk_ee, j1, err_norm, method,
                         yaw_err=float('nan'), path_start=None, path_desired=None):
        """
        Publish RViz markers:
          RED    sphere  — desired target position
          GREEN  sphere  — TF EE position (yellow→green as error shrinks)
          CYAN   sphere  — FK estimate of EE
          YELLOW sphere  — J1 base
          WHITE  line    — error vector from EE to target
          WHITE  text    — method name, error, goal index
          ORANGE line    — straight-line path from start to goal (path_start → target)
          ORANGE sphere  — current interpolated waypoint being tracked
        """
        # Stamp all markers with the current node time so RViz knows they are fresh
        now = self.get_clock().now().to_msg()

        # MarkerArray allows publishing all markers in a single message
        ma = MarkerArray()

        def mk(mid, mtype):
            """Create a base Marker with shared fields pre-filled."""
            m = Marker()
            m.header.frame_id    = WORLD_FRAME   # all markers live in world_enu
            m.header.stamp       = now            # timestamp for RViz staleness check
            m.ns                 = 'rrc_methods_vms'  # namespace groups markers in RViz
            m.id                 = mid            # unique ID within the namespace
            m.type               = mtype          # SPHERE, LINE_STRIP, TEXT_VIEW_FACING, etc.
            m.action             = Marker.ADD     # ADD creates or updates; DELETE removes
            m.pose.orientation.w = 1.0            # identity quaternion (no rotation)
            return m

        def sphere(mid, pos, r, g, b, size=0.025):
            """Create a coloured sphere marker at the given world_enu position."""
            m = mk(mid, Marker.SPHERE)
            m.pose.position.x = float(pos[0])  
            m.pose.position.y = float(pos[1])   
            m.pose.position.z = float(pos[2])   
            m.scale.x = m.scale.y = m.scale.z = size   # uniform sphere diameter
            m.color.r = r; m.color.g = g; m.color.b = b; m.color.a = 1.0
            return m

        # RED sphere — desired target position (DISTANT goal in world_enu, static shape, bright red)
        ma.markers.append(sphere(ID_TARGET, target, 1.0, 0.0, 0.0, 0.030))

        # GREEN/YELLOW sphere — true EE from TF; colour encodes how close we are.
        # closeness=1 (err=0) -> pure green.  closeness=0 (err>=10cm) -> yellow.
        # This gives an at-a-glance indication of convergence without reading numbers.
        if tf_ee is not None:
            # Normalise error to [0,1] over the range 0-10 cm
            closeness = float(np.clip(1.0 - err_norm / 0.1, 0.0, 1.0))
            m = mk(ID_TF_EE, Marker.SPHERE)
            m.pose.position.x = float(tf_ee[0])
            m.pose.position.y = float(tf_ee[1])
            m.pose.position.z = float(tf_ee[2])
            m.scale.x = m.scale.y = m.scale.z = 0.022   # slightly smaller than target
            m.color.r = 1.0 - closeness   # 1=yellow (far), 0=green (close)
            m.color.g = 1.0               # green channel always full
            m.color.b = 0.0
            m.color.a = 1.0
            ma.markers.append(m)

        # CYAN sphere — VMS FK estimate of EE.  If this drifts far from the green TF sphere,
        # the VMS FK model needs recalibration (FK_vs_TF metric in the log will be large).
        ma.markers.append(sphere(ID_FK_EE, fk_ee, 0.0, 0.8, 1.0, 0.015))

        # YELLOW sphere — J1 base position (origin of the arm-local ENU frame).
        # Helps verify that the arm_local→world transform is correct.
        if j1 is not None:
            ma.markers.append(sphere(ID_J1_BASE, j1, 1.0, 1.0, 0.0, 0.018))

        # WHITE line — error vector drawn from the current TF EE to the target.
        # The length and direction of this line visually encode the 3-D error.
        if tf_ee is not None:
            m2 = mk(ID_ERR_LINE, Marker.LINE_STRIP)
            m2.scale.x = 0.004              # line width in metres
            m2.color.r = m2.color.g = m2.color.b = 1.0   # white
            m2.color.a = 0.9

            # LINE_STRIP requires a list of Point messages for its vertices.
            # Two points define a single line segment (current EE → target).
            p1 = Point()
            p1.x = float(tf_ee[0]); p1.y = float(tf_ee[1]); p1.z = float(tf_ee[2])
            p2 = Point()
            p2.x = float(target[0]); p2.y = float(target[1]); p2.z = float(target[2])
            m2.points = [p1, p2]
            ma.markers.append(m2)

        # WHITE text — overlaid above the target sphere showing method, error, and goal.
        # TEXT_VIEW_FACING always rotates to face the camera, so it is readable from any angle.
        mt = mk(ID_TEXT, Marker.TEXT_VIEW_FACING)
        mt.pose.position.x = float(target[0])
        mt.pose.position.y = float(target[1])
        mt.pose.position.z = float(target[2]) + 0.06   # offset above the target sphere
        mt.scale.z = 0.025    # text height in metres (scale.z controls font size for text markers)
        mt.color.r = mt.color.g = mt.color.b = mt.color.a = 1.0   # fully opaque white
        t_yaw_cur = self.get_parameter('target_yaw').value
        if math.isfinite(t_yaw_cur) and math.isfinite(yaw_err):
            mt.text = (f'[{method}] err={err_norm:.3f}m  goal{self._goal_idx}\n'
                       f'yaw_err={math.degrees(yaw_err):.1f}deg')
        else:
            mt.text = f'[{method}] err={err_norm:.3f}m  goal{self._goal_idx}'
        ma.markers.append(mt)

        # ORANGE line — full intended straight-line path (path_start -> target).
        # Visible as soon as the position sub-task begins; stays fixed until the
        # goal changes, giving a clear reference for how straight the motion is.
        if path_start is not None:
            mp = mk(ID_PATH_LINE, Marker.LINE_STRIP)
            mp.scale.x = 0.006
            mp.color.r = 1.0; mp.color.g = 0.5; mp.color.b = 0.0; mp.color.a = 0.8
            ps = Point(); ps.x = float(path_start[0]); ps.y = float(path_start[1]); ps.z = float(path_start[2])
            pe = Point(); pe.x = float(target[0]);     pe.y = float(target[1]);     pe.z = float(target[2])
            mp.points = [ps, pe]
            ma.markers.append(mp)

        # ORANGE sphere — current interpolated waypoint the controller is tracking.
        # Slides from path_start to target over the goal_cycle_period, showing
        # exactly which point the DLS is driving toward at each tick.
        if path_desired is not None:
            ma.markers.append(sphere(ID_PATH_WP, path_desired, 1.0, 0.5, 0.0, 0.018))

        # Publish all markers in a single message to minimise latency jitter
        self.pub_markers.publish(ma)


    def save_log(self):
        """Write all accumulated log entries to a timestamped text file."""
        if not self._log_entries:
            self.get_logger().info('No log entries to save.')
            return
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        log_dir = os.path.expanduser('~/ROS2_Crash_Course/ros2_ws/src/hoi_control/logs')
        os.makedirs(log_dir, exist_ok=True)
        filepath = os.path.join(log_dir, f'vms_log_{timestamp}.txt')
        with open(filepath, 'w') as f:
            f.write(f'VMS node log — saved {datetime.datetime.now()}\n')
            f.write(f'Total entries: {len(self._log_entries)}\n')
            f.write('\n'.join(self._log_entries))
        self.get_logger().info(f'Log saved → {filepath}  ({len(self._log_entries)} entries)')


def main(args=None):
    rclpy.init(args=args)
    node = Lab2RRCMethodsVMSNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.save_log()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
