#!/usr/bin/env python3
"""
mission_coordinator_node.py
Top-level mission orchestrator for integrated frontier exploration + pick-and-place.

FSM
---
EXPLORING        — frontier exploration is running; wait for /frontier/exploration_complete
                   and for the scout to report a marker pose on /aruco/best_pose.
NAV_TO_STANDOFF  — publish stand-off goal to /path_planner/mission_goal and wait
                   for the robot to arrive (monitored via /turtlebot/odom).
ACTIVATE_PICK    — publish marker pose on /pick_place/marker_pose, call start_pick.
WAIT_PICK_DONE   — wait for /pick_place/status == 'IDLE_HOLDING'.
NAV_TO_PLACE     — publish place destination to /path_planner/mission_goal and wait.
ACTIVATE_PLACE   — publish place dest on /pick_place/place_dest, call start_place.
WAIT_PLACE_DONE  — wait for /pick_place/status == 'IDLE'.
DONE             — mission complete; stop.

Topics subscribed:
  /frontier/exploration_complete  (std_msgs/Bool)
  /aruco/best_pose                (geometry_msgs/PoseStamped, TRANSIENT_LOCAL)
  /pick_place/status              (std_msgs/String)
  /turtlebot/odom                 (nav_msgs/Odometry)

Topics published:
  /path_planner/mission_goal      (geometry_msgs/PoseStamped)
  /pick_place/marker_pose         (geometry_msgs/PoseStamped, TRANSIENT_LOCAL)
  /pick_place/place_dest          (geometry_msgs/PoseStamped, TRANSIENT_LOCAL)

Services called:
  /pick_place/start_pick          (std_srvs/Trigger)
  /pick_place/start_place         (std_srvs/Trigger)

Parameters:
  place_dest_x         (float, default 0.0)   — world_enu X of place destination
  place_dest_y         (float, default 3.0)   — world_enu Y of place destination
  standoff_dist        (float, default 0.80)  — m stand-off from marker for start_pick
  nav_arrival_radius   (float, default 0.20)  — m base position tolerance for nav goals
  nav_timeout_sec      (float, default 60.0)  — s before declaring nav timeout
  pick_timeout_sec     (float, default 120.0) — s before declaring pick timeout
  place_timeout_sec    (float, default 120.0) — s before declaring place timeout
"""

import math
import time
from enum import Enum, auto

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import Bool, String
from std_srvs.srv import Trigger
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry

_LATCHED_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)


class CoordState(Enum):
    EXPLORING       = auto()
    NAV_TO_STANDOFF = auto()
    ACTIVATE_PICK   = auto()
    WAIT_PICK_DONE  = auto()
    NAV_TO_PLACE    = auto()
    ACTIVATE_PLACE  = auto()
    WAIT_PLACE_DONE = auto()
    DONE            = auto()


class MissionCoordinatorNode(Node):

    def __init__(self):
        super().__init__('mission_coordinator_node')

        # ── Parameters ────────────────────────────────────────────────────────
        self.declare_parameter('place_dest_x',       0.0)
        self.declare_parameter('place_dest_y',       3.0)
        self.declare_parameter('standoff_dist',      0.80)
        self.declare_parameter('nav_arrival_radius', 0.20)
        self.declare_parameter('nav_timeout_sec',    60.0)
        self.declare_parameter('pick_timeout_sec',  120.0)
        self.declare_parameter('place_timeout_sec', 120.0)

        self._place_dest_x      = self.get_parameter('place_dest_x').value
        self._place_dest_y      = self.get_parameter('place_dest_y').value
        self._standoff_dist     = self.get_parameter('standoff_dist').value
        self._nav_arrival_radius = self.get_parameter('nav_arrival_radius').value
        self._nav_timeout_sec   = self.get_parameter('nav_timeout_sec').value
        self._pick_timeout_sec  = self.get_parameter('pick_timeout_sec').value
        self._place_timeout_sec = self.get_parameter('place_timeout_sec').value

        # ── FSM ───────────────────────────────────────────────────────────────
        self._state      = CoordState.EXPLORING
        self._state_t    = time.time()
        self._nav_goal   = None   # (x, y) current nav target

        # ── Inputs ────────────────────────────────────────────────────────────
        self._exploration_complete = False
        self._marker_pose: PoseStamped | None = None
        self._pick_place_status = 'IDLE'
        self._robot_x = 0.0
        self._robot_y = 0.0

        # ── Service futures ───────────────────────────────────────────────────
        self._pending_future = None

        # ── Publishers ────────────────────────────────────────────────────────
        self._pub_mission_goal = self.create_publisher(
            PoseStamped, '/path_planner/mission_goal', 10)
        self._pub_marker_pose  = self.create_publisher(
            PoseStamped, '/pick_place/marker_pose', _LATCHED_QOS)
        self._pub_place_dest   = self.create_publisher(
            PoseStamped, '/pick_place/place_dest', _LATCHED_QOS)

        # ── Service clients ───────────────────────────────────────────────────
        self._cli_start_pick  = self.create_client(Trigger, '/pick_place/start_pick')
        self._cli_start_place = self.create_client(Trigger, '/pick_place/start_place')

        # ── Subscribers ───────────────────────────────────────────────────────
        self.create_subscription(
            Bool, '/frontier/exploration_complete', self._exploration_done_cb, 10)
        self.create_subscription(
            PoseStamped, '/aruco/best_pose', self._marker_pose_cb, _LATCHED_QOS)
        self.create_subscription(
            String, '/pick_place/status', self._pick_place_status_cb, 10)
        self.create_subscription(
            Odometry, '/turtlebot/odom', self._odom_cb, 10)

        # ── FSM timer ─────────────────────────────────────────────────────────
        self.create_timer(0.5, self._fsm_tick)

        self.get_logger().info(
            f'MissionCoordinatorNode started — State: EXPLORING  '
            f'place_dest=({self._place_dest_x:.2f},{self._place_dest_y:.2f})')

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _exploration_done_cb(self, msg: Bool):
        if msg.data and not self._exploration_complete:
            self._exploration_complete = True
            self.get_logger().info('[COORD] Exploration complete signal received')

    def _marker_pose_cb(self, msg: PoseStamped):
        self._marker_pose = msg
        self.get_logger().info(
            f'[COORD] Marker pose received: '
            f'({msg.pose.position.x:.2f},{msg.pose.position.y:.2f},{msg.pose.position.z:.2f})',
            throttle_duration_sec=10.0)

    def _pick_place_status_cb(self, msg: String):
        self._pick_place_status = msg.data

    def _odom_cb(self, msg: Odometry):
        self._robot_x = msg.pose.pose.position.x
        self._robot_y = msg.pose.pose.position.y

    # ── FSM tick ──────────────────────────────────────────────────────────────

    def _fsm_tick(self):
        s = self._state
        if s == CoordState.EXPLORING:
            self._tick_exploring()
        elif s == CoordState.NAV_TO_STANDOFF:
            self._tick_nav(self._enter_activate_pick)
        elif s == CoordState.ACTIVATE_PICK:
            self._tick_activate_pick()
        elif s == CoordState.WAIT_PICK_DONE:
            self._tick_wait_pick()
        elif s == CoordState.NAV_TO_PLACE:
            self._tick_nav(self._enter_activate_place)
        elif s == CoordState.ACTIVATE_PLACE:
            self._tick_activate_place()
        elif s == CoordState.WAIT_PLACE_DONE:
            self._tick_wait_place()
        elif s == CoordState.DONE:
            self.get_logger().info('[COORD] Mission DONE.', throttle_duration_sec=30.0)

    def _tick_exploring(self):
        if not self._exploration_complete:
            self.get_logger().info(
                '[COORD] Exploring... waiting for /frontier/exploration_complete',
                throttle_duration_sec=15.0)
            return
        if self._marker_pose is None:
            self.get_logger().warn(
                '[COORD] Exploration complete but no marker pose from scout. '
                'Ensure aruco_scout_node is running.',
                throttle_duration_sec=10.0)
            return
        self.get_logger().info('[COORD] Exploration done + marker found → NAV_TO_STANDOFF')
        self._enter_nav_to_standoff()

    def _enter_nav_to_standoff(self):
        mp = self._marker_pose
        mx = mp.pose.position.x
        my = mp.pose.position.y
        # Stand-off: approach from robot's current direction
        dx = self._robot_x - mx
        dy = self._robot_y - my
        dist = math.hypot(dx, dy)
        if dist > 0.01:
            norm_x = dx / dist
            norm_y = dy / dist
        else:
            norm_x, norm_y = 1.0, 0.0
        goal_x = mx + norm_x * self._standoff_dist
        goal_y = my + norm_y * self._standoff_dist
        self._nav_goal = (goal_x, goal_y)
        self._publish_mission_goal(goal_x, goal_y)
        self.get_logger().info(
            f'[COORD] NAV_TO_STANDOFF: goal=({goal_x:.2f},{goal_y:.2f})')
        self._transition(CoordState.NAV_TO_STANDOFF)

    def _tick_nav(self, on_arrival):
        if self._nav_goal is None:
            return
        gx, gy = self._nav_goal
        dist = math.hypot(self._robot_x - gx, self._robot_y - gy)
        elapsed = time.time() - self._state_t

        self.get_logger().info(
            f'[COORD] {self._state.name}: dist_to_goal={dist:.2f}m  '
            f'elapsed={elapsed:.0f}/{self._nav_timeout_sec:.0f}s',
            throttle_duration_sec=5.0)

        if dist < self._nav_arrival_radius:
            self.get_logger().info(
                f'[COORD] {self._state.name}: arrived (dist={dist:.2f}m)')
            on_arrival()
            return

        if elapsed > self._nav_timeout_sec:
            self.get_logger().warn(
                f'[COORD] {self._state.name}: timeout — proceeding anyway '
                f'(dist={dist:.2f}m)')
            on_arrival()

    def _enter_activate_pick(self):
        self.get_logger().info('[COORD] → ACTIVATE_PICK')
        self._transition(CoordState.ACTIVATE_PICK)

    def _tick_activate_pick(self):
        if not self._cli_start_pick.service_is_ready():
            self.get_logger().info('[COORD] Waiting for /pick_place/start_pick service…',
                                   throttle_duration_sec=3.0)
            return

        # Publish marker pose (TRANSIENT_LOCAL) so pick-place node has it
        if self._marker_pose is not None:
            self._pub_marker_pose.publish(self._marker_pose)

        if self._pending_future is None:
            self.get_logger().info('[COORD] Calling /pick_place/start_pick')
            self._pending_future = self._cli_start_pick.call_async(Trigger.Request())
            return

        if self._pending_future.done():
            result = self._pending_future.result()
            self._pending_future = None
            if result.success:
                self.get_logger().info(f'[COORD] start_pick OK → WAIT_PICK_DONE')
                self._transition(CoordState.WAIT_PICK_DONE)
            else:
                self.get_logger().error(
                    f'[COORD] start_pick FAILED: {result.message} — retrying in 2s')
                self._state_t = time.time() - self._pick_timeout_sec + 2.0

    def _tick_wait_pick(self):
        elapsed = time.time() - self._state_t
        self.get_logger().info(
            f'[COORD] WAIT_PICK_DONE: status={self._pick_place_status}  '
            f'elapsed={elapsed:.0f}/{self._pick_timeout_sec:.0f}s',
            throttle_duration_sec=5.0)

        if self._pick_place_status == 'IDLE_HOLDING':
            self.get_logger().info('[COORD] Pick complete — entering NAV_TO_PLACE')
            self._enter_nav_to_place()
            return
        if elapsed > self._pick_timeout_sec:
            self.get_logger().warn('[COORD] WAIT_PICK_DONE timeout — attempting NAV_TO_PLACE')
            self._enter_nav_to_place()

    def _enter_nav_to_place(self):
        self._nav_goal = (self._place_dest_x, self._place_dest_y)
        self._publish_mission_goal(self._place_dest_x, self._place_dest_y)
        self.get_logger().info(
            f'[COORD] NAV_TO_PLACE: goal=({self._place_dest_x:.2f},{self._place_dest_y:.2f})')
        self._transition(CoordState.NAV_TO_PLACE)

    def _enter_activate_place(self):
        self.get_logger().info('[COORD] → ACTIVATE_PLACE')
        self._transition(CoordState.ACTIVATE_PLACE)

    def _tick_activate_place(self):
        if not self._cli_start_place.service_is_ready():
            self.get_logger().info('[COORD] Waiting for /pick_place/start_place service…',
                                   throttle_duration_sec=3.0)
            return

        # Publish place destination (TRANSIENT_LOCAL) so pick-place node has it
        dest_msg = PoseStamped()
        dest_msg.header.frame_id = 'world_enu'
        dest_msg.header.stamp    = self.get_clock().now().to_msg()
        dest_msg.pose.position.x = self._place_dest_x
        dest_msg.pose.position.y = self._place_dest_y
        dest_msg.pose.position.z = 0.0
        dest_msg.pose.orientation.w = 1.0
        self._pub_place_dest.publish(dest_msg)

        if self._pending_future is None:
            self.get_logger().info('[COORD] Calling /pick_place/start_place')
            self._pending_future = self._cli_start_place.call_async(Trigger.Request())
            return

        if self._pending_future.done():
            result = self._pending_future.result()
            self._pending_future = None
            if result.success:
                self.get_logger().info('[COORD] start_place OK → WAIT_PLACE_DONE')
                self._transition(CoordState.WAIT_PLACE_DONE)
            else:
                self.get_logger().error(
                    f'[COORD] start_place FAILED: {result.message} — retrying in 2s')

    def _tick_wait_place(self):
        elapsed = time.time() - self._state_t
        self.get_logger().info(
            f'[COORD] WAIT_PLACE_DONE: status={self._pick_place_status}  '
            f'elapsed={elapsed:.0f}/{self._place_timeout_sec:.0f}s',
            throttle_duration_sec=5.0)

        if self._pick_place_status == 'IDLE':
            self.get_logger().info('[COORD] Place complete — DONE!')
            self._transition(CoordState.DONE)
            return
        if elapsed > self._place_timeout_sec:
            self.get_logger().warn('[COORD] WAIT_PLACE_DONE timeout — declaring DONE')
            self._transition(CoordState.DONE)

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _transition(self, new_state: CoordState):
        self.get_logger().info(f'[COORD] {self._state.name} → {new_state.name}')
        self._state    = new_state
        self._state_t  = time.time()
        self._pending_future = None

    def _publish_mission_goal(self, x: float, y: float):
        """Publish a goal pose to path_planner via /path_planner/mission_goal."""
        msg = PoseStamped()
        msg.header.frame_id = 'world_enu'
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.pose.position.x = x
        msg.pose.position.y = y
        msg.pose.position.z = 0.0
        msg.pose.orientation.w = 1.0
        self._pub_mission_goal.publish(msg)
        self.get_logger().info(f'[COORD] Published mission goal: ({x:.2f},{y:.2f})')


def main(args=None):
    rclpy.init(args=args)
    node = MissionCoordinatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
