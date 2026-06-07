#!/usr/bin/env python3
"""
lab6_mobile_manip_node.py
=========================
ROS 2 equivalent of lab6_mobile_manipulator.py
------------------------------------------------
Full 5-DOF mobile manipulator control using Task-Priority:
  State  : [x, y, θ, q1, q2, q3]  (base pose + arm joints)
  Control: [v, ω, dq1, dq2, dq3]  (base velocities + arm velocities)

Task hierarchy:
  1. JointLimits (q1, q2)          – highest priority: safety
  2. Position3D (EE world pos)     – primary position task
  3. JointPosition (q1 = 0)        – keep arm base centred (null-space use)

The redundancy (5 DOF for a 3-D position task) is resolved by Task-Priority:
  - Safety limits always active
  - EE goes to desired world position
  - Remaining DOF used to keep the arm base at q1=0 (arm pointing forward)

The base is driven via /turtlebot/cmd_vel (geometry_msgs/Twist).
The arm joints are driven via the velocity controller.

Odometry is read from /turtlebot/odom (nav_msgs/Odometry) for base pose.

Corresponds to: lab6_mobile_manipulator.py
Changes vs lab:
  - 5-DOF Jacobian (MobileManipulator class) instead of 3-DOF
  - ROS interface (odom subscriber + cmd_vel publisher)
  - 3-D world-frame EE position task

Run:
    ros2 run hoi_control lab6_mobile_manip_node.py
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PointStamped, Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64MultiArray
import numpy as np

from hoi_control.swiftpro_robotics import (
    MobileManipulator, Position3D, JointPosition, JointLimits,
    DLS, scale_velocities, swiftpro_fk
)

DT       = 1.0 / 60.0
DAMPING  = 0.1

# Velocity limits
MAX_ARM_VEL  = 1.0    # rad/s
MAX_LINEAR   = 0.3    # m/s
MAX_ANGULAR  = 0.8    # rad/s

GOAL_PERIOD = 15.0    # seconds between goal changes

JOINT_CMD_TOPIC   = '/turtlebot/swiftpro/joint_velocity_controller/command'
JOINT_STATE_TOPIC = '/turtlebot/joint_states'
CMD_VEL_TOPIC     = '/turtlebot/cmd_vel'
ODOM_TOPIC        = '/turtlebot/odom'
EE_POS_TOPIC      = '/hoi/ee_position'
TARGET_TOPIC      = '/hoi/target_position'

# Joint limits for safety
Q1_LIMITS = (-1.57, 1.57)
Q2_LIMITS = ( 0.0,  1.40)


def random_world_target(base_x=0.0, base_y=0.0, base_theta=0.0):
    """
    Sample a world-frame EE target reachable from the current base position.
    """
    q1 = np.random.uniform(*Q1_LIMITS)
    q2 = np.random.uniform(*Q2_LIMITS)
    q3 = np.random.uniform(-0.8, 0.4)
    p_arm = swiftpro_fk([q1, q2, q3])

    cT, sT = np.cos(base_theta), np.sin(base_theta)
    px_w = base_x + cT*p_arm[0] - sT*p_arm[1]
    py_w = base_y + sT*p_arm[0] + cT*p_arm[1]
    pz_w = p_arm[2]

    return np.array([[px_w], [py_w], [pz_w]])


class Lab6MobileManipNode(Node):

    def __init__(self):
        super().__init__('lab6_mobile_manip_node')
        self.robot = MobileManipulator()

        # Build task hierarchy (5-DOF)
        initial_target = random_world_target()

        self._tasks = [
            JointLimits('q1_limits', joint_idx=2,
                        q_min=Q1_LIMITS[0], q_max=Q1_LIMITS[1], margin=0.05),
            JointLimits('q2_limits', joint_idx=3,
                        q_min=Q2_LIMITS[0], q_max=Q2_LIMITS[1], margin=0.05),
            Position3D('EE world position', initial_target),
            JointPosition('Arm q1 centre', np.array([[0.0]]), joint_idx=2),
        ]

        # Gains
        self._tasks[0].setGain(np.array([[10.0]])); self._tasks[0].setFF(np.zeros((1,1)))
        self._tasks[1].setGain(np.array([[10.0]])); self._tasks[1].setFF(np.zeros((1,1)))
        self._tasks[2].setGain(np.diag([2.0, 2.0, 2.0])); self._tasks[2].setFF(np.zeros((3,1)))
        self._tasks[3].setGain(np.array([[1.0]])); self._tasks[3].setFF(np.zeros((1,1)))

        # Subscribers
        self.sub_js   = self.create_subscription(
            JointState, JOINT_STATE_TOPIC, self._js_cb, 10)
        self.sub_odom = self.create_subscription(
            Odometry, ODOM_TOPIC, self._odom_cb, 10)

        # Publishers
        self.pub_arm_cmd = self.create_publisher(
            Float64MultiArray, JOINT_CMD_TOPIC, 10)
        self.pub_base_cmd = self.create_publisher(
            Twist, CMD_VEL_TOPIC, 10)
        self.pub_ee  = self.create_publisher(PointStamped, EE_POS_TOPIC, 10)
        self.pub_tgt = self.create_publisher(PointStamped, TARGET_TOPIC, 10)

        self.timer   = self.create_timer(DT, self._control_loop)
        self._got_js   = False
        self._got_odom = False
        self._goal_t   = 0.0

        self.get_logger().info('Lab6 Mobile Manipulator node started.')

    def _js_cb(self, msg: JointState):
        self.robot.update_arm_from_joint_states(list(msg.name), list(msg.position))
        self._got_js = True

    def _odom_cb(self, msg: Odometry):
        x     = msg.pose.pose.position.x
        y     = msg.pose.pose.position.y
        # Extract yaw from quaternion
        qz    = msg.pose.pose.orientation.z
        qw    = msg.pose.pose.orientation.w
        theta = 2.0 * np.arctan2(qz, qw)
        self.robot.update_base_from_odom(x, y, theta)
        self._got_odom = True

    def _control_loop(self):
        if not self._got_js or not self._got_odom:
            return

        self._goal_t += DT
        if self._goal_t >= GOAL_PERIOD:
            self._goal_t = 0.0
            new_tgt = random_world_target(
                self.robot.x, self.robot.y, self.robot.theta)
            self._tasks[2].setDesired(new_tgt)
            self.get_logger().info(
                f'New world target: [{new_tgt[0,0]:.2f}, {new_tgt[1,0]:.2f}, {new_tgt[2,0]:.2f}]')

        # ----------------------------------------------------------------
        # Recursive Task-Priority (5-DOF, identical algorithm to all labs)
        # ----------------------------------------------------------------
        n    = self.robot.getDOF()   # 5
        P    = np.eye(n)
        zeta = np.zeros((n, 1))

        for task in self._tasks:
            task.update(self.robot)

            if not task.isActive():
                continue

            Ji      = task.getJacobian()
            err_i   = task.getError()
            Ki      = task.getGain()
            ff_i    = task.getFF()
            xi_dot  = Ki @ err_i + ff_i

            Ji_bar     = Ji @ P
            Ji_bar_dls = DLS(Ji_bar, DAMPING)

            zeta = zeta + Ji_bar_dls @ (xi_dot - Ji @ zeta)

            Ji_bar_pinv = np.linalg.pinv(Ji_bar)
            P = P - Ji_bar_pinv @ Ji_bar

        # ----------------------------------------------------------------
        # Split zeta into base and arm commands
        # zeta = [v, omega, dq1, dq2, dq3]
        # ----------------------------------------------------------------
        zeta = zeta.flatten()

        v     = float(np.clip(zeta[0], -MAX_LINEAR,  MAX_LINEAR))
        omega = float(np.clip(zeta[1], -MAX_ANGULAR, MAX_ANGULAR))
        dq    = np.clip(zeta[2:5], -MAX_ARM_VEL, MAX_ARM_VEL)

        # --- Publish base velocity ---
        twist = Twist()
        twist.linear.x  = v
        twist.angular.z = omega
        self.pub_base_cmd.publish(twist)

        # --- Publish arm joint velocities ---
        arm_cmd = Float64MultiArray()
        arm_cmd.data = dq.tolist()
        self.pub_arm_cmd.publish(arm_cmd)

        # --- Publish EE and target for RViz ---
        p_ee  = self.robot.getEEPosition()
        p_tgt = self._tasks[2].getDesired().flatten()
        self._pub_pt(self.pub_ee,  p_ee,  'world_enu')
        self._pub_pt(self.pub_tgt, p_tgt, 'world_enu')

        self.get_logger().info(
            f'v={v:.2f} ω={omega:.2f} | '
            f'dq=[{dq[0]:.2f},{dq[1]:.2f},{dq[2]:.2f}] | '
            f'|err_EE|={np.linalg.norm(self._tasks[2].getError()) if self._tasks[2].err is not None else 0:.3f}',
            throttle_duration_sec=0.5)

        # Integrate internal state
        self.robot.integrate(np.concatenate([[v, omega], dq]), DT)

    def _pub_pt(self, pub, p, frame):
        msg = PointStamped()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = frame
        msg.point.x = float(p[0])
        msg.point.y = float(p[1])
        msg.point.z = float(p[2])
        pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = Lab6MobileManipNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
