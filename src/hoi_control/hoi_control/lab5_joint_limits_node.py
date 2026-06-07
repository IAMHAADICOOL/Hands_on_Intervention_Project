#!/usr/bin/env python3
"""
lab5_joint_limits_node.py
=========================
ROS 2 equivalent of lab5_joint_limits.py
-----------------------------------------
Task hierarchy (same priority order as lab5):
  1. JointLimits (q1)     – highest priority inequality task
  2. JointLimits (q2)     – second inequality task
  3. Position3D (EE)      – lowest priority equality task

The joint limits prevent any joint from exceeding safe angles while the
arm still tries to reach randomised 3-D targets every 10 seconds.

Corresponds to: lab5_joint_limits.py
Changes vs lab:
  - All 3 active joints are protected (lab had 1)
  - 3-D position task
  - ROS interface

Run:
    ros2 run hoi_control lab5_joint_limits_node.py
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Float64MultiArray
import numpy as np

from hoi_control.swiftpro_robotics import (
    SwiftProManipulator, Position3D, JointLimits,
    task_priority_step, scale_velocities, swiftpro_fk
)

DT       = 1.0 / 60.0
MAX_VEL  = 1.0
DAMPING  = 0.1
GOAL_PERIOD = 10.0

JOINT_CMD_TOPIC   = '/turtlebot/swiftpro/joint_velocity_controller/command'
JOINT_STATE_TOPIC = '/turtlebot/joint_states'
EE_POS_TOPIC      = '/hoi/ee_position'
TARGET_TOPIC      = '/hoi/target_position'

# Joint limits for the uArm Swift Pro (radians) – conservative safe values
Q1_LIMITS = (-1.57, 1.57)    # base yaw   ±90°
Q2_LIMITS = ( 0.0,  1.40)    # shoulder    0° – 80°
Q3_LIMITS = (-1.20, 0.60)    # elbow     –69° – 34°


def random_target():
    q1 = np.random.uniform(*Q1_LIMITS)
    q2 = np.random.uniform(*Q2_LIMITS)
    q3 = np.random.uniform(*Q3_LIMITS)
    return swiftpro_fk([q1, q2, q3]).reshape(3, 1)


class Lab5JointLimitsNode(Node):

    def __init__(self):
        super().__init__('lab5_joint_limits_node')
        self.arm = SwiftProManipulator()

        # --- Task hierarchy (same priority order as lab5) ---
        self._tasks = [
            JointLimits('q1_limits', joint_idx=0,
                        q_min=Q1_LIMITS[0], q_max=Q1_LIMITS[1], margin=0.05),
            JointLimits('q2_limits', joint_idx=1,
                        q_min=Q2_LIMITS[0], q_max=Q2_LIMITS[1], margin=0.05),
            Position3D('EE position', random_target()),
        ]
        self._tasks[0].setGain(np.array([[10.0]]))
        self._tasks[0].setFF(np.zeros((1, 1)))
        self._tasks[1].setGain(np.array([[10.0]]))
        self._tasks[1].setFF(np.zeros((1, 1)))
        self._tasks[2].setGain(np.diag([2.0, 2.0, 2.0]))
        self._tasks[2].setFF(np.zeros((3, 1)))

        self.sub_js = self.create_subscription(
            JointState, JOINT_STATE_TOPIC, self._js_cb, 10)
        self.pub_cmd = self.create_publisher(
            Float64MultiArray, JOINT_CMD_TOPIC, 10)
        self.pub_ee = self.create_publisher(
            PointStamped, EE_POS_TOPIC, 10)
        self.pub_tgt = self.create_publisher(
            PointStamped, TARGET_TOPIC, 10)

        self.timer = self.create_timer(DT, self._control_loop)
        self._got_js = False
        self._goal_t = 0.0
        self.get_logger().info('Lab5 Joint-Limits node started.')

    def _js_cb(self, msg):
        self.arm.update_from_joint_states(list(msg.name), list(msg.position))
        self._got_js = True

    def _control_loop(self):
        if not self._got_js:
            return

        self._goal_t += DT
        if self._goal_t >= GOAL_PERIOD:
            self._goal_t = 0.0
            self._tasks[2].setDesired(random_target())
            self.get_logger().info('New random target set.')

        zeta = task_priority_step(self._tasks, self.arm, damping=DAMPING)
        zeta = scale_velocities(zeta, MAX_VEL)

        cmd = Float64MultiArray()
        cmd.data = zeta.flatten().tolist() + [0.0]
        self.pub_cmd.publish(cmd)

        p = self.arm.getEEPosition()
        self._pub_pt(self.pub_ee,  p)
        self._pub_pt(self.pub_tgt, self._tasks[2].getDesired().flatten())

        q = self.arm.q
        self.get_logger().info(
            f'q=[{q[0]:.2f},{q[1]:.2f},{q[2]:.2f}]  '
            f'lim1_active={self._tasks[0].isActive()}  '
            f'lim2_active={self._tasks[1].isActive()}  '
            f'|err_EE|={np.linalg.norm(self._tasks[2].getError()) if self._tasks[2].err is not None else 0.0:.3f}',
            throttle_duration_sec=0.5)

        self.arm.integrate(zeta.flatten(), DT)

    def _pub_pt(self, pub, p):
        msg = PointStamped()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = 'turtlebot/base_footprint'
        msg.point.x = float(p[0])
        msg.point.y = float(p[1])
        msg.point.z = float(p[2])
        pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = Lab5JointLimitsNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
