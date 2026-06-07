#!/usr/bin/env python3
"""
lab2_kinematics_node.py
=======================
ROS 2 equivalent of lab2_kinematics.py
---------------------------------------
Applies constant joint velocities [dq1, dq2, dq3] and publishes the
resulting end-effector position as a ROS topic, while also printing a
table of FK results.

Corresponds to: lab2_kinematics.py
Changes vs lab:
  - No matplotlib; uses ROS topics + rviz for visualisation
  - FK is geometric (swiftpro_fk) instead of DH chain
  - Reads real joint states from simulator
  - Publishes /hoi/ee_position (geometry_msgs/PointStamped)

Run:
    ros2 run hoi_control lab2_kinematics_node.py
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Float64MultiArray
import numpy as np

from hoi_control.swiftpro_robotics import (
    SwiftProManipulator, swiftpro_fk, swiftpro_points, scale_velocities
)

# Constant joint velocities to apply (rad/s) – same spirit as lab2_kinematics
DQ = np.array([0.1, 0.2, -0.1])   # [dq1, dq2, dq3]
DT = 0.01                          # 100 Hz control loop
MAX_VEL = 0.5                      # rad/s

# JOINT_CMD_TOPIC = '/turtlebot/swiftpro/joint_velocity_controller/command'
JOINT_CMD_TOPIC = '/swiftpro/joint_velocity_controller/command'

# JOINT_STATE_TOPIC = '/turtlebot/joint_states'
JOINT_STATE_TOPIC = '/swiftpro/joint_states'
EE_POS_TOPIC = '/hoi/ee_position'


class Lab2KinematicsNode(Node):

    def __init__(self):
        super().__init__('lab2_kinematics_node')
        self.arm = SwiftProManipulator()

        # Subscriber
        self.sub_js = self.create_subscription(
            JointState, JOINT_STATE_TOPIC, self._js_cb, 10)

        # Publishers
        self.pub_cmd = self.create_publisher(
            Float64MultiArray, JOINT_CMD_TOPIC, 10)
        self.pub_ee = self.create_publisher(
            PointStamped, EE_POS_TOPIC, 10)

        # Timer – runs at 1/DT Hz
        self.timer = self.create_timer(DT, self._control_loop)

        self._got_joint_states = False
        self.get_logger().info('Lab2 Kinematics node started.')

    def _js_cb(self, msg: JointState):
        self.arm.update_from_joint_states(list(msg.name), list(msg.position))
        self._got_joint_states = True

    def _control_loop(self):
        if not self._got_joint_states:
            return

        # --- Forward Kinematics (geometric, no DH) ---
        p = swiftpro_fk(self.arm.q)

        # Log
        q = self.arm.q
        self.get_logger().info(
            f'q=[{q[0]:.3f}, {q[1]:.3f}, {q[2]:.3f}] rad  |  '
            f'EE=[{p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f}] m')

        # Publish EE position for RViz
        ee_msg = PointStamped()
        ee_msg.header.stamp = self.get_clock().now().to_msg()
        ee_msg.header.frame_id = 'world_ned'
        ee_msg.point.x, ee_msg.point.y, ee_msg.point.z = p[0], p[1], p[2]
        self.pub_ee.publish(ee_msg)

        # Apply constant joint velocities (as in lab2_kinematics)
        dq = DQ.copy().reshape(3,)
        # Clamp to max velocity
        s = np.max(np.abs(dq)) / MAX_VEL
        if s > 1.0:
            dq /= s

        cmd = Float64MultiArray()
        cmd.data = dq.tolist() + [0.0]
        self.pub_cmd.publish(cmd)

        # Integrate locally (simulator will update actual state)
        # We also integrate here to track where we expect to be
        self.arm.integrate(dq, DT)


def main(args=None):
    rclpy.init(args=args)
    node = Lab2KinematicsNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
