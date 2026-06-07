#!/usr/bin/env python3
"""
lab3_two_tasks_node.py
======================
ROS 2 equivalent of lab3_two_tasks.py
--------------------------------------
Two-task priority control for the uArm Swift Pro:
  Task 1: End-effector 3-D position  (primary or secondary depending on case)
  Task 2: Joint 1 (base yaw) position  (primary or secondary)

Switchable via ROS parameter 'case':
  'a' → EE position is primary  (exactly as lab3 case 'a')
  'b' → Joint 1 position is primary  (exactly as lab3 case 'b')

Corresponds to: lab3_two_tasks.py
Changes vs lab:
  - 3-D position task instead of 2-D
  - Geometric Jacobian
  - ROS interface

Run:
    ros2 run hoi_control lab3_two_tasks_node.py
    ros2 run hoi_control lab3_two_tasks_node.py --ros-args -p case:=a
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Float64MultiArray
import numpy as np

from hoi_control.swiftpro_robotics import (
    SwiftProManipulator, Position3D, JointPosition,
    DLS, task_priority_step, scale_velocities
)

DT        = 1.0 / 60.0
MAX_VEL   = 1.0
DAMPING   = 0.1

JOINT_CMD_TOPIC   = '/turtlebot/swiftpro/joint_velocity_controller/command'
JOINT_STATE_TOPIC = '/turtlebot/joint_states'
EE_POS_TOPIC      = '/hoi/ee_position'

# Desired EE position (NED, metres) – reachable point for the arm
SIGMA_EE_D = np.array([[0.20], [0.05], [-0.15]])    # 3×1
# Desired joint 1 position (rad)
SIGMA_Q1_D = np.array([[0.0]])                       # 1×1


class Lab3TwoTasksNode(Node):

    def __init__(self):
        super().__init__('lab3_two_tasks_node')

        # ROS parameter 'case': 'a' or 'b'
        self.declare_parameter('case', 'b')
        self._case = self.get_parameter('case').value
        self.get_logger().info(f'Lab3 Two-Tasks node, case={self._case}')

        self.arm = SwiftProManipulator()

        # Task objects
        self._task_ee = Position3D('EE position', SIGMA_EE_D.copy())
        self._task_ee.setGain(np.diag([2.0, 2.0, 2.0]))
        self._task_ee.setFF(np.zeros((3, 1)))

        self._task_q1 = JointPosition('Joint1 position', SIGMA_Q1_D.copy(), joint_idx=0)
        self._task_q1.setGain(np.array([[2.0]]))
        self._task_q1.setFF(np.zeros((1, 1)))

        # Assign priority based on case
        if self._case == 'a':
            self._tasks = [self._task_ee, self._task_q1]
        else:  # 'b'
            self._tasks = [self._task_q1, self._task_ee]

        self.sub_js = self.create_subscription(
            JointState, JOINT_STATE_TOPIC, self._js_cb, 10)
        self.pub_cmd = self.create_publisher(
            Float64MultiArray, JOINT_CMD_TOPIC, 10)
        self.pub_ee = self.create_publisher(
            PointStamped, EE_POS_TOPIC, 10)

        self.timer = self.create_timer(DT, self._control_loop)
        self._got_js = False

    def _js_cb(self, msg: JointState):
        self.arm.update_from_joint_states(list(msg.name), list(msg.position))
        self._got_js = True

    def _control_loop(self):
        if not self._got_js:
            return

        # --- Recursive Task-Priority (same algorithm as lab3_two_tasks.py) ---
        zeta = task_priority_step(self._tasks, self.arm, damping=DAMPING)
        zeta = scale_velocities(zeta, MAX_VEL)

        # --- Publish ---
        cmd = Float64MultiArray()
        cmd.data = zeta.flatten().tolist() + [0.0]
        self.pub_cmd.publish(cmd)

        # --- EE position for RViz ---
        p = self.arm.getEEPosition()
        ee_msg = PointStamped()
        ee_msg.header.stamp = self.get_clock().now().to_msg()
        ee_msg.header.frame_id = 'turtlebot/base_footprint'
        ee_msg.point.x, ee_msg.point.y, ee_msg.point.z = p[0], p[1], p[2]
        self.pub_ee.publish(ee_msg)

        # --- Log ---
        err1 = np.linalg.norm(self._task_ee.getError()) if self._task_ee.err is not None else 0.0
        err2 = np.linalg.norm(self._task_q1.getError()) if self._task_q1.err is not None else 0.0
        self.get_logger().info(
            f'case={self._case} | |err_EE|={err1:.4f} m | |err_q1|={err2:.4f} rad',
            throttle_duration_sec=0.5)

        self.arm.integrate(zeta.flatten(), DT)


def main(args=None):
    rclpy.init(args=args)
    node = Lab3TwoTasksNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
