#!/usr/bin/env python3
"""
lab5_obstacle_node.py
=====================
ROS 2 equivalent of lab5_obstacle_avoidance.py
------------------------------------------------
Task hierarchy:
  1. Obstacle3D (sphere 1)  – highest priority inequality tasks
  2. Obstacle3D (sphere 2)
  3. Position3D (EE pos)    – lowest priority equality task

The arm avoids 3-D spherical obstacles while tracking a sequence of
goal positions (cycling every 10 seconds exactly as in lab5).

Corresponds to: lab5_obstacle_avoidance.py
Changes vs lab:
  - 3-D spherical obstacles instead of 2-D circles
  - Geometric Jacobian
  - ROS interface

Run:
    ros2 run hoi_control lab5_obstacle_node.py
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Float64MultiArray
import numpy as np

from hoi_control.swiftpro_robotics import (
    SwiftProManipulator, Position3D, Obstacle3D,
    task_priority_step, scale_velocities, swiftpro_fk, DLS
)

DT         = 1.0 / 60.0
MAX_VEL    = 1.0
DAMPING    = 0.1
GOAL_PERIOD = 10.0

JOINT_CMD_TOPIC   = '/turtlebot/swiftpro/joint_velocity_controller/command'
JOINT_STATE_TOPIC = '/turtlebot/joint_states'
EE_POS_TOPIC      = '/hoi/ee_position'
TARGET_TOPIC      = '/hoi/target_position'

# Obstacle definitions (NED frame, metres) – placed in arm workspace
OBSTACLES = [
    {'pos': [0.20,  0.05, -0.10], 'r': 0.04},   # obstacle 1
    {'pos': [0.15, -0.05, -0.20], 'r': 0.04},   # obstacle 2
]

# Fixed goal sequence (NED, metres) – same concept as lab5 fixed_goal_positions
GOALS = [
    swiftpro_fk([0.3, 0.5, -0.3]).reshape(3,1),
    swiftpro_fk([-0.3, 0.8,  0.2]).reshape(3,1),
    swiftpro_fk([0.0, 1.0, -0.2]).reshape(3,1),
    swiftpro_fk([0.5, 0.3,  0.0]).reshape(3,1),
]


class Lab5ObstacleNode(Node):

    def __init__(self):
        super().__init__('lab5_obstacle_node')
        self.arm = SwiftProManipulator()

        # Build task list
        self._obs_tasks = [
            Obstacle3D(f'obs_{i}',
                       obs['pos'],
                       r_activate=obs['r'],
                       r_deactivate=obs['r'] + 0.02)
            for i, obs in enumerate(OBSTACLES)
        ]
        for ot in self._obs_tasks:
            ot.setGain(np.array([[10.0]]))
            ot.setFF(np.zeros((1,1)))

        self._task_ee = Position3D('EE position', GOALS[0].copy())
        self._task_ee.setGain(np.diag([2.0, 2.0, 2.0]))
        self._task_ee.setFF(np.zeros((3,1)))

        self._tasks = self._obs_tasks + [self._task_ee]

        self.sub_js = self.create_subscription(
            JointState, JOINT_STATE_TOPIC, self._js_cb, 10)
        self.pub_cmd = self.create_publisher(
            Float64MultiArray, JOINT_CMD_TOPIC, 10)
        self.pub_ee = self.create_publisher(
            PointStamped, EE_POS_TOPIC, 10)
        self.pub_tgt = self.create_publisher(
            PointStamped, TARGET_TOPIC, 10)

        self.timer   = self.create_timer(DT, self._control_loop)
        self._got_js = False
        self._goal_t = 0.0
        self._goal_idx = 0
        self.get_logger().info('Lab5 Obstacle-Avoidance node started.')

    def _js_cb(self, msg):
        self.arm.update_from_joint_states(list(msg.name), list(msg.position))
        self._got_js = True

    def _control_loop(self):
        if not self._got_js:
            return

        self._goal_t += DT
        if self._goal_t >= GOAL_PERIOD:
            self._goal_t = 0.0
            self._goal_idx = (self._goal_idx + 1) % len(GOALS)
            self._task_ee.setDesired(GOALS[self._goal_idx].copy())
            self.get_logger().info(f'Goal -> index {self._goal_idx}')

        # --- Recursive Task-Priority with inequality handling ---
        n    = self.arm.getDOF()
        P    = np.eye(n)
        zeta = np.zeros((n, 1))

        for task in self._tasks:
            task.update(self.arm)

            if not task.isActive():
                continue

            err_i = task.getError()

            # For obstacle tasks: skip if constraint already satisfied
            if isinstance(task, Obstacle3D) and err_i[0, 0] <= 0:
                continue

            Ji      = task.getJacobian()
            Ki      = task.getGain()
            ff_i    = task.getFF()
            xi_dot  = Ki @ err_i + ff_i
            Ji_bar  = Ji @ P
            Ji_dls  = DLS(Ji_bar, DAMPING)

            zeta = zeta + Ji_dls @ (xi_dot - Ji @ zeta)

            Ji_bar_pinv = np.linalg.pinv(Ji_bar)
            P = P - Ji_bar_pinv @ Ji_bar

        zeta = scale_velocities(zeta, MAX_VEL)

        cmd = Float64MultiArray()
        cmd.data = zeta.flatten().tolist() + [0.0]
        self.pub_cmd.publish(cmd)

        p = self.arm.getEEPosition()
        self._pub_pt(self.pub_ee,  p)
        self._pub_pt(self.pub_tgt, self._task_ee.getDesired().flatten())

        obs_info = ' '.join(
            [f'd{i}={ot.getDistance():.3f}({"ON" if ot.isActive() else "off"})'
             for i, ot in enumerate(self._obs_tasks)])
        self.get_logger().info(
            f'{obs_info}  '
            f'|err_EE|={np.linalg.norm(self._task_ee.getError()) if self._task_ee.err is not None else 0.0:.3f}',
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
    node = Lab5ObstacleNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
