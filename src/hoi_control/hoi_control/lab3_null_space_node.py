#!/usr/bin/env python3
"""
lab3_null_space_node.py
=======================
ROS 2 equivalent of lab3_null_space.py
---------------------------------------
Null-space control: primary task = hold EE at its initial position.
Secondary motion = sinusoidal joint velocities projected into the null space.

This demonstrates that with DOF > task dimension the arm can move its joints
while keeping the end-effector perfectly still.

Corresponds to: lab3_null_space.py
Changes vs lab:
  - 3-D position task (3×3 Jacobian, rank-3 – null space is empty for 3 DOF!)
    → To show the null-space effect, a reduced 2-D XY task is used, leaving
      the Z DOF free.  This mirrors the spirit of the lab (2-D task on a
      3-DOF robot  =  1-D null space).
  - Geometric Jacobian instead of DH

Run:
    ros2 run hoi_control lab3_null_space_node.py
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Float64MultiArray
import numpy as np

from hoi_control.swiftpro_robotics import (
    SwiftProManipulator, swiftpro_fk, swiftpro_jacobian, DLS, scale_velocities
)

DT       = 1.0 / 60.0
MAX_VEL  = 1.0
DAMPING  = 0.1
K        = np.diag([2.0, 2.0])      # 2×2 gain for 2-D XY task

JOINT_CMD_TOPIC   = '/turtlebot/swiftpro/joint_velocity_controller/command'
JOINT_STATE_TOPIC = '/turtlebot/joint_states'
EE_POS_TOPIC      = '/hoi/ee_position'


class Lab3NullSpaceNode(Node):

    def __init__(self):
        super().__init__('lab3_null_space_node')
        self.arm = SwiftProManipulator()

        self._sigma_d   = None   # Set on first joint state (hold initial position)
        self._got_js    = False
        self._t         = 0.0

        self.sub_js = self.create_subscription(
            JointState, JOINT_STATE_TOPIC, self._js_cb, 10)
        self.pub_cmd = self.create_publisher(
            Float64MultiArray, JOINT_CMD_TOPIC, 10)
        self.pub_ee = self.create_publisher(
            PointStamped, EE_POS_TOPIC, 10)

        self.timer = self.create_timer(DT, self._control_loop)
        self.get_logger().info('Lab3 Null-Space node started.')

    def _js_cb(self, msg: JointState):
        self.arm.update_from_joint_states(list(msg.name), list(msg.position))
        if not self._got_js:
            # Latch initial EE position as the desired target (2-D XY only)
            p = swiftpro_fk(self.arm.q)
            self._sigma_d = p[:2].copy()   # hold initial XY position
            self.get_logger().info(
                f'Initial EE position latched: XY={self._sigma_d}')
        self._got_js = True

    def _control_loop(self):
        if not self._got_js or self._sigma_d is None:
            return

        self._t += DT

        # --- Primary task: hold XY end-effector position (2-D) ---
        p     = swiftpro_fk(self.arm.q)
        sigma = p[:2]                            # current XY
        err   = (self._sigma_d - sigma).reshape(2, 1)

        J_full = swiftpro_jacobian(self.arm.q)   # 3×3
        Jbar   = J_full[:2, :]                   # 2×3  (XY rows only)

        J_pinv = np.linalg.pinv(Jbar)            # 3×2
        P      = np.eye(3) - J_pinv @ Jbar       # 3×3 null-space projector

        dq_primary = J_pinv @ K @ err            # 3×1

        # --- Null-space motion: sinusoidal arbitrary joint velocity ---
        # Same parametric form as lab3_null_space.py
        y = 3.0 * np.array([
            [np.sin(self._t)],
            [np.cos(2.0 * self._t)],
            [np.sin(3.0 * self._t)]
        ])                                        # 3×1

        dq_null = P @ y                           # 3×1  (projected into null space)

        dq = dq_primary + dq_null                 # 3×1  total

        dq = scale_velocities(dq, MAX_VEL)

        # --- Publish command ---
        cmd = Float64MultiArray()
        cmd.data = dq.flatten().tolist()
        self.pub_cmd.publish(cmd)

        # --- Publish EE position ---
        ee_msg = PointStamped()
        ee_msg.header.stamp = self.get_clock().now().to_msg()
        ee_msg.header.frame_id = 'turtlebot/base_footprint'
        ee_msg.point.x, ee_msg.point.y, ee_msg.point.z = p[0], p[1], p[2]
        self.pub_ee.publish(ee_msg)

        self.get_logger().info(
            f'|XY err|={np.linalg.norm(err):.4f} m  '
            f'q=[{self.arm.q[0]:.3f},{self.arm.q[1]:.3f},{self.arm.q[2]:.3f}]',
            throttle_duration_sec=0.5)

        self.arm.integrate(dq.flatten(), DT)


def main(args=None):
    rclpy.init(args=args)
    node = Lab3NullSpaceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
