#!/usr/bin/env python3
"""
lab4_tp_node.py
===============
ROS 2 equivalent of lab4_recursive_tp.py
-----------------------------------------
Full recursive Task-Priority algorithm with configurable task hierarchies,
randomised goals every 10 seconds, and support for all task types.

Task hierarchies (mirroring lab4 configurations a/b/c/d):
  a) [Position3D]
  b) [Position3D, HeightTask]       – analogous to config+orientation
  c) [Position3D, JointPosition]    – EE position + joint 1 hold
  d) [JointPosition, Position3D]    – joint 1 primary, EE secondary

Select via ROS parameter 'hierarchy': 'a', 'b', 'c', or 'd'.

Corresponds to: lab4_recursive_tp.py
Changes vs lab:
  - 3-D position / height tasks instead of 2-D
  - Geometric Jacobian
  - Randomised 3-D targets (within reachable workspace)
  - ROS interface

Run:
    ros2 run hoi_control lab4_tp_node.py
    ros2 run hoi_control lab4_tp_node.py --ros-args -p hierarchy:=c
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Float64MultiArray
import numpy as np

from hoi_control.swiftpro_robotics import (
    SwiftProManipulator, Position3D, HeightTask, JointPosition,
    task_priority_step, scale_velocities,
    L2, L3, L4, A1, D1   # link params for workspace sampling
)

DT       = 1.0 / 60.0
MAX_VEL  = 1.0
DAMPING  = 0.1
GOAL_PERIOD = 10.0          # seconds between random goal changes

JOINT_CMD_TOPIC   = '/turtlebot/swiftpro/joint_velocity_controller/command'
JOINT_STATE_TOPIC = '/turtlebot/joint_states'
EE_POS_TOPIC      = '/hoi/ee_position'
TARGET_TOPIC      = '/hoi/target_position'


def random_reachable_target():
    """
    Sample a random EE target inside the uArm's approximate reachable workspace.
    Returns a (3,1) column vector in NED frame.
    """
    # Sample q2, q3 within safe joint limits
    q2 = np.random.uniform(0.1, 1.2)
    q3 = np.random.uniform(-1.0, 0.5)
    q1 = np.random.uniform(-1.2, 1.2)

    from hoi_control.swiftpro_robotics import swiftpro_fk
    p = swiftpro_fk([q1, q2, q3])
    return p.reshape(3, 1)


class Lab4TPNode(Node):

    def __init__(self):
        super().__init__('lab4_tp_node')

        self.declare_parameter('hierarchy', 'a')
        hier = self.get_parameter('hierarchy').value
        self.get_logger().info(f'Lab4 Task-Priority node, hierarchy={hier}')

        self.arm = SwiftProManipulator()

        # --- Build task hierarchy matching lab4 configurations ---
        target = random_reachable_target()

        self._task_ee   = Position3D('EE position', target)
        self._task_ee.setGain(np.diag([2.0, 2.0, 2.0]))
        self._task_ee.setFF(np.zeros((3, 1)))

        self._task_h    = HeightTask('EE height', float(target[2]))
        self._task_h.setGain(np.array([[1.0]]))
        self._task_h.setFF(np.zeros((1, 1)))

        self._task_q1   = JointPosition('Joint1 pos', np.array([[0.0]]), joint_idx=0)
        self._task_q1.setGain(np.array([[2.0]]))
        self._task_q1.setFF(np.zeros((1, 1)))

        hierarchies = {
            'a': [self._task_ee],
            'b': [self._task_ee, self._task_h],
            'c': [self._task_ee, self._task_q1],
            'd': [self._task_q1, self._task_ee],
        }
        self._tasks = hierarchies.get(hier, hierarchies['a'])

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
        self._t = 0.0
        self._goal_t = 0.0

    def _js_cb(self, msg: JointState):
        self.arm.update_from_joint_states(list(msg.name), list(msg.position))
        self._got_js = True

    def _control_loop(self):
        if not self._got_js:
            return

        self._t      += DT
        self._goal_t += DT

        # --- Randomise target every GOAL_PERIOD seconds (same as lab4) ---
        if self._goal_t >= GOAL_PERIOD:
            self._goal_t = 0.0
            new_target = random_reachable_target()
            self._task_ee.setDesired(new_target)
            self._task_h.setDesired(np.array([[float(new_target[2])]]))
            self.get_logger().info(
                f'New target: [{new_target[0,0]:.3f}, {new_target[1,0]:.3f}, {new_target[2,0]:.3f}]')

        # --- Recursive Task-Priority (identical algorithm to lab4) ---
        zeta = task_priority_step(self._tasks, self.arm, damping=DAMPING)
        zeta = scale_velocities(zeta, MAX_VEL)

        # --- Publish ---
        cmd = Float64MultiArray()
        cmd.data = zeta.flatten().tolist() + [0.0]
        self.pub_cmd.publish(cmd)

        p = self.arm.getEEPosition()
        self._pub_point(self.pub_ee,  p,  'turtlebot/base_footprint')
        tgt = self._task_ee.getDesired().flatten()
        self._pub_point(self.pub_tgt, tgt, 'turtlebot/base_footprint')

        # Log task errors
        errs = []
        for task in self._tasks:
            if task.err is not None:
                errs.append(f'{task.name}={np.linalg.norm(task.err):.3f}')
        self.get_logger().info(' | '.join(errs), throttle_duration_sec=0.5)

        self.arm.integrate(zeta.flatten(), DT)

    def _pub_point(self, pub, p, frame):
        msg = PointStamped()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = frame
        msg.point.x = float(p[0])
        msg.point.y = float(p[1])
        msg.point.z = float(p[2])
        pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = Lab4TPNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
