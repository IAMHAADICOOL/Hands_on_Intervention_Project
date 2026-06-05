# Hands-on Intervention (HOI) Project

A ROS 2 simulation project implementing Task-Priority kinematic control for a **vehicle-manipulator system** consisting of a **Kobuki TurtleBot 2** differential-drive base and a **uArm Swift Pro** 4-DOF manipulator arm, simulated in [Stonefish](https://stonefish.readthedocs.io).

**Team:** Haadi / Huy / Phu

---

## System Overview

The physical system is a mobile manipulator with 6 controllable degrees of freedom:

```
ζ = [vx, ω, dq1, dq2, dq3, dq4]
     ────  ─────────────────────
     base        arm
```

| Component | Model | DOF |
|-----------|-------|-----|
| Mobile base | Kobuki TurtleBot 2 (differential drive) | 2 (linear velocity `v`, angular velocity `ω`) |
| Arm | uArm Swift Pro (parallelogram linkage) | 4 (`q1` base yaw, `q2` shoulder, `q3` elbow, `q4` EE yaw) |
| **Total** | VMS (Vehicle-Manipulator System) | **6** |

> **No Denavit-Hartenberg:** the uArm's parallelogram closed-chain linkage makes DH convention invalid. All kinematics are derived geometrically. The parallelogram constraint enforces `q4 = -(q2 + q3)`, keeping the wrist permanently horizontal and reducing independent position DOF to 3 on the arm alone.

---

## Package Structure

```
src/
├── hoi_control/              ← Main control package (see below)
├── kobuki_description/       ← URDF + meshes for the Kobuki base
├── swiftpro_description/     ← URDF + meshes for the uArm Swift Pro
├── turtlebot_description/    ← URDF for the combined TurtleBot robot
├── turtlebot_simulation_1/   ← Stonefish scenarios + simulation launch files
├── turtlebot_rviz/           ← RViz visualisation launch
├── stonefish_ros2/           ← ROS 2 bridge for the Stonefish simulator
├── scan_to_cloud2/           ← Utility: converts LaserScan → PointCloud2
└── tutorial_interfaces/      ← Custom service definitions (Pose.srv)
```

---

## `hoi_control` — Main Package

The core of the project. Contains all kinematics, task definitions, and ROS 2 control nodes.

### Kinematics Libraries

#### `swiftpro_robotics.py` — Arm-only kinematics

| Symbol | Description |
|--------|-------------|
| `swiftpro_fk(q)` | Geometric forward kinematics → EE position in arm-local ENU |
| `swiftpro_jacobian(q)` | 3×3 geometric Jacobian `[dq1, dq2, dq3] → [dx, dy, dz]` |
| `swiftpro_ik(p)` | Analytical closed-form inverse kinematics |
| `SwiftProManipulator` | 3-DOF state container with FK / Jacobian queries |
| `MobileManipulator` | 5-DOF model combining the differential-drive base + arm |

#### `swiftpro_robotics_rrc.py` — VMS kinematics (RRC variant)

| Symbol | Description |
|--------|-------------|
| `swiftpro_fk_with_tf_transform` | FK with live NED→ENU frame conversion via TF |
| `swiftpro_jacobian_vms_5dof` | 3×5 VMS Jacobian `[vx, ω, dq1, dq2, dq3] → [dx, dy, dz]` |
| `swiftpro_jacobian_vms_6dof` | 4×6 VMS Jacobian including EE yaw row |
| `SwiftProManipulator4DOF` | 4-DOF state container (adds `q4` EE yaw) |
| `VMSRobotState` | Full 6-DOF VMS state container for task-priority control |

#### Coordinate Frames

The project uses two frames throughout:

- **NED** (North-East-Down) — native Stonefish simulator frame, `z` points down
- **ENU** (East-North-Up) — ROS navigation and RViz frame, `z` points up

Helper functions `ned_to_enu` / `enu_to_ned` handle all conversions. TF is used to look up live transforms between `world_ned` and `world_enu`.

### Task Classes

Both kinematics libraries define a hierarchy of task types that plug directly into the Task-Priority solver:

| Task | Dimension | Description |
|------|-----------|-------------|
| `Position3D` | 3×N | 3-D EE position error |
| `Position2D_XY` | 2×N | Horizontal XY position only |
| `HeightTask` | 1×N | EE height (Z) control |
| `YawTask` | 1×N | EE yaw angle (`q1`) |
| `Configuration3D` | 4×N | Combined position + yaw |
| `JointPosition` | 1×N | Drive a single joint to a target angle |
| `JointLimits` | 1×N | Inequality task — keep joint within safe limits |
| `Obstacle3D` | 1×N | Inequality task — 3-D spherical obstacle avoidance |
| `VMSPositionTask` | 3×6 | Position task in 6-DOF VMS quasi-velocity space |
| `VMSYawTask` | 1×6 | EE yaw task in VMS |
| `VMSConfigurationTask` | 4×6 | Position + yaw in VMS |
| `VMSJointLimitsTask` | 1×6 | Joint limit avoidance in VMS |
| `VMSObstacleTask` | 1×6 | Spherical obstacle avoidance in VMS (cylindrical option) |
| `VMSBaseOrientationTask` | 1×6 | Drive base heading to a desired angle |
| `VMSJointCenteringTask` | 4×6 | Null-space joint re-centering between goals |

### Task-Priority Solver

```python
# Arm-only (3 or 4 DOF)
zeta = task_priority_step(tasks, robot, damping=0.1)

# VMS (6 DOF)
zeta = vms_task_priority_step(tasks, state, damping=0.1, method=2)
# method: 0=Jacobian transpose, 1=Moore-Penrose pinv, 2=DLS (default)
```

The solver iterates from highest to lowest priority. Each task executes in the **null-space** of all higher-priority tasks, so higher-priority tasks are never disturbed. Inactive inequality tasks (joint limits, obstacles) are skipped automatically.

**Numerical stability:** Damped Least-Squares (DLS) is used instead of the plain pseudo-inverse to remain well-conditioned near singular arm configurations.

### Lab Nodes

Each node is a self-contained ROS 2 node demonstrating one control concept, progressively building complexity:

| Node | Algorithm | Robot model |
|------|-----------|-------------|
| `lab2_kinematics_node.py` | Forward kinematics, constant joint velocities | Arm only (3 DOF) |
| `lab2_rrc_node.py` | Resolved-Rate Control — EE tracks a 3-D target | Arm only (3 DOF) |
| `lab2_rrc_methods_vms_node.py` | RRC with VMS, multiple inverse methods | VMS (6 DOF) |
| `lab2_pick_place_vms_node.py` | Autonomous pick-and-place via ArUco detection + FSM | VMS (6 DOF) |
| `lab3_null_space_node.py` | Null-space motion — joints move, EE stays fixed | Arm only |
| `lab3_two_tasks_node.py` | Two-task priority (switchable case a/b) | Arm only |
| `lab4_tp_node.py` | Full recursive Task-Priority, 4 configurable hierarchies | Arm only |
| `lab5_joint_limits_node.py` | Joint limit avoidance (inequality tasks) | Arm only |
| `lab5_obstacle_node.py` | 3-D spherical obstacle avoidance | Arm only |
| `lab6_mobile_manip_node.py` | Full 5-DOF mobile manipulator Task-Priority | MobileManipulator |

#### Pick-and-Place FSM (`lab2_pick_place_vms_node.py`)

The most complete node. Implements a finite-state machine for autonomous box retrieval using a camera-detected ArUco marker:

```
SEARCH → ALIGN_DIST → ALIGN_ANGLE → APPROACH_BOX_VMS → PICK_DESCEND
→ SUCTION_ON → PICK_ASCEND → NAVIGATE_TO_GOAL → PLACE_VMS_APPROACH
→ PLACE_DESCEND → SUCTION_OFF → PLACE_ASCEND → DONE
```

### ROS Topics

| Topic | Type | Direction | Description |
|-------|------|-----------|-------------|
| `/turtlebot/joint_states` | `sensor_msgs/JointState` | Subscribe | Arm joint positions from simulator |
| `/turtlebot/swiftpro/joint_velocity_controller/command` | `std_msgs/Float64MultiArray` | Publish | Arm joint velocity commands `[dq1, dq2, dq3, dq4]` |
| `/turtlebot/cmd_vel` | `geometry_msgs/Twist` | Publish | Base linear / angular velocity |
| `/turtlebot/odom` | `nav_msgs/Odometry` | Subscribe | Base odometry (mobile manipulator nodes) |
| `/hoi/ee_position` | `geometry_msgs/PointStamped` | Publish | Current EE position |
| `/hoi/markers` | `visualization_msgs/MarkerArray` | Publish | RViz target/error visualisation |

### Launch

A single launch file starts both the Stonefish simulation and one selected control node:

```bash
# Syntax
ros2 launch hoi_control hoi_control.launch.py node:=<node_name> [hierarchy:=<a|b|c|d>] [case:=<a|b>]

# Examples
ros2 launch hoi_control hoi_control.launch.py node:=lab2_rrc
ros2 launch hoi_control hoi_control.launch.py node:=lab4_tp hierarchy:=c
ros2 launch hoi_control hoi_control.launch.py node:=lab3_two_tasks case:=a
ros2 launch hoi_control hoi_control.launch.py node:=lab6_mobile_manip
```

Available `node` values: `lab2_kinematics`, `lab2_rrc`, `lab3_two_tasks`, `lab3_null_space`, `lab4_tp`, `lab5_joint_limits`, `lab5_obstacle`, `lab6_mobile_manip`.

---

## Dependencies

| Dependency | Purpose |
|------------|---------|
| ROS 2 (Humble or later) | Middleware |
| [Stonefish](https://github.com/patrykcieslak/stonefish) + `stonefish_ros2` | Physics simulation |
| `tf2_ros`, `tf2_geometry_msgs` | Frame transforms (NED ↔ ENU) |
| `rclpy`, `sensor_msgs`, `geometry_msgs`, `nav_msgs`, `visualization_msgs` | ROS 2 standard libraries |
| NumPy | All linear algebra |
| OpenCV + `cv_bridge` | ArUco marker detection (pick-and-place node) |

---

## Building

```bash
cd <workspace>
colcon build --packages-select hoi_control
source install/setup.bash
```
