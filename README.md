# Hands-on Intervention (HOI) Project

A ROS 2 simulation project implementing Task-Priority kinematic control for a **vehicle-manipulator system (VMS)** consisting of a **Kobuki TurtleBot 2** differential-drive base and a **uArm Swift Pro** 4-DOF manipulator arm, simulated in [Stonefish](https://stonefish.readthedocs.io).

The primary demonstration is an autonomous **pick-and-place** task: the robot searches for an ArUco-tagged box, navigates to it, picks it up with the suction-cup end-effector, carries it to a drop-off point, and releases it — all using Task-Priority resolved-rate control over the full 6-DOF VMS.

**Team:** Haadi / Huy / Phu

---

## System Overview

```
ζ = [vx, ω,  dq1, dq2, dq3, dq4]
     ──────  ──────────────────────
      base          arm
```

| Component | Model | DOF |
|-----------|-------|-----|
| Mobile base | Kobuki TurtleBot 2 (differential drive) | 2 (`v` linear, `ω` angular) |
| Arm | uArm Swift Pro (parallelogram linkage) | 4 (`q1` base yaw, `q2` shoulder, `q3` elbow, `q4` EE yaw) |
| **Total VMS** | | **6** |

> **No Denavit-Hartenberg:** the uArm's closed-chain parallelogram linkage makes DH convention invalid. All kinematics are derived geometrically. The parallelogram constraint enforces `q4 = -(q2 + q3)`, keeping the wrist permanently horizontal and reducing independent position DOF to 3 on the arm.

---

## Running

Two terminals are required: one for the simulation, one for the control node.

**Terminal 1 — Simulation**
```bash
source install/setup.bash
ros2 launch turtlebot_simulation_1 turtlebot_hoi.launch.py
```

**Terminal 2 — Pick-and-place controller**
```bash
source install/setup.bash
ros2 run hoi_control lab2_pick_place_vms_node.py
```

---

## Package Structure

```
src/
├── hoi_control/              ← Main control package
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

### Kinematics Libraries

#### `swiftpro_robotics.py` — Arm-only (3-DOF)

| Function / Class | Description |
|-----------------|-------------|
| `swiftpro_fk(q)` | Geometric FK → EE position in arm-local ENU |
| `swiftpro_jacobian(q)` | 3×3 geometric Jacobian `[dq1,dq2,dq3] → [dx,dy,dz]` |
| `swiftpro_ik(p)` | Closed-form analytical IK |
| `SwiftProManipulator` | 3-DOF state container with FK / Jacobian queries |
| `MobileManipulator` | 5-DOF model (differential-drive base + 3-DOF arm) |
| `ned_to_enu` / `enu_to_ned` | Frame conversion helpers |
| `DLS` / `weighted_DLS` | Damped Least-Squares pseudo-inverse |

#### `swiftpro_robotics_rrc.py` — VMS kinematics (6-DOF)

| Function / Class | Description |
|-----------------|-------------|
| `swiftpro_fk_with_tf_transform` | FK with live NED→ENU conversion via TF buffer |
| `swiftpro_fk_vms_5dof` | 5-DOF VMS FK: base (x,y,ψ) + arm (q1,q2,q3) → world EE position |
| `swiftpro_jacobian_vms_5dof` | 3×5 VMS Jacobian `[vx,ω,dq1,dq2,dq3] → [dx,dy,dz]` |
| `swiftpro_jacobian_vms_6dof` | 4×6 VMS Jacobian including EE yaw row |
| `SwiftProManipulator4DOF` | 4-DOF state container (adds `q4` EE yaw) |
| `VMSRobotState` | Full 6-DOF VMS state container for task-priority control |
| `DLS` / `weighted_DLS` / `scale_velocities` | Solver utilities |

#### Coordinate Frames

| Frame | Convention | Used for |
|-------|-----------|---------|
| NED (North-East-Down) | Z points down | Stonefish simulator native |
| ENU (East-North-Up) | Z points up | ROS nav topics, RViz, control loop |

The TF tree exposes both `world_ned` and `world_enu` frames. The RRC kinematics functions look up the live NED→ENU rotation via the TF buffer with a manual fallback (180° rotation around X).

### Task-Priority Solver

```python
# VMS (6 DOF) — used by the main node
zeta = vms_task_priority_step(tasks, state, damping=0.1, method=2)
# method: 0 = Jacobian transpose, 1 = Moore-Penrose pinv, 2 = DLS (default)
```

Each task executes in the **null-space** of all higher-priority tasks. Inactive inequality tasks (joint limits) are skipped automatically. DLS is used instead of the plain pseudo-inverse to remain numerically stable near singular arm configurations.

### Task Classes (VMS)

| Task | Jacobian | Description |
|------|----------|-------------|
| `VMSPositionTask` | 3×6 | 3-D EE position in world_enu |
| `VMSYawTask` | 1×6 | EE yaw `ψ + q1 + q4` |
| `VMSConfigurationTask` | 4×6 | Combined position + yaw |
| `VMSJointLimitsTask` | 1×6 | Inequality — keep one joint within URDF limits |
| `VMSJointPositionTask` | 1×6 | Drive one arm joint to a target angle |
| `VMSBaseOrientationTask` | 1×6 | Drive base heading ψ to a desired angle |
| `VMSJointCenteringTask` | 4×6 | Null-space re-centering of all arm joints |
| `VMSObstacleTask` | 1×6 | Spherical (or cylindrical) obstacle avoidance |
| `VMSYawQ4Task` | 1×6 | EE yaw via `q4` only (no base/arm coupling) |
| `VMSQ4ZeroTask` | 1×6 | Return `q4` to neutral after a yaw manoeuvre |

### Control Nodes

| Node | Purpose |
|------|---------|
| `lab2_pick_place_vms_node.py` | **Main node.** Autonomous ArUco pick-and-place via VMS Task-Priority + FSM |
| `lab2_rrc_methods_vms_node.py` | VMS resolved-rate control to a fixed target; tunable via ROS params |
| `lab2_rrc_node.py` | Arm-only resolved-rate control |
| `lab2_rrc_debug_node.py` | RRC debug node — TF-based error, RViz markers, runtime param tuning |
| `lab2_rrc_methods_debug_node.py` | 4-DOF RRC debug with runtime param tuning and auto goal cycling |

#### Pick-and-Place FSM (`lab2_pick_place_vms_node.py`)

```
SEARCH               rotate base, scan for ArUco marker
  ↓
ALIGN_DIST           drive base to stand-off point in front of marker
  ↓
ALIGN_ANGLE          rotate in place until face-on to marker
  ↓
APPROACH_BOX_VMS     VMS: drive EE to approach height above box
  ↓
PICK_DESCEND         arm-only: lower EE to box top (suction contact)
  ↓
SUCTION_ON           activate suction cup, wait for settle
  ↓
PICK_ASCEND          arm-only: lift EE back to approach height
  ↓
NAVIGATE_TO_GOAL     VMS: drive robot to drop-off location carrying box
  ↓
PLACE_VMS_APPROACH   VMS: drive EE to approach height above drop point
  ↓
PLACE_DESCEND        arm-only: lower box to floor
  ↓
SUCTION_OFF          deactivate suction, release box
  ↓
PLACE_ASCEND         arm-only: lift EE away from floor
  ↓
DONE
```

### ROS Topics

| Topic | Type | Direction | Description |
|-------|------|-----------|-------------|
| `/turtlebot/joint_states` | `sensor_msgs/JointState` | Subscribe | Arm joint positions from simulator |
| `/turtlebot/swiftpro/joint_velocity_controller/command` | `std_msgs/Float64MultiArray` | Publish | Arm velocity commands `[dq1,dq2,dq3,dq4]` |
| `/turtlebot/cmd_vel` | `geometry_msgs/Twist` | Publish | Base linear / angular velocity |
| `/hoi/rrc_methods_vms_markers` | `visualization_msgs/MarkerArray` | Publish | RViz target/error visualisation |

---

## Building

```bash
cd <workspace_root>
colcon build --packages-select hoi_control
source install/setup.bash
```

## Dependencies

| Dependency | Purpose |
|------------|---------|
| ROS 2 (Humble or later) | Middleware |
| [Stonefish](https://github.com/patrykcieslak/stonefish) + `stonefish_ros2` | Physics simulation |
| `tf2_ros`, `tf2_geometry_msgs` | Frame transforms (NED ↔ ENU) |
| `rclpy`, `sensor_msgs`, `geometry_msgs`, `nav_msgs`, `visualization_msgs` | ROS 2 standard libraries |
| NumPy | Linear algebra |
| OpenCV + `cv_bridge` | ArUco marker detection (pick-and-place node) |
