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

There are two pick-and-place node variants — one for simulation (`_node_2.py`) and one for the real robot (`_node.py`) — each with a corresponding launch file.

### Simulation (Stonefish)

Three terminals are required.

**Terminal 1 — Simulation**
```bash
source install/setup.bash
ros2 launch turtlebot_simulation_1 turtlebot_hoi.launch.py
```

**Terminal 2 — Pick-and-place controller (simulation node)**
```bash
source install/setup.bash
ros2 launch hoi_control pick_place_node_only_sim.launch.py
```

This launches `lab2_pick_place_vms_node_2.py`. All TF frames (`world_enu`, `base_footprint`, `end_effector`) are provided by the simulator, so no static TF publishers are needed.

---

### Real Robot

**Terminal 1 — Pick-and-place controller (real robot node)**
```bash
source install/setup.bash
ros2 launch hoi_control pick_place_node_only.launch.py
```

This launches `lab2_pick_place_vms_node.py` together with three static TF publishers that the real robot requires but the simulator provides automatically:

| Static TF publisher | Transform |
|---------------------|-----------|
| `odom_to_world_enu_broadcaster` | `odom` → `world_enu` (identity) |
| `enu_to_ned_broadcaster` | `world_enu` → `world_ned` (π/2 yaw, π roll) |
| `ee_broadcaster` | `swiftpro/link8B` → `end_effector` (7.22 cm Z offset, π yaw) |

RViz is also started automatically using the `turtlebot.rviz` config.

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

| Node | Target | Purpose |
|------|--------|---------|
| `lab2_pick_place_vms_node_2.py` | **Simulation** | Autonomous ArUco pick-and-place via VMS Task-Priority + FSM (Stonefish) |
| `lab2_pick_place_vms_node.py` | **Real robot** | Same FSM adapted for ros2_control, real camera, GPIO pump, live intrinsics |
| `lab2_rrc_methods_vms_node.py` | Both | VMS resolved-rate control to a fixed target; tunable via ROS params |
| `lab2_rrc_node.py` | Both | Arm-only resolved-rate control |
| `lab2_rrc_debug_node.py` | Both | RRC debug node — TF-based error, RViz markers, runtime param tuning |
| `lab2_rrc_methods_debug_node.py` | Both | 4-DOF RRC debug with runtime param tuning and auto goal cycling |

#### Pick-and-Place FSM (shared by both nodes)

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

---

### Simulation vs Real Robot — Code Differences

The two pick-and-place nodes share the same FSM structure and VMS controller. All differences exist to adapt the node to the real robot's hardware interface, sensor pipeline, and timing constraints.

#### 1. Kinematics library

| | Simulation (`_node_2.py`) | Real robot (`_node.py`) |
|-|--------------------------|-------------------------|
| Import | `swiftpro_robotics_rrc_2` | `swiftpro_robotics_rrc` |

#### 2. Runtime flags (real robot only)

The real robot node has three top-level flags that select which hardware interface is used. These are absent from the simulation node.

```python
USE_DYNAMIC_JOINT_STATES = True   # ros2_control DynamicJointState vs Stonefish JointState
USE_REAL_PUMP            = True   # GPIO publisher vs SetBool service
USE_TF_FOR_EE            = True   # TF lookup (with FK fallback) vs TF always
```

#### 3. Topic and frame names

| | Simulation | Real robot |
|-|-----------|-----------|
| `JOINT_CMD_TOPIC` | `/turtlebot/swiftpro/joint_velocity_controller/command` | `/turtlebot/joint_velocity_controller/commands` |
| `CAMERA_TOPIC` | `/turtlebot/camera/color/image_color` | `/turtlebot/camera/color/image_raw` |
| `J1_FRAME` | `turtlebot/swiftpro/manipulator_base_link` | `swiftpro/manipulator_base_link` |
| `BASE_FRAME` | `turtlebot/base_footprint` | `base_footprint` |
| `DYNAMIC_JOINT_STATE_TOPIC` | *(absent)* | `/turtlebot/dynamic_joint_states` |
| `CAMERA_INFO_TOPIC` | *(absent)* | `/turtlebot/camera/color/camera_info` |
| `PUMP_CMD_TOPIC` | *(absent)* | `/turtlebot/gpio_controller/commands` |

#### 4. Control rate

| | Simulation | Real robot |
|-|-----------|-----------|
| `CONTROL_HZ` | 60 Hz | 30 Hz |

#### 5. Tuned geometry constants

| Constant | Simulation | Real robot | Note |
|----------|-----------|-----------|------|
| `EE_TOUCH_Z_OFFSET` | −0.003 m | +0.005 m | Real EE sits slightly above box surface |
| `EE_TOUCH_FORWARD_OFFSET` | 0.0 m | +0.06 m | Real arm needs forward bias for suction contact |
| `EE_REACH_TOL` | 0.005 m | 0.009 m | Looser tolerance for real-robot noise |
| `GOAL_BASE_X` | 0.0 m | −0.3 m | Different drop-off X coordinate |
| `GOAL_BASE_Y` | 1.8 m | 0.0 m | Different drop-off Y coordinate |
| `SEARCH_SWEEP_ANGLE_ALIGN` | 90° | 180° | Wider sweep for real-robot uncertainty |

#### 6. Joint state reading

- **Simulation:** `_js_cb` reads `JointState` directly from `/turtlebot/joint_states`.
- **Real robot:** `_js_cb` is a no-op when `USE_DYNAMIC_JOINT_STATES = True`. Instead, `_dynamic_js_cb` reads `DynamicJointState` from `/turtlebot/dynamic_joint_states` using the joint names `swiftpro/joint1..4` and extracts the `position` interface value.

#### 7. Camera intrinsics

- **Simulation:** `CAMERA_MATRIX` and `DIST_COEFFS` are fixed global constants (no distortion, 69° HFOV).
- **Real robot:** Defaults (`_DEFAULT_CAMERA_MATRIX`, `_DEFAULT_DIST_COEFFS`) are used until `_camera_info_cb` fires, at which point the live calibration from `/turtlebot/camera/color/camera_info` overwrites `self._camera_matrix` / `self._dist_coeffs`. This runs once and the subscription continues to exist but early-returns after the first message.

#### 8. PnP flip correction (real robot only)

After `cv2.solvePnP`, the real robot node checks whether the marker Z-axis points away from the camera (`mz_cam[2] > 0`). If so, it rotates the result 180° around the marker X-axis to fix the sign convention that real cameras sometimes return. The simulation node does not apply this correction.

#### 9. Marker freshness (`_marker_fresh`, real robot only)

The real robot node adds `MARKER_FRESHNESS_S = 0.5 s` and a `_marker_fresh()` helper. Any `rvec`/`tvec` older than this threshold is treated as if no marker is visible. The camera callback sets `_new_camera_frame = True` on every firing. This prevents the controller from reacting to stale data between slow real-camera frames. The simulation node has no such guard.

#### 10. ALIGN_ANGLE state logic

This is the most significant behavioural difference between the two nodes.

**Simulation (`_node_2.py`):** Continuous P-controller. Every control tick:
1. If marker visible: compute `center_angle`, send `omega = -K * center_angle`, hold until `|center_angle| < tol`.
2. If marker not visible: sweep through `SEARCH_SEQUENCE_ALIGN` waypoints.

**Real robot (`_node.py`):** Three-phase step-and-wait controller to handle real camera latency:
1. **Phase 1 — Rotating:** robot turns toward a previously computed `_align_step_target_psi`. If a new camera frame arrives mid-rotation, the target is updated immediately (adaptive correction).
2. **Phase 2 — Waiting:** robot is stopped; control loop idles until `_new_camera_frame` is set by `_camera_cb`.
3. **Phase 3 — Decide:** new frame consumed; if aligned (`|center_angle| < tol`) → transition to `APPROACH_BOX_VMS`; if not → compute next step (`ALIGN_STEP_DEG` cap far from center, exact error close to center) and set `_align_step_target_psi`.

The `ALIGN_STEP_DEG = 8°` cap prevents over-rotation on slow camera feeds.

#### 11. APPROACH_BOX_VMS state logic

| | Simulation | Real robot |
|-|-----------|-----------|
| Target update | Computed once on first tick, then fixed | Refreshed every tick from latest fresh detection |
| Box lock (`_box_locked`) | Set on first tick when target is computed | Set only when transitioning out to `PICK_DESCEND` |
| Transition trigger | EE position tolerance (`EE_REACH_TOL`) | Elapsed time ≥ `APPROACH_VMS_PATH_PERIOD` (60 s) |
| Path period | `VMS_PATH_PERIOD` (30 s) | `APPROACH_VMS_PATH_PERIOD` (60 s, passed explicitly) |

The time-based transition in the real robot node ensures the full approach window is always used, even if the EE reaches the target early.

#### 12. `_vms_nav_step` signature

- **Simulation:** `_vms_nav_step(target, weight_matrix=None)` — path period is always `VMS_PATH_PERIOD`.
- **Real robot:** `_vms_nav_step(target, weight_matrix=None, path_period=None)` — `path_period` defaults to `VMS_PATH_PERIOD` but `APPROACH_BOX_VMS` passes `APPROACH_VMS_PATH_PERIOD`.

#### 13. `_update_vms_state` — EE source

- **Simulation:** Always uses TF lookup for EE position.
- **Real robot:** Respects `USE_TF_FOR_EE`. If `True`, TF is tried first with FK as fallback. If `False`, FK is used exclusively (with a throttled log message).

#### 14. `_compute_align_target` — stand-off direction

| | Simulation | Real robot |
|-|-----------|-----------|
| Direction source | Marker Z from `rvec` (`R @ [0,0,1]`) | Camera-to-marker direction (`-tvec/|tvec|`) |
| Reason | rvec is stable in simulation | rvec is noisy on real cameras due to planar marker ambiguity; tvec direction is always stable |

#### 15. Suction / pump actuation

- **Simulation:** Always calls the `SetBool` service (`/turtlebot/swiftpro/vacuum_gripper/set_pump`).
- **Real robot:** If `USE_REAL_PUMP = True`, publishes a `DynamicInterfaceGroupValues` message to `/turtlebot/gpio_controller/commands` (interface group `swiftpro/pump`, value `1.0`/`0.0`). Falls back to the service if `USE_REAL_PUMP = False`.

---

### ROS Topics

#### Simulation (`lab2_pick_place_vms_node_2.py`)

| Topic | Type | Direction | Description |
|-------|------|-----------|-------------|
| `/turtlebot/joint_states` | `sensor_msgs/JointState` | Subscribe | Arm joint positions from Stonefish |
| `/turtlebot/swiftpro/joint_velocity_controller/command` | `std_msgs/Float64MultiArray` | Publish | Arm velocity commands `[dq1,dq2,dq3,dq4]` |
| `/turtlebot/cmd_vel` | `geometry_msgs/Twist` | Publish | Base linear / angular velocity |
| `/turtlebot/camera/color/image_color` | `sensor_msgs/Image` | Subscribe | Camera image for ArUco detection |
| `/turtlebot/swiftpro/vacuum_gripper/set_pump` | `std_srvs/SetBool` (service) | Client | Suction on/off |
| `/hoi/pick_place_markers` | `visualization_msgs/MarkerArray` | Publish | RViz target/error visualisation |

#### Real Robot (`lab2_pick_place_vms_node.py`)

| Topic | Type | Direction | Description |
|-------|------|-----------|-------------|
| `/turtlebot/joint_states` | `sensor_msgs/JointState` | Subscribe | Unused when `USE_DYNAMIC_JOINT_STATES=True` |
| `/turtlebot/dynamic_joint_states` | `control_msgs/DynamicJointState` | Subscribe | Arm joint positions from ros2_control |
| `/turtlebot/joint_velocity_controller/commands` | `std_msgs/Float64MultiArray` | Publish | Arm velocity commands `[dq1,dq2,dq3,dq4]` |
| `/turtlebot/cmd_vel` | `geometry_msgs/Twist` | Publish | Base linear / angular velocity |
| `/turtlebot/camera/color/image_raw` | `sensor_msgs/Image` | Subscribe | Camera image for ArUco detection |
| `/turtlebot/camera/color/camera_info` | `sensor_msgs/CameraInfo` | Subscribe | Live camera intrinsics (K, D) |
| `/turtlebot/gpio_controller/commands` | `control_msgs/DynamicInterfaceGroupValues` | Publish | Pump on/off via GPIO when `USE_REAL_PUMP=True` |
| `/turtlebot/swiftpro/vacuum_gripper/set_pump` | `std_srvs/SetBool` (service) | Client | Pump fallback when `USE_REAL_PUMP=False` |
| `/hoi/pick_place_markers` | `visualization_msgs/MarkerArray` | Publish | RViz target/error visualisation |

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
