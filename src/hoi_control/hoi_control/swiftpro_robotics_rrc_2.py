import math
import numpy as np



from tf2_geometry_msgs import do_transform_point
from geometry_msgs.msg import Point, Vector3Stamped, TransformStamped
import rclpy
TF_AVAILABLE = True
from tf_transformations import quaternion_matrix
# uArm Swift Pro – geometric link parameters (metres)
# These constants were derived by tracing the URDF joint chain geometrically
# (no DH convention — invalid for closed-chain parallelogram linkages).
#
# The kinematic chain traced is:
#   link1 → joint2 → link2 → passive_joint1 → link3
#         → passive_joint7 → link8A → joint4 → link8B → end_effector
#
# L2 and L3 come directly from URDF joint origins:
#   passive_joint1: xyz="0 0 -0.142"  → L2 = 0.142 m (upper arm)
#   passive_joint7: xyz="0.1587 0 0"  → L3 = 0.1588 m (forearm, rounded)
#
# The base constants in swiftpro_fk (0.0127 for reach, -0.0382 for height)
# are NOT pure URDF values — they were empirically recalibrated by comparing
# FK output against TF ground truth at known joint configurations after
# switching the EE frame from link9 to end_effector (link8B + 0.0722m up).
# See swiftpro_fk docstring for the full derivation and calibration history. 
A1 = 0.0      # effective horizontal J1→J2 offset (negligible, from URDF joint2 x=0.0133)
D1 = 0.0333   # vertical height: J1 axis → J2 shoulder pivot (33.3 mm, from URDF joint2 z=-0.1056)
L2 = 0.142    # upper arm length  J2 → J3 pivot (from URDF passive_joint1 z=-0.142)
L3 = 0.1588   # forearm length    J3 → link8A  (from URDF passive_joint7 x=0.1587, rounded)
L4 = 0.0445   # wrist link; always horizontal due to parallelogram constraint

L_J4      = 0.0565   # link8A origin → joint4 pivot / link8B origin (m)
L_EE_TOOL = 0.0      # tool-tip extension beyond link8B origin along link8B x-axis (m)

# Joint angle limits (rad) — copied directly from the URDF <limit> tags.
# A small inset margin (LIMIT_MARGIN = 0.07 rad) is applied in the control node
# to stop joints slightly before the hard limit, giving the controller time
# to decelerate.
Q1_MIN = -1.571   # joint1 base yaw   lower (−π/2)
Q1_MAX =  1.571   # joint1 base yaw   upper (+π/2)
Q2_MIN = -1.571   # joint2 shoulder   lower (−π/2)
Q2_MAX =  0.05    # joint2 shoulder   upper
Q3_MIN = -1.571   # joint3 elbow      lower (−π/2)
Q3_MAX =  0.05    # joint3 elbow      upper
Q4_MIN = -1.571   # joint4 EE yaw     lower (−π/2)
Q4_MAX =  1.571   # joint4 EE yaw     upper (+π/2)

# Joint index mapping in /joint_states
JOINT_NAMES = [
    'turtlebot/swiftpro/joint1',   # index 0 – base yaw   (q1)
    'turtlebot/swiftpro/joint2',   # index 1 – shoulder   (q2)
    'turtlebot/swiftpro/joint3',   # index 2 – elbow      (q3)
]

# 4-DOF name list — same order as the q vector [q1, q2, q3, q4]
JOINT_NAMES_4DOF = JOINT_NAMES + ['turtlebot/swiftpro/joint4']


# Forward Kinematics with TF Buffer Transform
def swiftpro_fk_with_tf_transform(q, tf_buffer=None, source_frame='world_ned', target_frame='world_enu'):
    """
    Forward kinematics using TF buffer to look up coordinate frame transformations.
    
    This function computes the FK in NED frame, then uses the provided TF buffer
    to look up the transformation between source_frame and target_frame, and applies
    that transformation to convert the result.
    
    Arguments
    ---------
    q : array-like, shape (3,) or (4,)
        Joint angles [q1, q2, q3] or [q1, q2, q3, q4]
    
    tf_buffer : tf2_ros.Buffer, optional
        TF buffer containing the transform between source and target frames.
        If None, falls back to direct computation without TF.
    
    source_frame : str
        Source coordinate frame (default: 'world_ned')
    
    target_frame : str
        Target coordinate frame (default: 'world_enu')
    
    Returns
    -------
    p : np.ndarray, shape (3,)
        [x, y, z] displacement from J1 in target_frame-aligned axes.
    """
    q1, q2, q3 = float(q[0]), float(q[1]), float(q[2])
    
    # Compute position in NED frame
    # This is the intermediate representation before frame transformation
    # q1_eff = q1 + np.pi / 2
    q1_eff = q1 - np.pi
    r = 0.0127 - L2 * np.sin(q2) + L3 * np.cos(q3)
    z_enu = -0.0382 + L2 * np.cos(q2) + L3 * np.sin(q3)
    r_ee = r + L_J4
    z_ee = z_enu + 0.0722
    
    # In NED frame
    x_ned = r_ee * np.cos(q1_eff)     # North component in NED
    y_ned = r_ee * np.sin(q1_eff)     # East component in NED
    z_ned = -z_ee                      # Down component in NED
    
    pos_ned = np.array([x_ned, y_ned, z_ned])
    

    if tf_buffer is not None and TF_AVAILABLE:
        try:
            
            transform = tf_buffer.lookup_transform(
                target_frame, 
                source_frame, 
                rclpy.time.Time()
            )
            
            # Extract rotation matrix from the quaternion in the transform

            quat = transform.transform.rotation
            q_array = [quat.x, quat.y, quat.z, quat.w]
            transform_matrix = quaternion_matrix(q_array)
            rotation_matrix = transform_matrix[:3, :3]
            
            # Apply the rotation to convert from NED to ENU
            pos_transformed = rotation_matrix @ pos_ned
            
            return pos_transformed
            
        except Exception as e:
            # If lookup fails, fall back to manual rotation
            print(f"TF lookup failed: {e}. Using manual rotation.")
            # Apply manual 180° rotation around x-axis
            rotation_matrix = np.array([
                [1.0,  0.0,   0.0],
                [0.0, -1.0,   0.0],
                [0.0,  0.0,  -1.0],
            ])
            pos_transformed = rotation_matrix @ pos_ned
            return pos_transformed
    else:
        # Fallback: apply manual rotation matrix (180° around x-axis)
        # NED to ENU: [1, 0, 0; 0, -1, 0; 0, 0, -1]
        rotation_matrix = np.array([
            [1.0,  0.0,   0.0],
            [0.0, -1.0,   0.0],
            [0.0,  0.0,  -1.0],
        ])
        pos_transformed = rotation_matrix @ pos_ned
        return pos_transformed

# ---------------------------------------------------------------------------
# Geometric Forward Kinematics
# ---------------------------------------------------------------------------

# 3-DOF Geometric Jacobian with TF Buffer Transform
def swiftpro_jacobian_with_tf_transform(q, tf_buffer=None, source_frame='world_ned', target_frame='world_enu'):
    """
    Compute the 3-DOF position Jacobian by calculating in NED frame, then
    transforming to target frame using TF buffer or fallback rotation.
    
    This function mirrors swiftpro_fk_with_tf_transform: it computes the Jacobian
    in NED frame, then applies the frame transformation via TF or manual rotation.
    
    Mathematical basis:
    If p_target = R @ p_ned (position transform), then:
        J_target = R @ J_ned (velocity/Jacobian transform)
    
    Arguments
    ---------
    q : array-like, shape (3,) or (4,) - only first 3 elements used.
    
    tf_buffer : tf2_ros.Buffer, optional
        TF buffer containing the transform between source and target frames.
        If None, falls back to manual NED→ENU rotation.
    
    source_frame : str
        Source coordinate frame (default: 'world_ned')
    
    target_frame : str
        Target coordinate frame (default: 'world_enu')

    Returns
    -------
    J : np.ndarray, shape (3, 3)
        Rows → [dx, dy, dz] in target_frame-aligned axes.
        Columns → [dq1, dq2, dq3].
    """
    q1, q2, q3 = float(q[0]), float(q[1]), float(q[2])

    # Compute Jacobian in NED frame using q1_eff = q1 - π
    # (matching swiftpro_fk_with_tf_transform convention)
    q1_eff = q1 - np.pi
    s1_ned, c1_ned = np.sin(q1_eff), np.cos(q1_eff)

    s2, c2 = np.sin(q2), np.cos(q2)
    s3, c3 = np.sin(q3), np.cos(q3)

    r      = 0.0127 - L2 * s2 + L3 * c3   # reach to link8A
    r_ee   = r + L_J4                       # total reach including joint4 offset

    # Derivatives of r_ee and z_ee with respect to joint angles
    dr_ee_dq2 = -L2 * c2
    dr_ee_dq3 = -L3 * s3
    dz_ee_dq2 = -L2 * s2
    dz_ee_dq3 =  L3 * c3

    # Construct NED Jacobian:
    # ───────────────────────
    # x_ned = r_ee * cos(q1_eff)
    # y_ned = r_ee * sin(q1_eff)
    # z_ned = -z_ee
    J_ned = np.array([
        # ∂x_ned/∂qi:
        [ -r_ee * s1_ned,    c1_ned * dr_ee_dq2,    c1_ned * dr_ee_dq3 ],
        # ∂y_ned/∂qi:
        [  r_ee * c1_ned,    s1_ned * dr_ee_dq2,    s1_ned * dr_ee_dq3 ],
        # ∂z_ned/∂qi: (z_ned = -z_ee, so ∂z_ned/∂qi = -∂z_ee/∂qi)
        [  0.0,             -dz_ee_dq2,             -dz_ee_dq3         ],
    ])

    # Transform Jacobian from NED to target frame
    if tf_buffer is not None and TF_AVAILABLE:
        try:
            transform = tf_buffer.lookup_transform(
                target_frame, 
                source_frame, 
                rclpy.time.Time()
            )
            
            # Extract rotation matrix from the quaternion
            quat = transform.transform.rotation
            q_array = [quat.x, quat.y, quat.z, quat.w]
            transform_matrix = quaternion_matrix(q_array)
            rotation_matrix = transform_matrix[:3, :3]
            
        except Exception as e:
            # If lookup fails, use fallback rotation
            print(f"TF lookup failed: {e}. Using fallback rotation for Jacobian.")
            rotation_matrix = np.array([
                [1.0,  0.0,   0.0],
                [0.0, -1.0,   0.0],
                [0.0,  0.0,  -1.0],
            ])
    else:
        # Fallback: NED to ENU rotation (180° around x-axis)
        rotation_matrix = np.array([
            [1.0,  0.0,   0.0],
            [0.0, -1.0,   0.0],
            [0.0,  0.0,  -1.0],
        ])

    # Apply rotation: J_target = R @ J_ned
    J_target = rotation_matrix @ J_ned

    return J_target


def swiftpro_jacobian_pos4_with_tf_transform(q, tf_buffer=None, source_frame='world_ned', target_frame='world_enu'):
    """
    3×4 position Jacobian with TF frame transform, including q4 EE yaw column.

    Mirrors swiftpro_jacobian_pos4 but uses swiftpro_jacobian_with_tf_transform
    for the first three columns so the result is expressed in target_frame.

    Arguments
    ---------
    q : array-like, shape (4,) – [q1, q2, q3, q4]
    tf_buffer : tf2_ros.Buffer, optional
    source_frame : str  (default: 'world_ned')
    target_frame : str  (default: 'world_enu')

    Returns
    -------
    J : np.ndarray, shape (3, 4)
    """
    q1, q4 = float(q[0]), float(q[3])
    J3 = swiftpro_jacobian_with_tf_transform(q, tf_buffer=tf_buffer,
                                              source_frame=source_frame,
                                              target_frame=target_frame)  # (3, 3)

    # 4th column: zero when L_EE_TOOL=0 (q4 is pure orientation, no positional effect)
    angle_j4 = q1 + q4
    col4 = np.array([
        L_EE_TOOL * np.sin(angle_j4),
        L_EE_TOOL * np.cos(angle_j4),
        0.0
    ])

    return np.hstack([J3, col4.reshape(3, 1)])   # (3, 4)


def swiftpro_jacobian_full4_with_tf_transform(q, tf_buffer=None, source_frame='world_ned', target_frame='world_enu'):
    """
    4×4 full Jacobian with TF frame transform mapping [dq1, dq2, dq3, dq4] → [dx, dy, dz, d(yaw)].

    Use this for combined position + EE yaw control when the NED→ENU (or any
    custom) frame transform must be applied via TF.

    The yaw row [1, 0, 0, 1] is frame-independent: EE yaw = q1 + q4 is a
    kinematic identity that holds regardless of coordinate frame.

    When using this Jacobian the error vector must be 4×1: [ex, ey, ez, e_yaw].
    Always wrap e_yaw to [−π, π] before use.

    Arguments
    ---------
    q : array-like, shape (4,)
    tf_buffer : tf2_ros.Buffer, optional
    source_frame : str  (default: 'world_ned')
    target_frame : str  (default: 'world_enu')

    Returns
    -------
    J : np.ndarray, shape (4, 4)
    """
    J_pos4  = swiftpro_jacobian_pos4_with_tf_transform(q, tf_buffer=tf_buffer,
                                                        source_frame=source_frame,
                                                        target_frame=target_frame)  # (3, 4)
    yaw_row = np.array([[1.0, 0.0, 0.0, 1.0]])       # dyaw/d[q1,q2,q3,q4]
    return np.vstack([J_pos4, yaw_row])               # (4, 4)


# 5-DOF VMS: Differential Drive Base (vx, vyaw) + 3-DOF Arm (q1, q2, q3)
# State: q = [x, y, psi, q1, q2, q3] (position, yaw, arm angles)
# Quasi-velocities: ζ = [vx, vyaw, dq1, dq2, dq3] (body-frame linear, yaw, joint vels)
# Jacobian: 3×5 (EE position derivative w.r.t. 5 DOF)


def swiftpro_fk_vms_5dof(j1_world, arm_q, base_yaw, tf_buffer):
    """
    EE position in world_enu for the VMS (Turtlebot + uArm Swift Pro).

    swiftpro_fk_with_tf_transform uses q1_eff = q1 − π internally, which
    is calibrated for q1 measured in world frame (from NED North axis).
    The robot's q1, however, is measured relative to the base heading ψ.
    To reconcile: subtract base_yaw from q1 before calling FK so that the
    modified q1 is world-frame-relative:

        q_mod[0] = q1 − ψ
        q1_eff   = q_mod[0] − π  =  q1 − ψ − π  (inside FK)

    The FK then computes the EE-from-J1 vector in NED and the TF buffer
    (or manual fallback rotation) converts it to world_enu.
    Adding j1_world gives the absolute EE position.

    Parameters
    ----------
    j1_world : (3,) J1 (link1) position in world_enu — from TF.
    arm_q    : (3,) or (4,) arm joint angles [q1, q2, q3, ...].
    base_yaw : current base yaw ψ in world_enu (rad).
    tf_buffer: tf2_ros.Buffer for NED→ENU frame transform; None uses fallback.

    Returns
    -------
    p_ee : (3,) EE position in world_enu.
    """
    q_mod    = np.array(arm_q, dtype=float)
    q_mod[0] -= (base_yaw)   # correct q1 for current base yaw
    return np.asarray(j1_world, dtype=float) + swiftpro_fk_with_tf_transform(q_mod, tf_buffer=tf_buffer)

def swiftpro_jacobian_vms_5dof(ee_world, base_world, base_yaw, arm_q, tf_buffer):
    """
    3×5 VMS position Jacobian in world_enu.

    Maps quasi-velocities ζ = [vx, ω, dq1, dq2, dq3] -> EE linear velocity.

    Geometric column derivation:

      col_vx  : body x-axis in world = [cos ψ, sin ψ, 0]ᵀ
                Prismatic DOF: translating the base moves the EE along the
                current heading direction.

      col_ω   : z_B × (p_EE − p_BASE) = [−Δy, Δx, 0]ᵀ
                Revolute DOF around the vertical axis at the BASE centre
                (base_footprint), NOT at J1/link1.

      col_dqi : arm position Jacobian (3×3) from swiftpro_jacobian_with_tf_transform,
                evaluated at q_mod[0] = q1 − ψ.
                Subtracting base_yaw from q1 converts the arm's base-relative
                joint angle into a world-frame-relative angle, which is what
                swiftpro_jacobian_with_tf_transform expects.  The TF buffer (or
                manual fallback rotation) then converts the NED Jacobian columns
                to world_enu, giving correct ∂p/∂qi at the current base yaw ψ.

    Parameters
    ----------
    ee_world  : (3,) current EE position in world_enu (TF or FK).
    base_world: (3,) or (2,) base centre (base_footprint) in world_enu.
    base_yaw  : current base yaw ψ in world_enu (rad).
    arm_q     : (3,) or (4,) arm joint angles [q1, q2, q3, ...].
    tf_buffer : tf2_ros.Buffer for NED→ENU frame transform; None uses fallback.

    Returns
    -------
    J : (3, 5) Jacobian — columns [vx, ω, dq1, dq2, dq3].
    """
    bx, by = float(base_world[0]), float(base_world[1])
    ex, ey = float(ee_world[0]),   float(ee_world[1])

    # Column 0: vx (prismatic along body x-axis in world) 
    col_vx = np.array([math.cos(base_yaw), math.sin(base_yaw), 0.0])

    # Column 1: ω (revolute around body z at BASE centre) 
    # jacobianLink pattern: col = z_B × (p_EE − p_B), z_B=[0,0,1]
    col_omega = np.array([-(ey - by), (ex - bx), 0.0])

    # Columns 2–4: arm joints
    # Substitute q_mod[0] = q1 − ψ — same world-frame correction as FK.
    # swiftpro_jacobian_with_tf_transform then gives ∂p/∂qi in world_enu at current ψ.
    q_mod    = np.array(arm_q, dtype=float)
    q_mod[0] -= (base_yaw)
    J_arm_world = swiftpro_jacobian_with_tf_transform(q_mod, tf_buffer=tf_buffer)  # (3, 3) correct at current ψ

    return np.column_stack([col_vx, col_omega, J_arm_world])   # (3, 5)


def swiftpro_jacobian_vms_6dof(ee_world, base_world, base_yaw, arm_q, tf_buffer):
    """
    4×6 VMS Jacobian for combined position + EE yaw control.

    Extends swiftpro_jacobian_vms_6dof by adding a q4 column and a yaw row.

    ζ = [vx, ω, dq1, dq2, dq3, dq4] → [dx, dy, dz, d(yaw)]

    q4 position column: [0, 0, 0]ᵀ  (L_EE_TOOL = 0, pure orientation DOF)

    Yaw row derivation:
        EE yaw in world_enu = base_yaw + q1 + q4
        d(yaw)/d[vx, ω, dq1, dq2, dq3, dq4] = [0, 1, 1, 0, 0, 1]
    """
    J_pos5 = swiftpro_jacobian_vms_5dof(
        ee_world, base_world, base_yaw, arm_q, tf_buffer)       # (3, 5)

    J_pos6  = np.hstack([J_pos5, np.zeros((3, 1))])             # (3, 6)  q4 col = 0
    yaw_row = np.array([[0.0, 1.0, 1.0, 0.0, 0.0, 1.0]])       # (1, 6)
    return np.vstack([J_pos6, yaw_row])                         # (4, 6)


# Damped Least-Squares pseudo-inverse
def DLS(A, damping):
    """
    Damped Least-Squares pseudo-inverse.
        A_dls = A^T (A A^T + lambda^2*I)^(-1)

    Preferred over the plain Moore-Penrose pseudo-inverse near singular
    configurations because it trades a small amount of accuracy for numerical
    stability.  Near singularities, A·A^T becomes ill-conditioned and its
    plain inverse explodes — adding lambda^2·I ensures the matrix is always
    invertible.

    Trade-off:
      larger lambda → smoother motion, larger steady-state position error.
      smaller lambda → more accurate, approaches plain pseudo-inverse behaviour.
      lambda = 0     → exactly equivalent to plain pseudo-inverse.
    """
    lam2 = damping ** 2
    m    = A.shape[0]
    return A.T @ np.linalg.inv(A @ A.T + lam2 * np.eye(m))


def weighted_DLS(A, damping, W):
    """
    Weighted Damped Least-Squares pseudo-inverse.
        A_wdls = W^(-1) Aᵀ (A W^(-1) Aᵀ + lambda^2*I)^(-1)

    W is a positive-definite weight matrix that penalises certain joints more
    heavily.  For example, a diagonal W with joint velocity limits on the
    diagonal prevents fast joints from dominating the solution.
    """
    lam2  = damping ** 2
    m     = A.shape[0]
    W_inv = np.linalg.inv(W)
    return W_inv @ A.T @ np.linalg.inv(A @ W_inv @ A.T + lam2 * np.eye(m))


def scale_velocities(zeta, max_vel):
    """
    Scale the joint velocity vector uniformly so that no element exceeds
    max_vel in absolute value.  Preserves the direction of motion — all
    joints slow down by the same factor so the arm still moves toward the
    target, just slower.

    If the maximum absolute value in zeta is already ≤ max_vel, no scaling
    is applied and zeta is returned unchanged.
    """
    s = np.max(np.abs(zeta)) / max_vel
    if s > 1.0:
        zeta = zeta / s
    return zeta



#  SwiftProManipulator4DOF  –  4-DOF kinematic state manager
class SwiftProManipulator4DOF:
    """
    Kinematic state manager for the uArm Swift Pro with 4 active joints.

    Wraps the four active joint angles [q1, q2, q3, q4] and exposes FK and
    Jacobian queries needed by the RRC control loop.  Updated each control
    tick from the /joint_states topic via update_from_joint_states().

    Joint roles:
      q1 — base yaw:     rotates the whole arm around the vertical axis
      q2 — shoulder:     tilts the upper arm up/down
      q3 — elbow:        tilts the forearm (but due to parallelogram, this is
                         independent of q2 in world_enu — see FK docstring)
      q4 — EE yaw:       rotates the end-effector about the vertical axis

    Usage inside a ROS node:
        arm = SwiftProManipulator4DOF()
        arm.update_from_joint_states(msg.name, msg.position)
        p   = arm.getEEPosition()       # (3,) EE position relative to J1
        J   = arm.getEEJacobianPos()   # (3, 4) position Jacobian
        J4  = arm.getEEJacobianFull()  # (4, 4) position + yaw Jacobian
        yaw = arm.getEEYaw()           # scalar EE yaw = q1 + q4
    """

    DOF = 4   # q1, q2, q3, q4 (all active)

    def __init__(self, q0=None, tf_buffer=None):
        """
        q0        : initial joint angles [q1, q2, q3, q4] (rad). Defaults to zeros.
        tf_buffer : tf2_ros.Buffer for frame transforms. If None, falls back to
                    manual NED→ENU rotation inside the TF-variant kinematics calls.
        """
        self.q         = np.zeros(4) if q0 is None else np.array(q0, dtype=float)
        self.tf_buffer = tf_buffer

    #  State update                                                        
    def update_from_joint_states(self, names, positions):
        """
        Parse a sensor_msgs/JointState message and extract q1-q4.

        Joints are matched by name (not by index) so the ordering in the
        JointState message does not matter.  If joint4 is not present in the
        message (e.g. the simulator does not publish it), q4 retains its
        previous value.

        Arguments
        ---------
        names     : list[str]   - JointState.name
        positions : list[float] - JointState.position  (same length as names)
        """
        pos_map = dict(zip(names, positions))
        for i, jn in enumerate(JOINT_NAMES_4DOF):
            if jn in pos_map:
                self.q[i] = pos_map[jn]

    def integrate(self, dq, dt):
        """
        Forward-Euler integration: q <- q + dq · dt.

        Used when the control node maintains its own internal joint angle
        estimate (e.g. when /joint_states is unavailable or delayed).
        In normal operation, update_from_joint_states() is preferred because
        it uses the simulator's ground-truth joint positions.
        """
        self.q = self.q + np.array(dq, dtype=float).flatten()[:4] * dt


    #  Kinematics queries                                                  
    def getDOF(self):
        return self.DOF

    def getJointAngles(self):
        """Returns a copy of [q1, q2, q3, q4] in radians."""
        return self.q.copy()

    def getJointPos(self, idx):
        """Return scalar angle of joint idx (0-based) in radians."""
        return float(self.q[idx])

    def getEEPosition(self):
        """
        3-D EE position in world_enu-aligned axes, relative to J1 (metres).
        Only q1, q2, q3 affect position; q4 is pure orientation (yaw only).
        Add J1's world_enu position from TF to get absolute world coordinates.
        """
        return swiftpro_fk_with_tf_transform(self.q, tf_buffer=self.tf_buffer)

    def getEEYaw(self):
        """
        Scalar EE yaw angle in world_enu (radians).
        yaw = q1 + q4: both joints rotate the EE about the vertical axis.
        The parallelogram constraint keeps pitch and roll fixed at all times.
        """
        return float(self.q[0]) + float(self.q[3])

    def getEEJacobianPos(self):
        """
        3X4 position Jacobian mapping dq -> linear EE velocity in
        world_enu-aligned axes.  4th column = [0,0,0] when L_EE_TOOL=0
        (q4 is a pure orientation DOF with no positional effect).
        """
        return swiftpro_jacobian_pos4_with_tf_transform(self.q, tf_buffer=self.tf_buffer)

    def getEEJacobianFull(self):
        """
        4X4 Jacobian mapping dq -> [dx, dy, dz, d(yaw)] in world_enu-aligned
        axes.  Use for combined position + EE yaw control tasks.
        """
        return swiftpro_jacobian_full4_with_tf_transform(self.q, tf_buffer=self.tf_buffer)



#  Task-Priority Control for VMS (Turtlebot + SwiftPro)
#
#  Quasi-velocity space: ζ = [vx, ω, dq1, dq2, dq3, dq4]  (6 DOF)
#
#  Joint index mapping used by VMSJointLimitsTask:
#    0 = vx   (base linear, no joint limit)
#    1 = ω    (base yaw rate / psi)
#    2 = q1   (arm base yaw)
#    3 = q2   (arm shoulder)
#    4 = q3   (arm elbow)
#    5 = q4   (EE yaw)

class VMSRobotState:
    """
    Lightweight state container for VMS task-priority control.

    Call update() each control tick before calling vms_task_priority_step().
    All Task subclasses below use this interface instead of a full robot model.
    """
    DOF = 6

    def __init__(self):
        self.ee_world  = np.zeros(3)
        self.base_pos  = np.zeros(2)
        self.base_psi  = 0.0
        self.arm_q     = np.zeros(4)   # [q1, q2, q3, q4]
        self.tf_buffer = None

    def update(self, ee_world, base_pos, base_psi, arm_q, tf_buffer=None):
        self.ee_world  = np.asarray(ee_world, dtype=float)
        self.base_pos  = np.asarray(base_pos, dtype=float)[:2]
        self.base_psi  = float(base_psi)
        self.arm_q     = np.asarray(arm_q,    dtype=float)
        self.tf_buffer = tf_buffer

    def update_from_joint_states(self, names, positions):
        """Parse a JointState message and update arm_q [q1,q2,q3,q4] in place.
        Joints are matched by name so message ordering doesn't matter."""
        pos_map = dict(zip(names, positions))
        for i, jn in enumerate(JOINT_NAMES_4DOF):
            if jn in pos_map:
                self.arm_q[i] = pos_map[jn]

    def getDOF(self):
        return self.DOF

    def getEEPosition(self):
        return self.ee_world.copy()

    def getEEYaw(self):
        # return self.base_psi + self.arm_q[0] + self.arm_q[3]
        return self.arm_q[3]

    def getEEJacobian(self):
        """3×6 position Jacobian — top 3 rows of the full 4×6 Jacobian."""
        return self._jacobian6()[:3, :]

    def getFullJacobian(self):
        """4×6 configuration Jacobian [position; yaw row]."""
        return self._jacobian6()

    def getJointPos(self, idx):
        """
        Returns the position of the DOF at quasi-velocity index idx.
          idx=1  → base_psi
          idx=2  → q1,  idx=3 → q2,  idx=4 → q3,  idx=5 → q4
        """
        if idx == 1:
            return self.base_psi
        elif 2 <= idx <= 5:
            return float(self.arm_q[idx - 2])
        return 0.0

    def _jacobian6(self):
        bw = np.array([self.base_pos[0], self.base_pos[1], 0.0])
        return swiftpro_jacobian_vms_6dof(
            self.ee_world, bw, self.base_psi, self.arm_q, self.tf_buffer)


# ---------------------------------------------------------------------------
#  Task base class
# ---------------------------------------------------------------------------

class Task:
    """Abstract base — same interface as swiftpro_robotics.Task."""
    def __init__(self, name, desired):
        self.name    = name
        self.sigma_d = np.asarray(desired, dtype=float)
        self.ff_vel  = np.zeros_like(self.sigma_d)
        self.K       = np.eye(self.sigma_d.shape[0])
        self.J       = None
        self.err     = None

    def update(self, _):        pass
    def setDesired(self, v):    self.sigma_d = np.asarray(v, dtype=float)
    def getDesired(self):       return self.sigma_d
    def getJacobian(self):      return self.J
    def getError(self):         return self.err
    def setFF(self, v):         self.ff_vel = np.asarray(v, dtype=float)
    def getFF(self):            return self.ff_vel
    def setGain(self, K):       self.K = K
    def getGain(self):          return self.K
    def isActive(self):         return True


# ---------------------------------------------------------------------------
#  Task subclasses for VMS

class VMSPositionTask(Task):
    """
    3-D EE position task in world_enu.
    Jacobian: 3×6 (position rows of swiftpro_jacobian_vms_6dof).
    desired: shape (3,1) — [px, py, pz]
    """
    def __init__(self, name, desired_pos):
        super().__init__(name, np.asarray(desired_pos, dtype=float).reshape(3, 1))
        self.J   = np.zeros((3, 6))
        self.err = np.zeros((3, 1))

    def update(self, state):
        p        = state.getEEPosition().reshape(3, 1)
        self.err = self.sigma_d - p
        self.J   = state.getEEJacobian()   # 3×6


class VMSYawTask(Task):
    """
    EE yaw task: σ = psi + q1 + q4.
    Jacobian row: [0, 1, 1, 0, 0, 1]  (1×6).
    desired: scalar yaw (rad)
    """
    def __init__(self, name, desired_yaw):
        super().__init__(name, np.array([[float(desired_yaw)]]))
        self.J   = np.zeros((1, 6))
        self.err = np.zeros((1, 1))

    def update(self, state):
        raw_err = float(self.sigma_d[0, 0]) - state.getEEYaw()
        raw_err = (raw_err + np.pi) % (2 * np.pi) - np.pi
        self.err = np.array([[raw_err]])
        self.J   = np.array([[0., 1., 1., 0., 0., 1.]])


class VMSConfigurationTask(Task):
    """
    Combined 3-D position + yaw task.
    Jacobian: 4×6 (full swiftpro_jacobian_vms_6dof).
    desired: shape (4,1) — [px, py, pz, yaw]
    """
    def __init__(self, name, desired):
        super().__init__(name, np.asarray(desired, dtype=float).reshape(4, 1))
        self.J   = np.zeros((4, 6))
        self.err = np.zeros((4, 1))

    def update(self, state):
        p       = state.getEEPosition().reshape(3, 1)
        pos_err = self.sigma_d[:3] - p

        raw_yaw_err = float(self.sigma_d[3, 0]) - state.getEEYaw()
        raw_yaw_err = (raw_yaw_err + np.pi) % (2 * np.pi) - np.pi

        self.err = np.vstack([pos_err, [[raw_yaw_err]]])
        self.J   = state.getFullJacobian()   # 4×6



class VMSJointLimitsTask(Task):
    """
    Joint limit avoidance for a single DOF in the 6-DOF quasi-velocity space.
    Activates when the joint enters the margin zone near a hard limit and
    commands a velocity that pushes it back toward the safe region.

    joint_idx: quasi-velocity index (2=q1, 3=q2, 4=q3, 5=q4)
    q_min/q_max: URDF hard limits
    margin: activation zone inside the hard limit (rad)
    """
    def __init__(self, name, joint_idx, q_min, q_max, margin=0.07, hysteresis_ratio=3.0):
        super().__init__(name, np.zeros((1, 1)))
        self.joint_idx        = joint_idx
        self.q_min            = q_min
        self.q_max            = q_max
        self.margin           = margin
        self.hysteresis_ratio = hysteresis_ratio  # δ = margin * ratio
        self._active    = False
        self._direction = 0
        self.current_q  = 0.0
        self.J   = np.zeros((1, 6))
        self.err = np.zeros((1, 1))

    def update(self, state):
        self.current_q = state.getJointPos(self.joint_idx)

        # alpha = margin (activation threshold)
        # delta = margin * hysteresis_ratio (deactivation threshold, ratio > 1 avoids chatter)
        delta = self.margin * self.hysteresis_ratio

        if not self._active:
            # Activate when joint enters the danger zone near a limit
            if self.current_q >= self.q_max - self.margin:
                self._direction = -1
                self._active    = True
            elif self.current_q <= self.q_min + self.margin:
                self._direction =  1
                self._active    = True
        else:
            # Deactivate only on the side that triggered (direction-specific, per slide)
            if self._direction == -1 and self.current_q <= self.q_max - delta:
                self._active    = False
                self._direction = 0
            elif self._direction == 1 and self.current_q >= self.q_min + delta:
                self._active    = False
                self._direction = 0

        if self._active:
            self.err = np.array([[float(self._direction)]])
            self.J   = np.zeros((1, 6))
            self.J[0, self.joint_idx] = 1.0
        else:
            self.err = np.array([[0.0]])
            self.J   = np.zeros((1, 6))

    def isActive(self):         return self._active
    def getCurrentPosition(self): return self.current_q


class VMSJointCenteringTask(Task):
    """
    Lowest-priority task: attract all arm joints toward their midpoints.

    In the null-space of every higher-priority task the solver uses this to
    gently re-center q1-q4. This prevents the configuration reached at goal N
    from being a bad starting point for goal N+1 — a pure reactive controller
    would otherwise be path-dependent because it has no memory or planning.

    Jacobian: 4×6, identity columns for [dq1, dq2, dq3, dq4] (indices 2-5).
    Error: q_center - q_current for each arm joint.
    """
    def __init__(self, name, q_centers):
        super().__init__(name, np.array(q_centers, dtype=float).reshape(4, 1))
        self.J   = np.zeros((4, 6))
        self.J[0, 2] = 1.0   # dq1
        self.J[1, 3] = 1.0   # dq2
        self.J[2, 4] = 1.0   # dq3
        self.J[3, 5] = 1.0   # dq4
        self.err = np.zeros((4, 1))

    def update(self, state):
        self.err = self.sigma_d - state.arm_q.reshape(4, 1)


class VMSQ4ZeroTask(Task):
    """
    Drive q4 directly to zero.

    Error:    0 − q4_current  (direct joint error, not EE yaw error)
    Jacobian: [0, 0, 0, 0, 0, 1]  — only dq4.

    Use this after VMSYawQ4Task has finished so the suction-cup yaw joint
    is returned to its neutral position before the next goal starts.
    Base and arm joints q1-q3 are untouched.
    """
    def __init__(self):
        super().__init__('q4_zero', np.zeros((1, 1)))
        self.J   = np.zeros((1, 6))
        self.J[0, 5] = 1.0   # dq4
        self.err = np.zeros((1, 1))

    def update(self, state):
        self.err = np.array([[-float(state.arm_q[3])]])   # 0 − q4_current


class VMSYawQ4Task(Task):
    """
    EE yaw task using ONLY q4 (joint4 = EE yaw joint, index 5 in dzeta).

    Run this AFTER the position task has converged. Because L_EE_TOOL = 0,
    q4 contributes [0, 0, 0] to EE position, so this task has exactly zero
    coupling with any position DOF. The base (vx, ω) and arm joints q1-q3
    are never touched.

    Jacobian row: [0, 0, 0, 0, 0, 1]  — only dq4.
    Error: wrap(target_yaw − (psi + q1 + q4)).

    Limitation: q4 ∈ [−π/2, π/2], so the achievable yaw offset from the
    current (psi + q1) is at most ±1.57 rad. Check that
    |target_yaw − psi_final − q1_final| < Q4_MAX before relying on this.
    """
    def __init__(self, name, desired_yaw=0.0):
        super().__init__(name, np.array([[float(desired_yaw)]]))
        self.J   = np.zeros((1, 6))
        self.J[0, 5] = 1.0   # index 5 = dq4
        self.err = np.zeros((1, 1))

    def update(self, state):
        raw_err = float(self.sigma_d[0, 0]) - state.getEEYaw()
        raw_err = (raw_err + np.pi) % (2 * np.pi) - np.pi
        self.err = np.array([[raw_err]])
        # Jacobian is constant — only q4


class VMSJointPositionTask(Task):
    """
    Joint-space position task for a single arm joint in the 6-DOF VMS.
    Jacobian: 1×6 row with a single 1.0 at the joint's quasi-velocity column.
    desired: scalar joint angle (rad)
    arm_joint_idx: 0=q1, 1=q2, 2=q3, 3=q4  (maps to zeta cols 2–5)
    """
    def __init__(self, name, desired_angle, arm_joint_idx):
        super().__init__(name, np.array([[float(desired_angle)]]))
        self.joint_idx = arm_joint_idx + 2   # zeta col: q1→2, q2→3, q3→4, q4→5
        self.J   = np.zeros((1, 6))
        self.J[0, self.joint_idx] = 1.0      # single-entry row Jacobian
        self.err = np.zeros((1, 1))

    def update(self, state):
        current_q = state.getJointPos(self.joint_idx)
        self.err  = self.sigma_d - np.array([[current_q]])


class VMSBaseOrientationTask(Task):
    """
    Base yaw (heading) task for the 6-DOF VMS.

    Drives the robot base to a desired heading angle ψ_d by commanding the
    angular quasi-velocity ω (index 1 in the 6-DOF dzeta vector).

    σ   = ψ  (base yaw in world_enu, rad)
    J   = [0, 1, 0, 0, 0, 0]  — only ω contributes to ψ̇
    err = wrap(ψ_d − ψ_current)  (wrapped to [-π, π] to take the short arc)

    Use this as a high-priority task before a position task when the robot
    needs to face a specific direction first, or as a null-space task to
    maintain a preferred heading while the EE tracks a goal.

    Parameters
    ----------
    name         : task label (string)
    desired_yaw  : target base heading (rad)
    """
    def __init__(self, name, desired_yaw=0.0):
        super().__init__(name, np.array([[float(desired_yaw)]]))
        self.J   = np.zeros((1, 6))
        self.J[0, 1] = 1.0   # index 1 = ω (base angular quasi-velocity)
        self.err = np.zeros((1, 1))

    def update(self, state):
        raw_err = float(self.sigma_d[0, 0]) - state.base_psi
        raw_err = (raw_err + np.pi) % (2 * np.pi) - np.pi   # wrap to [-π, π]
        self.err = np.array([[raw_err]])
        # Jacobian is constant — ω is the only DOF that changes ψ


class ArmJointPositionTask(Task):
    """
    Joint-space position task for a single arm joint — ARM-ONLY (base fixed).

    Designed to work with SwiftProManipulator4DOF (and any object with a .q
    array of shape (4,)).  The Jacobian is 1×4 so it plugs into a 4-DOF
    arm-only solver, NOT the 6-DOF VMS solver.

    For the VMS equivalent use VMSJointPositionTask (1×6 Jacobian).

    Mirrors the JointPosition task from lab4_robotics.py, adapted for the
    uArm Swift Pro joint naming.

    σ   = q_i  (arm joint angle, rad)
    J   = [0, …, 1, …, 0]  — 1×4 row with 1.0 at arm_joint_idx
    err = q_i_desired − q_i_current

    Parameters
    ----------
    name           : task label (string)
    desired_angle  : target joint angle (rad)
    arm_joint_idx  : 0=q1, 1=q2, 2=q3, 3=q4
    """
    def __init__(self, name, desired_angle, arm_joint_idx):
        super().__init__(name, np.array([[float(desired_angle)]]))
        self.arm_joint_idx = arm_joint_idx
        self.J   = np.zeros((1, 4))
        self.J[0, arm_joint_idx] = 1.0   # single-entry row Jacobian (1×4)
        self.err = np.zeros((1, 1))

    def update(self, arm):
        """
        arm : SwiftProManipulator4DOF  (must have .q array of shape (4,))
        """
        current_q = float(arm.q[self.arm_joint_idx])
        self.err  = self.sigma_d - np.array([[current_q]])
        # Jacobian is constant — does not change with configuration


# ---------------------------------------------------------------------------
#  Recursive Task-Priority solver for VMS

def vms_task_priority_step(tasks, state, damping=0.1, method=2):
    """
    One step of the recursive Task-Priority algorithm for the VMS.

    Processes tasks from highest to lowest priority.  Each task operates
    in the null-space of all higher-priority tasks, so higher-priority tasks
    are never disturbed by lower-priority ones.

    Arguments
    ---------
    tasks   : list[Task]       — ordered highest → lowest priority
    state   : VMSRobotState    — current robot state (call update() first)
    damping : float            — DLS λ (only used when method=2)
    method  : int              — inverse method for velocity update:
                                   0 = Jacobian transpose  (J^T, no inversion)
                                   1 = Moore-Penrose pseudo-inverse  (np.linalg.pinv)
                                   2 = Damped Least-Squares (default, stable near singularities)

    Returns
    -------
    zeta : np.ndarray, shape (6,1)
        Joint velocity command [vx, ω, dq1, dq2, dq3, dq4].
    """
    n    = state.getDOF()   # 6
    P    = np.eye(n)
    zeta = np.zeros((n, 1))

    for task in tasks:
        task.update(state)

        if not task.isActive():
            continue

        err_i  = task.getError()
        Ji     = task.getJacobian()
        Ki     = task.getGain()
        ff_i   = task.getFF()
        xi_dot = Ki @ err_i + ff_i

        Ji_bar = Ji @ P

        # Compute the velocity-update inverse according to the chosen method.
        if method == 0:
            Ji_bar_inv = Ji_bar.T                    # Jacobian transpose
        elif method == 1:
            Ji_bar_inv = np.linalg.pinv(Ji_bar)      # Moore-Penrose pseudo-inverse
        else:
            Ji_bar_inv = DLS(Ji_bar, damping)        # Damped Least-Squares (default)

        zeta = zeta + Ji_bar_inv @ (xi_dot - Ji @ zeta)

        # Null-space projector always uses Moore-Penrose pinv for geometric correctness.
        Ji_bar_pinv = np.linalg.pinv(Ji_bar)
        P = P - Ji_bar_pinv @ Ji_bar

    return zeta


class VMSObstacleTask(Task):
    """
    3D spherical obstacle avoidance task for the VMS (mobile manipulator).

    Mirrors the Obstacle2D inequality task from lab5 but operates in 3D
    world_enu and uses the full 6-DOF VMS Jacobian, so avoidance is achieved
    through any combination of base motion (vx, ω) and arm joints (dq1-dq4).

    Obstacle is modelled as a sphere centred at obs_pos_world with radius r_a.
    Hysteresis (r_d > r_a) prevents rapid on/off chattering at the boundary.

    Activation function (same pattern as JointLimits):
      - Activates when   distance(EE, obs) <= r_a
      - Deactivates when distance(EE, obs) >= r_d   (r_d > r_a required)

    Error:    r_a − distance  (positive inside the exclusion sphere)
    Jacobian: direction.T @ J_pos_3x6   →  1X6 row
              where direction = unit vector from obstacle centre to EE.

    To use cylindrical obstacles (ignore height): pass a 2D obs_pos [x, y]
    and set cylindrical=True — the task then uses only the horizontal distance.
    """

    def __init__(self, name, obs_pos_world, r_a, r_d, cylindrical=False):
        """
        Parameters
        ----------
        name          : str   — task name shown in logs
        obs_pos_world : (2,) or (3,) array — obstacle centre in world_enu
        r_a           : float — activation radius (exclusion zone boundary)
        r_d           : float — deactivation radius, must be > r_a
        cylindrical   : bool  — if True, ignore Z (horizontal distance only)
        """
        super().__init__(name, np.zeros((1, 1)))
        self.obs_pos    = np.array(obs_pos_world, dtype=float)
        self.r_a        = float(r_a)
        self.r_d        = float(r_d)
        self.cylindrical = cylindrical
        self._active    = False
        self.distance   = float('inf')
        self.J          = np.zeros((1, 6))
        self.err        = np.zeros((1, 1))

    def update(self, state):
        ee = state.ee_world   # (3,) current EE position in world_enu

        if self.cylindrical:
            # Horizontal distance only — useful for floor-standing obstacles
            displacement = ee[:2] - self.obs_pos[:2]
        else:
            displacement = ee - self.obs_pos[:3]

        self.distance = float(np.linalg.norm(displacement))

        # Hysteresis activation / deactivation
        if not self._active:
            if self.distance <= self.r_a:
                self._active = True
        else:
            if self.distance >= self.r_d:
                self._active = False

        if self._active and self.distance > 1e-6:
            direction = displacement / self.distance   # unit vector obstacle -> EE

            J_pos = state.getEEJacobian()   # 3 X 6 VMS position Jacobian

            if self.cylindrical:
                # Map 2D horizontal direction to the x/y rows of J_pos
                self.J = direction.reshape(1, 2) @ J_pos[:2, :]   # 1 X 6
            else:
                self.J = direction.reshape(1, 3) @ J_pos           # 1 X 6

            self.err = np.array([[self.r_a - self.distance]])
        else:
            self.J   = np.zeros((1, 6))
            self.err = np.array([[0.0]])

    def isActive(self):
        return self._active

    def getDistance(self):
        return self.distance