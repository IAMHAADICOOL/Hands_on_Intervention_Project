"""
integrated_mission.launch.py
Launches the complete integrated frontier-exploration + pick-and-place mission.

Launch sequence:
  1. Stonefish simulation + all exploration nodes (from exploration_sim.launch.py)
  2. ArUco scout node — passive marker detection during exploration
  3. Integrated pick-place node — waits in IDLE until coordinator activates it
  4. Mission coordinator node — orchestrates exploration → pick → place

Pass place_dest_x / place_dest_y to override the default place location:
  ros2 launch hoi_control integrated_mission.launch.py place_dest_x:=2.0 place_dest_y:=4.0
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def xterm(title):
    return f'xterm -title "{title}" -geometry 120x30 -e'


def generate_launch_description():

    # ── Launch arguments ───────────────────────────────────────────────────────
    declare_place_x = DeclareLaunchArgument(
        'place_dest_x', default_value='0.0',
        description='world_enu X of the place destination (m)')
    declare_place_y = DeclareLaunchArgument(
        'place_dest_y', default_value='3.0',
        description='world_enu Y of the place destination (m)')
    declare_standoff = DeclareLaunchArgument(
        'standoff_dist', default_value='0.80',
        description='Stand-off distance from marker for approach (m)')
    declare_nav_radius = DeclareLaunchArgument(
        'nav_arrival_radius', default_value='0.20',
        description='Base position tolerance to declare navigation arrival (m)')

    place_dest_x      = LaunchConfiguration('place_dest_x')
    place_dest_y      = LaunchConfiguration('place_dest_y')
    standoff_dist     = LaunchConfiguration('standoff_dist')
    nav_arrival_radius = LaunchConfiguration('nav_arrival_radius')

    # ── Sub-launch: exploration stack ──────────────────────────────────────────
    # Includes: Stonefish sim, RTAB-Map, EKF, image_crop, arm_retract,
    #           global_costmap, DWA local costmap, DWA service,
    #           frontier_node, path_planner_tb
    exploration_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('online_motion_planning'),
                'launch', 'exploration_sim.launch.py'
            )
        )
    )

    # ── Standalone camera overlay (optional — omit if using integrated_mission_node's window) ──
    # aruco_camera_node = Node(
    #     package='hoi_control',
    #     executable='aruco_camera_node.py',
    #     name='aruco_camera_node',
    #     output='screen',
    #     prefix=xterm('ArucoCamera'),
    #     parameters=[{
    #         'marker_id':    1,
    #         'marker_size':  0.050,
    #         'camera_topic': '/turtlebot/camera/color/image_color',
    #     }],
    # )

    # ── ArUco scout node ───────────────────────────────────────────────────────
    # Runs throughout the mission, updates /aruco/best_pose whenever marker seen.
    aruco_scout_node = Node(
        package='hoi_control',
        executable='aruco_scout_node.py',
        name='aruco_scout_node',
        output='screen',
        prefix=xterm('ArucoScout'),
        parameters=[{
            'marker_id':    71,
            'marker_size':  0.050,
            'camera_topic': '/turtlebot/camera/color/image_color',
            'world_frame':  'world_enu',
            'camera_frame': 'camera_color_optical_frame',
        }],
    )

    # ── Integrated pick-place node ─────────────────────────────────────────────
    # Starts in IDLE.  The coordinator calls /pick_place/start_pick when ready.
    integrated_mission_node = Node(
        package='hoi_control',
        executable='lab2_integrated_mission_node.py',
        name='integrated_mission_node',
        output='screen',
        prefix=xterm('PickPlace'),
    )

    # ── Mission coordinator node ───────────────────────────────────────────────
    # Orchestrates exploration → NAV_TO_STANDOFF → PICK → NAV_TO_PLACE → PLACE.
    mission_coordinator_node = Node(
        package='hoi_control',
        executable='mission_coordinator_node.py',
        name='mission_coordinator_node',
        output='screen',
        prefix=xterm('Coordinator'),
        parameters=[{
            'place_dest_x':       place_dest_x,
            'place_dest_y':       place_dest_y,
            'standoff_dist':      standoff_dist,
            'nav_arrival_radius': nav_arrival_radius,
            'nav_timeout_sec':    60.0,
            'pick_timeout_sec':  120.0,
            'place_timeout_sec': 120.0,
        }],
    )

    # ── Timing ────────────────────────────────────────────────────────────────
    # Mission nodes start 5 s after exploration stack so TF is ready.
    mission_nodes = TimerAction(
        period=5.0,
        actions=[
            aruco_scout_node,
            integrated_mission_node,
            mission_coordinator_node,
        ]
    )

    return LaunchDescription([
        declare_place_x,
        declare_place_y,
        declare_standoff,
        declare_nav_radius,
        exploration_launch,
        mission_nodes,
    ])
