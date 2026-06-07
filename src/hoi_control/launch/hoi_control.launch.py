"""
hoi_control.launch.py
=====================
Launches the turtlebot_hoi simulation AND one selected control node.

Usage:
    ros2 launch hoi_control hoi_control.launch.py node:=lab2_kinematics
    ros2 launch hoi_control hoi_control.launch.py node:=lab2_rrc
    ros2 launch hoi_control hoi_control.launch.py node:=lab3_two_tasks
    ros2 launch hoi_control hoi_control.launch.py node:=lab3_null_space
    ros2 launch hoi_control hoi_control.launch.py node:=lab4_tp
    ros2 launch hoi_control hoi_control.launch.py node:=lab5_joint_limits
    ros2 launch hoi_control hoi_control.launch.py node:=lab5_obstacle
    ros2 launch hoi_control hoi_control.launch.py node:=lab6_mobile_manip
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


# Map of node names to their executable names in this package
NODE_MAP = {
    'lab2_kinematics':   'lab2_kinematics_node.py',
    'lab2_rrc':          'lab2_rrc_node.py',
    'lab3_two_tasks':    'lab3_two_tasks_node.py',
    'lab3_null_space':   'lab3_null_space_node.py',
    'lab4_tp':           'lab4_tp_node.py',
    'lab5_joint_limits': 'lab5_joint_limits_node.py',
    'lab5_obstacle':     'lab5_obstacle_node.py',
    'lab6_mobile_manip': 'lab6_mobile_manip_node.py',
}


def generate_launch_description():
    pkg_sim  = FindPackageShare('turtlebot_simulation')
    pkg_ctrl = FindPackageShare('hoi_control')

    declare_node = DeclareLaunchArgument(
        'node',
        default_value='lab4_tp',
        description='Which control node to launch. Options: ' + ', '.join(NODE_MAP.keys())
    )

    declare_hierarchy = DeclareLaunchArgument(
        'hierarchy', default_value='a',
        description='Task hierarchy for lab4_tp (a/b/c/d)')

    declare_case = DeclareLaunchArgument(
        'case', default_value='b',
        description='Priority case for lab3_two_tasks (a/b)')

    # --- Include the simulation launch (same as turtlebot_hoi.launch.py) ---
    sim_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([pkg_sim, 'launch', 'turtlebot_basic.launch.py'])
        ]),
        launch_arguments={
            'scenario_description': PathJoinSubstitution(
                [pkg_sim, 'scenarios', 'turtlebot_hoi.scn'])
        }.items()
    )

    # --- Control node (selected by 'node' argument) ---
    # NOTE: Because ROS 2 launch substitutions are lazy, we launch all nodes
    # and gate them with a condition based on the argument.
    # A cleaner pattern uses OpaqueFunction – used here for simplicity.

    from launch.actions import OpaqueFunction

    def launch_control_node(context, *args, **kwargs):
        node_name = LaunchConfiguration('node').perform(context)
        hierarchy = LaunchConfiguration('hierarchy').perform(context)
        case      = LaunchConfiguration('case').perform(context)

        if node_name not in NODE_MAP:
            raise ValueError(
                f"Unknown node '{node_name}'. Choose from: {list(NODE_MAP.keys())}")

        params = {}
        if node_name == 'lab4_tp':
            params['hierarchy'] = hierarchy
        if node_name == 'lab3_two_tasks':
            params['case'] = case

        ctrl_node = Node(
            package='hoi_control',
            executable=NODE_MAP[node_name],
            name=node_name,
            output='screen',
            parameters=[params] if params else []
        )
        return [ctrl_node]

    return LaunchDescription([
        declare_node,
        declare_hierarchy,
        declare_case,
        sim_launch,
        OpaqueFunction(function=launch_control_node),
    ])
