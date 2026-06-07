import math
from launch import LaunchDescription
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    pkg_tb_desc = FindPackageShare('turtlebot_description')

    return LaunchDescription([
        # odom → world_enu (identity): real robot only has odom, code expects world_enu
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='odom_to_world_enu_broadcaster',
            arguments=['0.0', '0.0', '0.0', '0.0', '0.0', '0.0', 'odom', 'world_enu'],
        ),

        # world_enu → world_ned (same rotation as Stonefish simulation)
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='enu_to_ned_broadcaster',
            arguments=['0.0', '0.0', '0.0',
                       str(math.pi / 2), '0.0', str(math.pi),
                       'world_enu', 'world_ned'],
        ),

        # End-effector static TF (link8B → end_effector)
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='ee_broadcaster',
            arguments=[
                '0.0', '0.0', '0.0722',
                '0.0', '0.0', str(math.pi),
                'swiftpro/link8B',
                'end_effector',
            ],
        ),

        # Pick-and-place VMS node
        Node(
            package='hoi_control',
            executable='lab2_pick_place_vms_node.py',
            name='pick_place_vms',
            output='screen',
            emulate_tty=True,
        ),

        # RViz
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', PathJoinSubstitution([pkg_tb_desc, 'rviz', 'turtlebot.rviz'])],
        ),
    ])
