from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    args = [
        DeclareLaunchArgument('target_x', default_value='0.0'),
        DeclareLaunchArgument('target_y', default_value='0.159'),
        DeclareLaunchArgument('target_z', default_value='0.40'),
        DeclareLaunchArgument('gain_k',   default_value='0.5'),
        DeclareLaunchArgument('max_vel',  default_value='0.3'),
        DeclareLaunchArgument('damping',  default_value='0.1'),
        DeclareLaunchArgument('method',   default_value='2'),
        DeclareLaunchArgument('enabled',  default_value='true'),
    ]

    debug_node = Node(
        package='hoi_control',
        executable='lab2_rrc_debug_node.py',
        name='lab2_rrc_debug',
        output='screen',
        parameters=[{
            'target_x':    LaunchConfiguration('target_x'),
            'target_y':    LaunchConfiguration('target_y'),
            'target_z':    LaunchConfiguration('target_z'),
            'gain_k':      LaunchConfiguration('gain_k'),
            'max_vel':     LaunchConfiguration('max_vel'),
            'damping':     LaunchConfiguration('damping'),
            'method':      LaunchConfiguration('method'),
            'enabled':     LaunchConfiguration('enabled'),
            'manual_mode': False,
            'manual_dq1':  0.0,
            'manual_dq2':  0.0,
            'manual_dq3':  0.0,
        }],
    )

    return LaunchDescription(args + [debug_node])
