cmake_minimum_required(VERSION 3.8)
project(hoi_control)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)

# Install the Python package (all scripts under hoi_control/)
ament_python_install_package(${PROJECT_NAME})

# Install each node script so it is executable via ros2 run
install(PROGRAMS
  hoi_control/lab2_kinematics_node.py
  hoi_control/lab2_rrc_node.py
  hoi_control/lab2_rrc_debug_node.py
  hoi_control/lab2_rrc_methods_debug_node.py
  hoi_control/lab2_rrc_methods_vms_node.py
  hoi_control/lab2_rrc_methods_vms_config_node.py
  hoi_control/lab2_pick_place_vms_node.py
  hoi_control/lab2_pick_place_vms_node_2.py
  hoi_control/lab2_integrated_mission_node.py
  hoi_control/aruco_scout_node.py
  hoi_control/aruco_camera_node.py
  hoi_control/mission_coordinator_node.py
  hoi_control/lab3_two_tasks_node.py
  hoi_control/lab3_null_space_node.py
  hoi_control/lab4_tp_node.py
  hoi_control/lab5_joint_limits_node.py
  hoi_control/lab5_obstacle_node.py
  hoi_control/lab6_mobile_manip_node.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch files
install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

ament_package()
