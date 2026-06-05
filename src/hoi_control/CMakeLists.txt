cmake_minimum_required(VERSION 3.8)
project(hoi_control)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)

# Install the Python package (all scripts under hoi_control/)
ament_python_install_package(${PROJECT_NAME})

# Install each node script so it is executable via ros2 run
install(PROGRAMS
  hoi_control/lab2_rrc_node.py
  hoi_control/lab2_rrc_debug_node.py
  hoi_control/lab2_rrc_methods_debug_node.py
  hoi_control/lab2_rrc_methods_vms_node.py
  hoi_control/lab2_pick_place_vms_node.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
