cmake_minimum_required(VERSION 3.0.2)
project(swarm_simulation)
find_package(catkin REQUIRED COMPONENTS rospy std_msgs geometry_msgs nav_msgs gazebo_ros)
catkin_package()
catkin_install_python(PROGRAMS
  scripts/detection_node.py
  scripts/swarm_manager.py
  scripts/formation_controller.py
  scripts/drone_bridge.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
