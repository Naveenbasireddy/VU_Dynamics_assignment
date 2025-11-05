cmake_minimum_required(VERSION 3.0.2)
project(swarm_sim)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  geometry_msgs
  sensor_msgs
  nav_msgs
  tf
  gazebo_ros
  cv_bridge
)

catkin_package()
include_directories(${catkin_INCLUDE_DIRS})
